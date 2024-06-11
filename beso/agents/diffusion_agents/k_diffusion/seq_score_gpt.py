import logging
import math 
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
import einops

from transformer_blocks.utils import RMSNorm
from transformer_blocks.moe_models import TransformerFiLMMoE
from transformer_blocks.transformer_decoders import TransformerFiLMDecoder

from .transformer_utils import *

logger = logging.getLogger(__name__)
    

class StepsDiffusionGPT(nn.Module):
    """the full GPT score model, with a context size of block_size"""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        device: str,
        goal_conditioned: bool,
        action_dim: int,
        embed_dim: int,
        embed_pdrob: float,
        attn_pdrop: float,
        resid_pdrop: float,
        n_layers: int,
        n_heads: int,
        goal_seq_len: int,
        obs_seq_len: int,
        action_seq_len: int,
        sigma_vocab_size: int,
        goal_drop: float = 0,
        linear_output: bool = True,
        use_rel_pos_embed: bool = False,
        use_abs_pos_embed: bool = True,
        use_moe_model: bool = False,
    ):
        super().__init__()
        self.device = device
        self.use_moe_model = use_moe_model
        self.goal_conditioned = goal_conditioned
        self.use_rel_pos_emb = use_rel_pos_embed
        self.use_abs_pos_emb = use_abs_pos_embed
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        self.action_seq_len = action_seq_len
        
        seq_size = goal_seq_len + obs_seq_len - 1 + action_seq_len
        self.tok_emb = nn.Linear(obs_dim, embed_dim)
        self.goal_emb = nn.Linear(goal_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop
        
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        # transformer
        if self.use_moe_model:
            self.blocks = TransformerFiLMMoE(
                embed_dim=embed_dim,
                n_heads=n_heads,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                n_layers=n_layers,
                film_cond_dim=embed_dim,
                use_rms_norm=False,
                use_cross_attention=False,
                block_size=block_size,
                bias=False,
                causal=True,
                use_rot_embed=False,
                rotary_xpos=False,
                mlp_pdrop=0.1,
                use_noise_encoder=True,
                cond_router=True,
                num_experts=4,
                top_k=2,
                router_normalize=True,
            )
        else:
            self.blocks = nn.Sequential(
                *[Block(
                    embed_dim,
                    n_heads,
                    attn_pdrop,
                    resid_pdrop,
                    block_size,
                    use_rel_pos_embed
                ) for _ in range(n_layers)]
            )
        # decoder head
        self.ln_f = nn.LayerNorm(embed_dim) if not self.use_moe_model else RMSNorm(embed_dim)
        
        self.block_size = block_size
        
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        # we need another embedding for the sigma
        self.sigma_emb = nn.Linear(1, embed_dim) 
        # get an action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)
        # action pred module 
        if linear_output:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100), 
                nn.GELU(),  
                nn.Linear(100, self.action_dim)
            )
        else:
            self.action_pred = nn.Linear(embed_dim, self.action_dim)
        
        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, StepsDiffusionGPT):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, RMSNorm)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        
        return optim_groups
    
    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(
        self, 
        states,
        actions, 
        goals,
        sigma,
        uncond: Optional[bool] =False,
        keep_last_actions: Optional[bool] = False
    ):  
        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # get the sigma embedding
        sigmas = sigma.log() / 4
        sigmas = einops.rearrange(sigmas, 'b -> b 1')
        emb_t = self.sigma_emb(sigmas.to(torch.float32))
        if len(states.shape) == 3:
            emb_t = einops.rearrange(emb_t, 'b d -> b 1 d')
        
        # reshape goals 
        if len(goals.shape) == 2:
            goals = einops.rearrange(goals, 'b d -> b 1 d')
        if goals.shape[1] == states.shape[1] and self.goal_seq_len == 1:
            goals = goals[:, 0, :]
            goals = einops.rearrange(goals, 'b d -> b 1 d')
        # define the total length of the input sequence       
        if self.goal_conditioned:
            second_half_idx = self.goal_seq_len + t + 1
        else:
            second_half_idx = t + 1
        # during training randomly mask out the goal
        # to train the conditional model with classifier-free guidance wen need 
        # to 0 out some of the conditional during training with a desrired probability
        # it is usually in the range of 0,1 to 0.2 
        if self.training:
            goals = self.mask_cond(goals)
        # we want to use unconditional sampling during clasisfier free guidance
        if uncond:
            goals = torch.zeros_like(goals).to(self.device)  
        
        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        goal_embed = self.tok_emb(goals)
        action_embed = self.action_emb(actions.to(torch.float32))
        
        # if not uncond:
        if self.goal_conditioned:
            position_embeddings = self.pos_emb[
            :, :(t + self.goal_seq_len + self.action_seq_len - 1), :
            ]  # each position maps to a (learnable) vector
        else: # without goal conditioning we only have the obs sequence 
            position_embeddings = self.pos_emb[
            :, :t, :
            ]
        # note, that the goal states are at the beginning of the sequence since they are available 
        # for all states s_1, .., s_t otherwise the masking would not make sense
        if self.use_abs_pos_emb:
            position_embeddings = self.pos_emb[:, :(t + self.goal_seq_len + self.action_seq_len - 1), :]
            goal_x = self.drop(goal_embed + position_embeddings[:, :self.goal_seq_len, :])
            state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])
            action_x = self.drop(action_embed + position_embeddings[:, (self.goal_seq_len + t - 1):, :])
        else:
            goal_x = self.drop(goal_embed)
            state_x = self.drop(state_embed)
            action_x = self.drop(action_embed)
        # next we stack everything together 
        if self.goal_conditioned:
            input_seq = torch.cat([emb_t, goal_x, state_x, action_x], dim=1)
        else:
            input_seq = torch.cat([emb_t, state_x, action_x], dim=1)
        
        # Note we need to also adept the action masks 
        if self.use_moe_model:
            x = self.blocks(input_seq, emb_t)
        else:
            x = self.blocks(input_seq)
            x = self.ln_f(x)
        
        # now we want the last half of the output      
        action_outputs =x[:, second_half_idx:, :]
        pred_actions = self.action_pred(action_outputs)
        '''if not self.training and len(input_seq) == 1:
            pred_actions = pred_actions[0, 0] # only get the first action of the sequence 
            pred_actions = einops.rearrange(pred_actions, 'b -> 1 b')'''
        return pred_actions
    
    def mask_cond(self, cond, force_mask=False):
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            # TODO Check which one is correct
            mask = torch.bernoulli(torch.ones((bs, t, d), device=cond.device) * self.cond_mask_prob) # .view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            # mask = torch.bernoulli(torch.ones((bs, t, 1), device=cond.device) * self.cond_mask_prob)
            # mask = einops.repeat(mask, 'b t 1 -> b t (1 d)', d=d)
            return cond * (1. - mask)
        else:
            return cond

    def get_params(self):
        return self.parameters()


# TODO Not debugged yet!
class TwoCamsStepsDiffusionGPT(StepsDiffusionGPT):
    """the full GPT score model, with a context size of block_size"""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        device: str,
        goal_conditioned: bool,
        action_dim: int,
        embed_dim: int,
        embed_pdrob: float,
        attn_pdrop: float,
        resid_pdrop: float,
        n_layers: int,
        n_heads: int,
        goal_seq_len: int,
        obs_seq_len: int,
        action_seq_len: int,
        sigma_vocab_size: int,
        goal_drop: float = 0.1,
        linear_output: bool = True,
        use_rel_pos_embed: bool = False,
        use_abs_pos_embed: bool = True,
    ):
        # Call the parent class's __init__ method
        super(TwoCamsStepsDiffusionGPT, self).__init__(
            obs_dim=obs_dim,
            goal_dim=goal_dim,
            device=device,
            goal_conditioned=goal_conditioned,
            action_dim=action_dim,
            embed_dim=embed_dim,
            embed_pdrob=embed_pdrob,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            n_layers=n_layers,
            n_heads=n_heads,
            goal_seq_len=goal_seq_len,
            obs_seq_len=obs_seq_len,
            action_seq_len=action_seq_len,
            sigma_vocab_size=sigma_vocab_size,
            goal_drop=goal_drop,
            linear_output=linear_output,
            use_rel_pos_embed=use_rel_pos_embed,
            use_abs_pos_embed=use_abs_pos_embed
        )
        self.incam_embed = nn.Linear(self.obs_dim, self.embed_dim)
        # overwrite sequence
        seq_size = goal_seq_len + obs_seq_len - 1 + action_seq_len
        self.pos_emb = nn.Parameter(torch.zeros(1, 2*obs_seq_len, embed_dim))
        # overwrite the blocks 
        self.blocks = nn.Sequential(
            *[Block(
                embed_dim,
                n_heads,
                attn_pdrop,
                resid_pdrop,
                goal_seq_len +1 + obs_seq_len*2 +action_seq_len
            ) for _ in range(n_layers)]
        )
        # get another position Embedding for the position inside the sequence
        # self.seq_pos_embed = nn.Parameter(torch.zeros(1, 2*seq_size, embed_dim))
        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
    
    def forward(
        self, 
        states,
        actions, 
        goals,
        sigma,
        uncond: Optional[bool] =False,
    ):  
        b, t, dim = states.size()
        # we have two cam images
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # get the sigma embedding
        sigmas = sigma.log() / 4
        sigmas = einops.rearrange(sigmas, 'b -> b 1')
        emb_t = self.sigma_emb(sigmas)
        if len(states.shape) == 3:
            emb_t = einops.rearrange(emb_t, 'b d -> b 1 d')
        
        # reshape goals 
        if len(goals.shape) == 2:
            goals = einops.rearrange(goals, 'b d -> b 1 d')
        if goals.shape[1] == states.shape[1] and self.goal_seq_len == 1:
            goals = goals[:, 0, :]
            goals = einops.rearrange(goals, 'b d -> b 1 d')
        # define the total length of the input sequence       
        if self.goal_conditioned:
            second_half_idx = self.goal_seq_len + t*2 + 1
        else:
            second_half_idx = t + 1
        # during training randomly mask out the goal
        # to train the conditional model with classifier-free guidance wen need 
        # to 0 out some of the conditional during training with a desrired probability
        # it is usually in the range of 0,1 to 0.2 
        if goals.shape[-1] == 2 * self.obs_dim:
            goals = goals[:, :, :self.obs_dim]
        if self.training:
            goals = self.mask_cond(goals)
        # we want to use unconditional sampling during clasisfier free guidance
        if uncond:
            goals = torch.zeros_like(goals).to(self.device)  
        
        # embed them into linear representations for the transformer
        states_global = self.tok_emb (states[:, :, :self.obs_dim])
        incam_states = self.incam_embed(states[:, :, self.obs_dim:])
        
        state_embed = torch.stack((states_global, incam_states), dim=2).reshape(b, 2*t, self.embed_dim)
        state_embed += self.pos_emb[:, :, :]
        goal_embed = self.tok_emb(goals)
        action_embed = self.action_emb(actions)
        
        # if not uncond:
        if self.goal_conditioned:
            position_embeddings = self.pos_emb[
            :, :(t + self.goal_seq_len + self.action_seq_len - 1), :
            ]  # each position maps to a (learnable) vector
        else: # without goal conditioning we only have the obs sequence 
            position_embeddings = self.pos_emb[
            :, :t, :
            ]
        # note, that the goal states are at the beginning of the sequence since they are available 
        # for all states s_1, .., s_t otherwise the masking would not make sense
        goal_x = self.drop(goal_embed + position_embeddings[:, :self.goal_seq_len, :])
        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len+t), :])
        # the action get the same position embedding as the related states 
        action_x = self.drop(action_embed + position_embeddings[:, (self.goal_seq_len+t-1):, :])
        
        # next we stack everything together 
        if self.goal_conditioned:
            input_seq = torch.cat([emb_t, goal_x, state_x, action_x], dim=1)
        else:
            input_seq = torch.cat([emb_t, state_x, action_x], dim=1)
        
        # Note we need to also adept the action masks 
        x = self.blocks(input_seq)
        x = self.ln_f(x)
        
        # now we want the last half of the output      
        action_outputs =x[:, second_half_idx:, :]
        pred_actions = self.action_pred(action_outputs)
        '''if not self.training and len(input_seq) == 1:
            pred_actions = pred_actions[0, 0] # only get the first action of the sequence 
            pred_actions = einops.rearrange(pred_actions, 'b -> 1 b')'''
        return pred_actions
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        # no_decay.add("seq_pos_embed")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        
        return optim_groups