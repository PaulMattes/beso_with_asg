import torch
from torch import nn
from torchvision import models
from einops import einops

class ResNet(nn.Module):
    def __init__(self, model_name, obs_dim, device) -> None:
        super(ResNet, self).__init__()
        if model_name == "ResNet18":
            self.model = models.resnet18(pretrained=False).to(device)
        elif model_name == "ResNet50":
            self.model = models.resnet50(pretrained=False).to(device)
        
        n_inputs = self.model.fc.in_features    
        
        self.fc_layer = nn.Linear(n_inputs, obs_dim).to(device)
        
        self.model.fc = self.fc_layer
        
        self.state_modality = 'observation'
        self.goal_modality = 'goal_observation'
        self.device = device
         
    def forward(self, input):
        obs = input[self.state_modality].to(self.device)
        goal = input[self.goal_modality].to(self.device) if self.goal_modality in input else None
        
        x = self.single_forward(obs, obs.shape[1])
        with torch.no_grad():
            y = self.single_forward(goal, goal.shape[1])
        
        return x, y
            
    def single_forward(self, x, ws):
        org_len = len(x.shape)
        
        if org_len == 2:
            re_input = einops.rearrange(x, "bs (c w h) -> bs c w h", c=3, w=224, h=224)
        else:
            re_input = einops.rearrange(x, "bs ws (c w h) -> (bs ws) c w h", c=3, w=224, h=224)
        
        emb = self.model(re_input)

        if org_len == 2:
            output = emb
        else:
            output = einops.rearrange(emb, "(bs ws) o -> bs ws o", ws=ws)
        
        return output
    
class ResNetMLP(nn.Module):
    def __init__(self, model_name, obs_dim, device) -> None:
        super(ResNet, self).__init__()
        if model_name == "ResNet18":
            self.model = models.resnet18(pretrained=True).to(device)
        elif model_name == "ResNet50":
            self.model = models.resnet50(pretrained=True).to(device)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        n_inputs = self.model.fc.in_features    
        
        self.fc_layer = nn.Linear(n_inputs, obs_dim).to(device)
        
        self.model.fc = self.fc_layer
        
        self.predict = False
        
        self.state_modality = 'observation'
        self.goal_modality = 'goal_observation'
        self.device = device
         
    def forward(self, input):
        obs = input[self.state_modality].to(self.device)
        goal = input[self.goal_modality].to(self.device) if self.goal_modality in input else None
        
        if not self.predict:
            x = self.fc_layer(obs)
            with torch.no_grad():
                y = self.fc_layer(goal)
        else:
            x = self.single_forward(obs, obs.shape[1])
            with torch.no_grad():
                y = self.single_forward(goal, goal.shape[1])
        
        return x, y
            
    def single_forward(self, x, ws):
        org_len = len(x.shape)
        
        if org_len == 2:
            re_input = einops.rearrange(x, "bs (c w h) -> bs c w h", c=3, w=224, h=224)
        else:
            re_input = einops.rearrange(x, "bs ws (c w h) -> (bs ws) c w h", c=3, w=224, h=224)
        
        emb = self.model(re_input)

        if org_len == 2:
            output = emb
        else:
            output = einops.rearrange(emb, "(bs ws) o -> bs ws o", ws=ws)
        
        return output