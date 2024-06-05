import torch
from torch import nn
from torchvision import models
from einops import einops

class ResNet18(nn.Module):
    def __init__(self, device, window_size, goal_window_size) -> None:
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False).to(device)
        self.state_modality = 'observation'
        self.goal_modality = 'goal_observation'
        self.device = device
        self.window_size = window_size
        self.goal_window_size = goal_window_size
         
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
            x = einops.rearrange(x, "bs (w h) -> bs w h", w=128, h=128)
        else:
            x = einops.rearrange(x, "bs ws (w h) -> (bs ws) w h", w=128, h=128)
        x = einops.repeat(x, "bsws w h -> bsws c w h", c=3)
        
        x = self.embed(x)

        if org_len == 2:
            pass
        else:
            x = einops.rearrange(x, "(bs ws) o -> bs ws o", ws=ws)
        
        return x
    
    def embed(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        #x = self.fc(x)
        
        return x