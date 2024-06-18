import torch
from torch import nn
from torchvision import models
from einops import einops
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, FiLMConv, global_mean_pool, global_add_pool, global_max_pool, dense, pool
import networkx as nx
import numpy as np

OBJECT_NAMES = [
        "bottom burner",
        "top burner",
        "light switch",
        "slide cabinet",
        "hinge cabinet",
        "microwave",
        "kettle",
    ]
AFFORDANCE_NAMES = [
    "turn on",
    "turn off",
    "open",
    "close",
    "pick up",
    "put down"
]

class ImageGraphNet(nn.Module):
    def __init__(self, model_name, obs_dim, hidden_dim, emb_type, device) -> None:
        super(ImageGraphNet, self).__init__()
        if model_name == "ResNet18":
            self.vision_model = models.resnet18(pretrained=False).to(device)
        elif model_name == "ResNet50":
            self.vision_model = models.resnet50(pretrained=False).to(device)
        
        n_inputs = self.vision_model.fc.in_features    
        self.fc_layer = nn.Linear(n_inputs, obs_dim).to(device)
        self.vision_model.fc = self.fc_layer
        
        self.graph_model = GraphEmbedder(13, hidden_dim, obs_dim, 0, 'GCN', 'add').to(device)
        
        self.emb_type = emb_type
        
        self.state_modality = 'observation'
        self.goal_modality = 'goal_observation'
        self.device = device
         
    def forward(self, input):
        obs = input[self.state_modality].to(self.device)
        goal = input[self.goal_modality].to(self.device) if self.goal_modality in input else None
        if 'graph_observation' in input.keys():
            graph = self.create_graph_from_scratch(input['graph_observation'], ["Robot"] + OBJECT_NAMES + AFFORDANCE_NAMES).to(self.device)
        else:
            graph = self.create_graph(input)
        
        x = self.single_forward(obs, obs.shape[1])
        graph_obs = self.graph_model(graph, obs.shape[0])
        if self.emb_type == 'none':
            pass
        elif self.emb_type == 'add':
            x = graph_obs + x
        elif self.emb_type == 'concat':
            x = torch.concatenate((x, graph_obs), dim=-1)
        
        with torch.no_grad():
            y = self.single_forward(goal, goal.shape[1])
            if self.emb_type == 'concat':
                if len(y.shape) == 2:
                    y = einops.repeat(y, "bs dim -> bs (dim add)", add=2)
                else:
                    y = einops.repeat(y, "bs ws dim -> bs ws (dim add)", add=2)
        
        return x, y

    def create_graph(self, input):
        graph_cols_list = input['cols'].to(self.device)
        graph_rows_list = input['rows'].to(self.device)
        graph_feature_list = input['features'].to(self.device)
        graph_edge_weight_list = input['weights'].to(self.device)
        
        graph_list = []
        
        for i in range(graph_cols_list.shape[0]):
            for j in range(graph_cols_list.shape[1]):
                graph_edge_index = torch.stack((graph_cols_list[i][j], graph_rows_list[i][j]))
                graph_feature = graph_feature_list[i][j].to(torch.float32)
                graph_edge_weight = graph_edge_weight_list[i][j].to(torch.float32)
                data = Data(x=graph_feature, edge_index=graph_edge_index, edge_attr=graph_edge_weight)
                graph_list.append(data)
        graph_batch = Batch.from_data_list(graph_list)
        return graph_batch
    
    def create_graph_from_scratch(self, graph, oa_list):
        adj = nx.to_scipy_sparse_array(graph).tocoo()
        row = adj.row
        col = adj.col
        
        nodes = []
        for node in graph.nodes:
            split = node.split("_")
            feature_vec = np.zeros((len(oa_list),))
            feature_vec[oa_list.index(split[0])] = 1
            nodes.append(feature_vec.tolist())
            
        weights = []
        for edge in graph.edges(data=True):
            weights.append(edge[-1]['weight'])
        
        graph_edge_index = torch.tensor([row, col]).int()
        graph_feature = torch.tensor(nodes).float()
        graph_edge_weight = torch.tensor(weights).float()
        
        return Data(x=graph_feature, edge_index=graph_edge_index, edge_attr=graph_edge_weight)
    
    def single_forward(self, x, ws):
        org_len = len(x.shape)
        
        if org_len == 2:
            re_input = einops.rearrange(x, "bs (c w h) -> bs c w h", c=3, w=224, h=224)
        else:
            re_input = einops.rearrange(x, "bs ws (c w h) -> (bs ws) c w h", c=3, w=224, h=224)
        
        emb = self.vision_model(re_input)

        if org_len == 2:
            output = emb
        else:
            output = einops.rearrange(emb, "(bs ws) o -> bs ws o", ws=ws)
        
        return output

class GraphEmbedder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer, layer_type, pool_type):
        super(GraphEmbedder, self).__init__()
        
        if layer_type == "GCN":
            layer_class = GCNConv
        elif layer_type == "GAT":
            layer_class = GATConv
        elif layer_type == "FiLM":
            layer_class = FiLMConv
            
        if pool_type == "mean":
            pool_class = global_mean_pool
        elif pool_type == "add":
            pool_class = global_add_pool
        elif pool_type == "max":
            pool_class = global_max_pool
        
        self.pooling_layer = pool_class
        
        self.input = layer_class(input_dim, hidden_dim)
        self.layers = []
        for _ in range(num_layer):
            self.layers.append(layer_class(hidden_dim, hidden_dim))
        self.output = layer_class(hidden_dim, output_dim)
        
    def forward(self, data, batch_size):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.input(x, edge_index, edge_weight)
        x = torch.relu(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
            x = torch.relu(x)
        output = self.output(x, edge_index, edge_weight)

        graph_embedding = self.pooling_layer(output, data.batch)
        if graph_embedding.shape[0] == 1:
            pass
        else:
            graph_embedding = einops.rearrange(graph_embedding, "(bs ws) dim -> bs ws dim", bs=batch_size)

        return graph_embedding

class ResNetMLP(nn.Module):
    def __init__(self, model_name, obs_dim, device) -> None:
        super(ResNetMLP, self).__init__()
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
                y = self.fc_layer(goal)
        
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