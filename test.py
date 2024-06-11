import gym
from beso.envs.franka_kitchen.dataloader import RelayKitchenTrajectoryDataset
from beso.envs.franka_kitchen.kitchen_env import KitchenBase, KitchenWrapper
from torch.utils.data import DataLoader
import numpy as np
from einops import einops
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import networkx as nx

def plot_graph(G, name = None):
    # Define node and edge visualization settings
    node_size = 10000
    edge_width = 2
    font_size = 12

    plt.figure(1, figsize=(32,18))
    
    G_copy = G.copy()
    edges = G_copy.copy().edges()
    
    for (u,v) in edges:
        weight = G_copy.get_edge_data(u, v)['weight']
        if weight == 0:
            G_copy.remove_edge(u,v)
            
    nodes = G_copy.copy().nodes()
    for u in nodes:
        if G_copy.degree(u) == 0:
            G_copy.remove_node(u)
    
    # Position nodes using the Spring layout
    pos = nx.spring_layout(G_copy, k=0.8, iterations=50)

    # Draw nodes and edges
    nx.draw_networkx_nodes(G_copy, pos, node_color='skyblue', node_size=node_size, alpha=0.9)
    nx.draw_networkx_edges(G_copy, pos, width=edge_width, arrowstyle='->', arrowsize=20, min_target_margin=50, min_source_margin=50)

    # Custom function to offset edge labels for better clarity
    def edge_label_offset(pos, x_offset=0.01, y_offset=0.05):
        pos_higher = {}
        for k, v in pos.items():
            pos_higher[k] = (v[0] + x_offset, v[1] + y_offset)
        return pos_higher

    # Compute new positions for edge labels
    edge_labels_pos = edge_label_offset(pos)

    # Draw edge labels using the computed positions
    edge_labels = nx.get_edge_attributes(G_copy, 'weight')
    nx.draw_networkx_edge_labels(G_copy, edge_labels_pos, edge_labels=edge_labels, font_color='red')

    # Draw node labels
    nx.draw_networkx_labels(G_copy, pos, font_size=font_size, font_weight='bold')

    # # Remove axes
    # plt.axis('off')

    plt.show()
    
    # Save the graph
    
    # plt.savefig(dataset_path + "test_graph.png")
    
    # plt.clf()
    
def create_graph():
    Graph = nx.DiGraph()
    Graph.add_node('Robot')
    for name in object_names:
        Graph.add_node(name)
        if name == "bottom burner" or name == "top burner" or name == "light switch":
            turnOn = affordance_names[0] + "_" + name
            turnOff = affordance_names[1] + "_" + name
            Graph.add_node(turnOn)
            Graph.add_node(turnOff)
            Graph.add_edge("Robot", turnOn, weight=0)
            Graph.add_edge("Robot", turnOff, weight=0)
            Graph.add_edge(turnOn, name, weight=0)
            Graph.add_edge(turnOff, name, weight=1)
        elif name == "slide cabinet" or name == "hinge cabinet" or name == "microwave":
            open = affordance_names[2] + "_" + name
            close = affordance_names[3] + "_" + name
            Graph.add_node(open)
            Graph.add_node(close)
            Graph.add_edge("Robot", open, weight=0)
            Graph.add_edge("Robot", close, weight=0)
            Graph.add_edge(open, name, weight=0)
            Graph.add_edge(close, name, weight=1)
        elif name == "kettle":
            pickUp = affordance_names[4] + "_" + name
            putDown = affordance_names[5] + "_" + name
            Graph.add_node(pickUp)
            Graph.add_node(putDown)
            Graph.add_edge("Robot", pickUp, weight=0)
            Graph.add_edge("bottom burner", putDown, weight=1)
            Graph.add_edge("top burner", putDown, weight=0)
            Graph.add_edge(pickUp, name, weight=0)
            Graph.add_edge(putDown, name, weight=1)
    return Graph

def convert_graph(Graph, oa_list):
    # create edge index from 
    adj = nx.to_scipy_sparse_array(Graph).tocoo()
    row = adj.row
    col = adj.col
    # General structure:
    # First array describes the source nodes [0,0,1,1,2,3]
    # Second array describes the goal nodes  [1,2,0,3,0,1]
    #
    # 0 <-> 1
    # ^     ^
    # |     |
    # v     v
    # 2     3
    
    nodes = []
    for node in Graph.nodes:
        split = node.split("_")
        feature_vec = np.zeros((len(oa_list),))
        feature_vec[oa_list.index(split[0])] = 1
        nodes.append(feature_vec.tolist())
        
    weights = []
    for edge in Graph.edges(data=True):
        weights.append(edge[-1]['weight'])
    
    row_tensor = torch.tensor(row)
    col_tensor = torch.tensor(col)
    feature_tensor = torch.tensor(nodes).int()
    weight_tensor = torch.tensor(weights).int()
    
    return row_tensor, col_tensor, feature_tensor, weight_tensor
    

def update_graph(Graph, goal):
    ind = goal
    name = object_names[ind]
    
    if ind == 0 or ind == 1 or ind == 2:
        turnOn = affordance_names[0] + "_" + name
        turnOff = affordance_names[1] + "_" + name
        Graph["Robot"][turnOn]['weight'] = 1
        Graph[turnOn][name]['weight'] = 1
        Graph[turnOff][name]['weight'] = 0
    elif ind == 3 or ind == 4 or ind == 5:
        open = affordance_names[2] + "_" + name
        close = affordance_names[3] + "_" + name
        Graph["Robot"][open]['weight'] = 1
        Graph[open][name]['weight'] = 1
        Graph[close][name]['weight'] = 0
    elif ind == 6: # TODO task that modifies the graph at the start and at the end of execution -> fix it
        pickUp = affordance_names[4] + "_" + name
        putDown = affordance_names[5] + "_" + name
        Graph["Robot"][pickUp]['weight'] = 1
        Graph["bottom burner"][putDown]['weight'] = 0
        Graph["top burner"][putDown]['weight'] = 1
        Graph[pickUp][name]['weight'] = 1
        Graph[putDown][name]['weight'] = 0
        Graph.add_edge("bottom burner", putDown, weight=1)
        Graph.add_edge("top burner", putDown, weight=0)
        Graph.add_edge(pickUp, name, weight=0)
        Graph.add_edge(putDown, name, weight=1)

path = "/home/paul/Desktop/datasets/image_relay_kitchen/"
data_path = "/home/paul/Desktop/datasets/image_relay_kitchen/images/"
dataset_path = "/home/paul/Desktop/datasets/image_relay_kitchen/temp/"
                
obs = np.load(path + "observations_seq.npy")
obs = einops.rearrange(obs, "b t ... -> t b ...")

object_names = [
        "bottom burner",
        "top burner",
        "light switch",
        "slide cabinet",
        "hinge cabinet",
        "microwave",
        "kettle",
    ]
affordance_names = [
    "turn on",
    "turn off",
    "open",
    "close",
    "pick up",
    "put down"
]
goals = torch.load(path + "onehot_goals.pth")
goals = einops.rearrange(goals, "b t ... -> t b ...")

Graph = create_graph()

for i in range(goals.shape[0]):
    goal = torch.argmax(goals[i][0])
    rows = []
    cols = []
    features = []
    weights = []
    for j in range(goals.shape[1]):
        current_goal = torch.argmax(goals[i][j])
        if goal != current_goal and j != 0:
            update_graph(Graph, goal)
            goal = current_goal
            print("graph updated: ", j)
        r, c, f, w = convert_graph(Graph, ["Robot"] + object_names + affordance_names)
        rows.append(r)
        cols.append(c)
        features.append(f)
        weights.append(w)
    rows = torch.stack(rows)
    cols = torch.stack(cols)
    features = torch.stack(features)
    weights = torch.stack(weights)
    print("debug")
        
#plot_graph(Graph)

print("debug")

# transform = transforms.Compose([transforms.Resize((224,224)),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
#                                 #transforms.Grayscale()
#                             ])

# for i in tqdm(range(0,1)):
#     #obs = torch.zeros(409, 224*224*3)
#     obs = np.zeros((409, 224*224*3))
#     feasible_step = int(np.sum(masks[i]))
#     for j in tqdm(range(feasible_step)):
#         episode_name = data_path + "demonstration_" + str(i) + "_step_" + str(j) + ".jpg"
#         img = Image.open(episode_name)
#         img_tensor = transform(img)
#         #obs[j] = torch.flatten(img_tensor)
#         obs[j] = torch.flatten(img_tensor).numpy()
#     #torch.save(obs, dataset_path + "img_tensor_demo_{}.pth".format(i))
#     np.save(dataset_path + "img_tensor_demo_{}".format(i), obs)
        
# #relay_traj = RelayKitchenTrajectoryDataset(data_path, onehot_goals=True)

# #dataloader = DataLoader(relay_traj, batch_size=1, shuffle=False)

# observations = np.load(data_path + "observations_seq.npy")
# masks = np.load(data_path + "existence_mask.npy")

# observations = einops.rearrange(observations, "b t ... -> t b ...")
# masks = einops.rearrange(masks, "b t ... -> t b ...")
# total_img_tensor = []

# env = gym.make('kitchen-all-v0')
# #env = KitchenWrapper(gym_env, visual_input=False)

# for epi in range(observations.shape[0]):
#     epi_img_list = []
#     print("episode:", epi)
#     feasible_step = int(np.sum(masks[epi]))
#     for step in tqdm(range(feasible_step)):
#         q_pos = observations[epi, step][:30]
#         q_vel = np.zeros(29)
#         # env.set_state(q_pos, q_vel)
#         env.sim.set_state(np.concatenate((q_pos, q_vel)))
#         env.sim.forward()
#         np_img = env.render(mode="rgb_array")
#         im = Image.fromarray(np_img)
#         im = im.resize((640, 640), Image.Resampling.LANCZOS)
#         im.save(dataset_path + "demonstration_" + str(epi) + "_step_" + str(step) + ".jpg")