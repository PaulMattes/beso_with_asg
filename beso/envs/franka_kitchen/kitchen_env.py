import logging
import adept_envs
from adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1
import os
import numpy as np
import torch
import einops
import gym
import torch.nn.functional as F
from torchvision import models, transforms
import networkx as nx
import matplotlib.pyplot as plt

OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}
OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}
BONUS_THRESH = 0.9 #0.3

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

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="w",
)

class KitchenWrapper(gym.Wrapper):
    def __init__(self, env, input):
        super(KitchenWrapper, self).__init__(env)
        self.env = env
        self.input = input
        self.transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((224,224)),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
                                        #transforms.Grayscale()
                                    ])

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        if self.input == 'image':
            return_obs = self.env.render(mode="rgb_array")
            return self.preprocess_img(return_obs)
        elif self.input == 'image_graph':
            return_obs = self.env.render(mode="rgb_array")
            return [self.preprocess_img(return_obs), self.env.graph_obs]
        else:
            return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.input == "image":
            return_obs = self.env.render(mode="rgb_array")
            return self.preprocess_img(return_obs), reward, done, info
        elif self.input == 'image_graph':
            return_obs = self.env.render(mode="rgb_array")
            return [self.preprocess_img(return_obs), self.graph_obs], reward, done, info
        else:
            return obs, reward, done, info
    
    def preprocess_img(self, img):
        tensor_img = self.transform(img.copy())
        tensor_img = einops.rearrange(tensor_img, "(bs c) w h -> bs (c w h)", bs=1)
        return tensor_img

def create_graph():
    Graph = nx.DiGraph()
    Graph.add_node('Robot')
    for name in OBJECT_NAMES:
        Graph.add_node(name)
        if name == "bottom burner" or name == "top burner" or name == "light switch":
            turnOn = AFFORDANCE_NAMES[0] + "_" + name
            turnOff = AFFORDANCE_NAMES[1] + "_" + name
            Graph.add_node(turnOn)
            Graph.add_node(turnOff)
            Graph.add_edge("Robot", turnOn, weight=0)
            Graph.add_edge("Robot", turnOff, weight=0)
            Graph.add_edge(turnOn, name, weight=0)
            Graph.add_edge(turnOff, name, weight=1)
        elif name == "slide cabinet" or name == "hinge cabinet" or name == "microwave":
            open = AFFORDANCE_NAMES[2] + "_" + name
            close = AFFORDANCE_NAMES[3] + "_" + name
            Graph.add_node(open)
            Graph.add_node(close)
            Graph.add_edge("Robot", open, weight=0)
            Graph.add_edge("Robot", close, weight=0)
            Graph.add_edge(open, name, weight=0)
            Graph.add_edge(close, name, weight=1)
        elif name == "kettle":
            pickUp = AFFORDANCE_NAMES[4] + "_" + name
            putDown = AFFORDANCE_NAMES[5] + "_" + name
            Graph.add_node(pickUp)
            Graph.add_node(putDown)
            Graph.add_edge("Robot", pickUp, weight=0)
            Graph.add_edge("bottom burner", putDown, weight=1)
            Graph.add_edge("top burner", putDown, weight=0)
            Graph.add_edge(pickUp, name, weight=0)
            Graph.add_edge(putDown, name, weight=1)
    return Graph

def update_graph(Graph, goal):
    ind = goal
    name = OBJECT_NAMES[ind]
    
    if ind == 0 or ind == 1 or ind == 2:
        turnOn = AFFORDANCE_NAMES[0] + "_" + name
        turnOff = AFFORDANCE_NAMES[1] + "_" + name
        Graph["Robot"][turnOn]['weight'] = 1
        Graph[turnOn][name]['weight'] = 1
        Graph[turnOff][name]['weight'] = 0
    elif ind == 3 or ind == 4 or ind == 5:
        open = AFFORDANCE_NAMES[2] + "_" + name
        close = AFFORDANCE_NAMES[3] + "_" + name
        Graph["Robot"][open]['weight'] = 1
        Graph[open][name]['weight'] = 1
        Graph[close][name]['weight'] = 0
    elif ind == 6: # TODO task that modifies the graph at the start and at the end of execution -> fix it
        pickUp = AFFORDANCE_NAMES[4] + "_" + name
        putDown = AFFORDANCE_NAMES[5] + "_" + name
        Graph["Robot"][pickUp]['weight'] = 1
        Graph["bottom burner"][putDown]['weight'] = 0
        Graph["top burner"][putDown]['weight'] = 1
        Graph[pickUp][name]['weight'] = 1
        Graph[putDown][name]['weight'] = 0
        Graph.add_edge("bottom burner", putDown, weight=1)
        Graph.add_edge("top burner", putDown, weight=0)
        Graph.add_edge(pickUp, name, weight=0)
        Graph.add_edge(putDown, name, weight=1)

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

class KitchenBase(KitchenTaskRelaxV1):
    # A string of element names. The robot's task is then to modify each of
    # these elements appropriately.
    TASK_ELEMENTS = []
    ALL_TASKS = [
        "bottom burner",
        "top burner",
        "light switch",
        "slide cabinet",
        "hinge cabinet",
        "microwave",
        "kettle",
    ]
    REMOVE_TASKS_WHEN_COMPLETE = True
    TERMINATE_ON_TASK_COMPLETE = True
    TERMINATE_ON_WRONG_COMPLETE = False
    COMPLETE_IN_ANY_ORDER = (
        True  # This allows for the tasks to be completed in arbitrary order.
    )

    def __init__(
        self, dataset_url=None, ref_max_score=None, ref_min_score=None, **kwargs
    ):
        self.tasks_to_complete = list(self.TASK_ELEMENTS)
        self.all_completions = []
        self.goal_masking = True
        self.graph_obs = create_graph()
        super(KitchenBase, self).__init__(**kwargs)

    def set_goal_masking(self, goal_masking=True):
        """Sets goal masking for goal-conditioned approaches (like RPL)."""
        self.goal_masking = goal_masking

    def _get_task_goal(self, task=None, actually_return_goal=False):
        if task is None:
            task = ["microwave", "kettle", "bottom burner", "light switch"]
        new_goal = np.zeros_like(self.goal)
        if self.goal_masking and not actually_return_goal:
            return new_goal
        for element in task:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal

        return new_goal

    def reset_model(self):
        self.tasks_to_complete = list(self.TASK_ELEMENTS)
        self.all_completions = []
        self.graph_obs = create_graph()
        return super(KitchenBase, self).reset_model()

    def _get_reward_n_score(self, obs_dict):
        reward_dict, score = super(KitchenBase, self)._get_reward_n_score(obs_dict)
        reward = 0.0
        next_q_obs = obs_dict["qp"]
        next_obj_obs = obs_dict["obj_qp"]
        next_goal = self._get_task_goal(
            task=self.TASK_ELEMENTS, actually_return_goal=True
        )  # obs_dict['goal']
        idx_offset = len(next_q_obs)
        completions = []
        all_completed_so_far = True
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] - next_goal[element_idx]
            )
            complete = distance < BONUS_THRESH
            condition = (
                complete and all_completed_so_far
                if not self.COMPLETE_IN_ANY_ORDER
                else complete
            )
            if condition:  # element == self.tasks_to_complete[0]:
                logging.info("Task {} completed!".format(element))
                completions.append(element)
                self.all_completions.append(element)
                update_graph(self.graph_obs, self.ALL_TASKS.index(element))
            all_completed_so_far = all_completed_so_far and complete
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            [self.tasks_to_complete.remove(element) for element in completions]
        bonus = float(len(completions))
        reward_dict["bonus"] = bonus
        reward_dict["r_total"] = bonus
        score = bonus
        return reward_dict, score

    def step(self, a, b=None):
        obs, reward, done, env_info = super(KitchenBase, self).step(a, b=b)
        if self.TERMINATE_ON_TASK_COMPLETE:
            done = not self.tasks_to_complete
        if self.TERMINATE_ON_WRONG_COMPLETE:
            all_goal = self._get_task_goal(task=self.ALL_TASKS)
            for wrong_task in list(set(self.ALL_TASKS) - set(self.TASK_ELEMENTS)):
                element_idx = OBS_ELEMENT_INDICES[wrong_task]
                distance = np.linalg.norm(obs[..., element_idx] - all_goal[element_idx])
                complete = distance < BONUS_THRESH
                if complete:
                    done = True
                    break
        env_info["all_completions"] = self.all_completions
        return obs, reward, done, env_info

    def get_goal(self):
        """Loads goal state from dataset for goal-conditioned approaches (like RPL)."""
        raise NotImplementedError

    def _split_data_into_seqs(self, data):
        """Splits dataset object into list of sequence dicts."""
        seq_end_idxs = np.where(data["terminals"])[0]
        start = 0
        seqs = []
        for end_idx in seq_end_idxs:
            seqs.append(
                dict(
                    states=data["observations"][start : end_idx + 1],
                    actions=data["actions"][start : end_idx + 1],
                )
            )
            start = end_idx + 1
        return 