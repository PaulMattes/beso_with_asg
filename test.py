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

path = "/home/paul/Desktop/datasets/image_relay_kitchen/"
data_path = "/home/paul/Desktop/datasets/image_relay_kitchen/images/"
dataset_path = "/home/paul/Desktop/datasets/image_relay_kitchen/temp/"
                     
masks = np.load(path + "existence_mask.npy")
masks = einops.rearrange(masks, "b t ... -> t b ...")

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
                                #transforms.Grayscale()
                            ])

for i in tqdm(range(52,566)):
    obs = torch.zeros(409, 224*224*3)
    feasible_step = int(np.sum(masks[i]))
    for j in tqdm(range(feasible_step)):
        episode_name = data_path + "demonstration_" + str(i) + "_step_" + str(j) + ".jpg"
        img = Image.open(episode_name)
        img_tensor = transform(img)
        obs[j] = torch.flatten(img_tensor)
    torch.save(obs, dataset_path + "img_tensor_demo_{}.pth".format(i))        
        
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