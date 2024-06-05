import os
from typing import Optional, Callable, Any

from pathlib import Path
import torch
from torch.utils.data import TensorDataset, Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from beso.networks.scaler.scaler_class import Scaler
from beso.envs.dataloaders.trajectory_loader import TrajectoryDataset, get_train_val_sliced, split_traj_datasets
from beso.envs.utils import transpose_batch_timestep



class RelayKitchenTrajectoryDataset(TensorDataset, TrajectoryDataset):
    def __init__(self, data_directory, device="cpu", onehot_goals=False):
        data_directory = Path(data_directory)
        observations = torch.from_numpy(
            np.load(data_directory / "observations_seq.npy")
        )[:, :, :30] # only get ninzero stuff
        actions = torch.from_numpy(np.load(data_directory / "actions_seq.npy"))
        masks = torch.from_numpy(np.load(data_directory / "existence_mask.npy"))
        goals = torch.load(data_directory / "onehot_goals.pth")
        # The current values are in shape T x N x Dim, move to N x T x Dim
        observations, actions, masks, goals = transpose_batch_timestep(
            observations, actions, masks, goals
        )
        self.masks = masks
        tensors = [observations, actions, masks]
        if onehot_goals:
            tensors.append(goals)
        tensors = [t.to(device).float() for t in tensors]
        TensorDataset.__init__(self, *tensors)
        self.actions = self.tensors[1]
        self.observations = self.tensors[0]
        self.onehot_goals = goals

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_all_observations(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.observations[i, :T, :])
        return torch.cat(result, dim=0)

    def __getitem__(self, idx):
        T = self.masks[idx].sum().int().item()
        return tuple(x[idx, :T] for x in self.tensors)


def get_relay_kitchen_train_val(
    data_directory,
    train_fraction=0.9,
    random_seed=42,
    device="cpu",
    window_size=10,
    goal_conditional: Optional[str] = None,
    future_seq_len: Optional[int] = None,
    min_future_sep: int = 0,
    only_sample_tail: bool = False,
    only_sample_seq_end: bool = False,
    obs_modalities: str = 'state',
    transform: Optional[Callable[[Any], Any]] = None,
):
    if goal_conditional is not None:
        assert goal_conditional in ["future", "onehot"]
    
    if obs_modalities == "state":
        dataset = RelayKitchenTrajectoryDataset(data_directory, onehot_goals=(goal_conditional == "onehot"))
    elif obs_modalities == "image":
        dataset = RelayKitchenVisionTrajectoryDataset(data_directory, onehot_goals=(goal_conditional == "onehot"))
    
    return get_train_val_sliced(
        dataset,
        train_fraction,
        random_seed,
        device,
        window_size,
        future_conditional=(goal_conditional == "future"),
        min_future_sep=min_future_sep,
        future_seq_len=future_seq_len,
        transform=transform,
        only_sample_tail=only_sample_tail,
        only_sample_seq_end=only_sample_seq_end,
    ), dataset


class RelayKitchenVisionTrajectoryDataset(TensorDataset, TrajectoryDataset):
    def __init__(self, data_directory_path, device="cpu", onehot_goals=False):
        data_directory = Path(data_directory_path)
        
        # path = data_directory_path + "/image_resnet_embedding/"
        # observations = torch.zeros((566, 409, 512))
        # for idx, name in enumerate(os.listdir(path)):
        #     episode_name = path + name
        #     img_embedding = torch.load(episode_name)
        #     img_embedding.requires_grad = False
        #     observations[idx] = img_embedding
        
        path = data_directory_path + "/pre_processed_images/"
        # images are 128 * 128 * 1 = 16384 or 64 * 64 * 3 = 12288
        observations = torch.zeros((566, 409, 128*128*1))
        for idx, name in tqdm(enumerate(os.listdir(path))):
            episode_name = path + name
            img_tensor = torch.load(episode_name)
            observations[idx] = img_tensor
        
        actions = torch.from_numpy(np.load(data_directory / "actions_seq.npy"))
        masks = torch.from_numpy(np.load(data_directory / "existence_mask.npy"))
        goals = torch.load(data_directory / "onehot_goals.pth")
        # The current values are in shape T x N x Dim, move to N x T x Dim
        actions, masks, goals = transpose_batch_timestep(
            actions, masks, goals
        )
        self.masks = masks
        tensors = [observations, actions, masks]
        tensors.append(goals)
        tensors = [t.to(device).float() for t in tensors]
        TensorDataset.__init__(self, *tensors)
        self.actions = self.tensors[1]
        self.observations = self.tensors[0]
        self.onehot_goals = goals

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)
    
    def get_all_observations(self):
        result = []
        # mask out invalid observations
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.observations[i, :T, :])
        return torch.cat(result, dim=0)

    def __getitem__(self, idx):
        T = self.masks[idx].sum().int().item()
        return tuple(x[idx, :T] for x in self.tensors)
