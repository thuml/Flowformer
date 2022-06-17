import torch
from torch.utils.data import Dataset


import numpy as np
import pickle
import os
from tqdm import tqdm


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


class ContextDataset(Dataset):

    def __init__(self, env_name, dataset, max_ep_len, max_len, reward_scale,
                 version='v2', mode='normal', gamma=1.,
                 data_dir='data/preprocessed_data', raw_data_dir='data/raw_data'):

        self.env_name = env_name
        self.dataset = dataset
        self.version = version

        self.max_len = max_len
        self.max_ep_len = max_ep_len
        self.reward_scale = reward_scale
        self.mode = mode
        self.gamma = gamma

        buffer_path = os.path.join(data_dir, self.buffer_name)
        if os.path.exists(buffer_path):
            self.load(buffer_path)
        else:
            # load dataset
            dataset_path = os.path.join(raw_data_dir, f'{env_name}-{dataset}-{version}.pkl')
            with open(dataset_path, 'rb') as f:
                trajectories = pickle.load(f)

            # get state & action dim
            state_dim = trajectories[0]['observations'].shape[-1]
            act_dim = trajectories[0]['actions'].shape[-1]

            # save all path information into separate lists
            states, traj_lens, returns = [], [], []
            for path in trajectories:
                if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
                    path['rewards'][-1] = path['rewards'].sum()
                    path['rewards'][:-1] = 0.
                states.append(path['observations'])
                traj_lens.append(len(path['observations']))
                returns.append(path['rewards'].sum())
            traj_lens, returns = np.array(traj_lens), np.array(returns)
            num_timesteps = sum(traj_lens)

            self.info = {
                'num_trajs': len(traj_lens),
                'num_timesteps': num_timesteps,
                'avg_return': np.mean(returns),
                'std_return': np.std(returns),
                'max_return': np.max(returns),
                'min_return': np.min(returns),
            }

            # used for input normalization
            states = np.concatenate(states, axis=0)
            self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

            s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
            for traj in tqdm(trajectories, desc='prepare dataset'):
                for si in range(traj['rewards'].shape[0]):

                    # get sequences from dataset
                    s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
                    a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
                    r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
                    if 'terminals' in traj:
                        d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
                    else:
                        d.append(traj['dones'][si:si + max_len].reshape(1, -1))
                    timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
                    timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
                    rtg_full = discount_cumsum(traj['rewards'][si:], gamma=gamma)
                    rtg.append(rtg_full[:s[-1].shape[1] + 1].reshape(1, -1, 1))
                    if rtg[-1].shape[1] <= s[-1].shape[1]:
                        rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

                    # padding and state + reward normalization
                    tlen = s[-1].shape[1]
                    s[-1] = np.concatenate([s[-1], np.zeros((1, max_len - tlen, state_dim))], axis=1)
                    s[-1] = (s[-1] - self.state_mean) / self.state_std  # normalize state
                    a[-1] = np.concatenate([a[-1], np.ones((1, max_len - tlen, act_dim)) * -10.], axis=1)
                    r[-1] = np.concatenate([r[-1], np.zeros((1, max_len - tlen, 1))], axis=1)
                    d[-1] = np.concatenate([d[-1], np.ones((1, max_len - tlen)) * 2], axis=1)
                    rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, max_len - tlen, 1))], axis=1) / reward_scale
                    timesteps[-1] = np.concatenate([timesteps[-1], np.zeros((1, max_len - tlen))], axis=1)
                    mask.append(np.concatenate([np.ones((1, tlen)), np.zeros((1, max_len - tlen))], axis=1))

            self.states = np.concatenate(s, axis=0)
            self.actions = np.concatenate(a, axis=0)
            self.rewards = np.concatenate(r, axis=0)
            self.dones = np.concatenate(d, axis=0)
            self.returns_to_go = np.concatenate(rtg, axis=0)
            self.timesteps = np.concatenate(timesteps, axis=0)
            self.masks = np.concatenate(mask, axis=0)

            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            self.save(buffer_path)

        self.print_info()

    @property
    def buffer_name(self):
        return f"context-{self.env_name}-{self.dataset}-{self.version}-{self.mode}-k{self.max_len}-e{self.max_ep_len}-g{str(self.gamma)}-s{str(self.reward_scale)}.pkl"

    def load(self, from_path):
        with open(from_path, "rb") as f:
            buffer = pickle.load(f)
            self.states = buffer['states']
            self.actions = buffer['actions']
            self.rewards = buffer['rewards']
            self.dones = buffer['dones']
            self.returns_to_go = buffer['returns_to_go']
            self.timesteps = buffer['timesteps']
            self.masks = buffer['masks']
            self.info = buffer['info']
            self.state_mean = buffer['state_mean']
            self.state_std = buffer['state_std']

    def save(self, to_path):
        with open(to_path, 'wb') as f:
            buffer = {
                'env_name': self.env_name,
                'dataset': self.dataset,
                'version': self.version,
                'mode': self.mode,
                'max_ep_len': self.max_ep_len,
                'max_len': self.max_len,
                'reward_scale': self.reward_scale,
                'gamma': self.gamma,

                'states': self.states,
                'actions': self.actions,
                'rewards': self.rewards,
                'dones': self.dones,
                'returns_to_go': self.returns_to_go,
                'timesteps': self.timesteps,
                'masks': self.masks,

                'info': self.info,
                'state_mean': self.state_mean,
                'state_std': self.state_std,
            }
            pickle.dump(buffer, f, protocol=4)

    def print_info(self):
        print('=' * 50)
        print(f'Starting new experiment: {self.env_name} {self.dataset} {self.version}')
        print(f'{self.info["num_trajs"]} trajectories, {self.info["num_timesteps"]} timesteps found')
        print(f'Average return: {self.info["avg_return"]:.2f}, std: {self.info["std_return"]:.2f}')
        print(f'Max return: {self.info["max_return"]:.2f}, min: {self.info["min_return"]:.2f}')
        print('=' * 50)

    def __getitem__(self, idx):
        return torch.from_numpy(self.states[idx]).to(dtype=torch.float32), \
            torch.from_numpy(self.actions[idx]).to(dtype=torch.float32), \
            torch.from_numpy(self.rewards[idx]).to(dtype=torch.float32), \
            torch.from_numpy(self.dones[idx]).to(dtype=torch.long), \
            torch.from_numpy(self.returns_to_go[idx]).to(dtype=torch.float32), \
            torch.from_numpy(self.timesteps[idx]).to(dtype=torch.long), \
            torch.from_numpy(self.masks[idx])

    def __len__(self):
        return len(self.states)
