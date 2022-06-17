import gym
import d4rl.infos
import numpy as np
import torch

import argparse
import os
from tqdm import tqdm, trange
from coolname import generate_slug

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

import utils
from data.context_dataset import ContextDataset
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer


def experiment(
    variant,
):
    device = variant.get('device', 'cuda')

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']

    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.
    else:
        raise NotImplementedError

    env.seed(args.seed + 100)

    ref_min_score = d4rl.infos.REF_MIN_SCORE[f'{env_name}-{dataset}-v2']
    ref_max_score = d4rl.infos.REF_MAX_SCORE[f'{env_name}-{dataset}-v2']
    print('normalize score by', ref_min_score, ref_max_score)

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    batch_size = variant['batch_size']
    K = variant['K']
    mode = variant.get('mode', 'normal')
    num_eval_episodes = variant['num_eval_episodes']
    num_iterations = variant['num_steps_per_iter'] * variant['max_iters']

    # load dataset
    dataset = ContextDataset(
        env_name=env_name,
        dataset=dataset,
        max_ep_len=max_ep_len,
        max_len=K,
        reward_scale=scale,
        mode=variant['mode'],
    )
    if variant['only_prepare_dataset']:
        exit(0)

    sampler = RandomSampler(dataset, replacement=True, num_samples=batch_size * num_iterations)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size,
                            pin_memory=True,
                            num_workers=variant['num_workers'])
    dataiter = iter(dataloader)

    # used for input normalization
    state_mean = dataset.state_mean
    state_std = dataset.state_std

    def get_batch(batch_size=256, max_len=K):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = next(dataiter)
        return states.to(device), \
            actions.to(device), \
            rewards.to(device), \
            dones.to(device), \
            rtg.to(device), \
            timesteps.to(device), \
            attention_mask.to(device)

    def eval_episodes(target_rew):
        def fn(model, video=False, video_path=None):
            returns, lengths = [], []
            bar = tqdm(range(num_eval_episodes), desc='num_eval_episodes')
            for _ in bar:
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                            video_enabled=video,
                            video_path=video_path,
                            episode_id=_,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
                bar.set_postfix(returns=int(ret))
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
                f'num_eval_episodes': num_eval_episodes,
            }
        return fn

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4 * variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    if model_type == 'dt':
        trainer = SequenceTrainer(
            variant=variant,
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
            ref_min_score=ref_min_score,
            ref_max_score=ref_max_score,
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            variant=variant,
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
            ref_min_score=ref_min_score,
            ref_max_score=ref_max_score,
        )

    if variant['eval']:
        trainer.eval(variant['model_path'], video=True)
        exit(0)

    for _ in trange(variant['max_iters'], desc='epoch'):
        num_eval_episodes = variant['num_eval_episodes'] if _ + 1 == variant['max_iters'] else 10
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=_ + 1, print_logs=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--only_prepare_dataset', default=False, action='store_true')

    parser.add_argument('--exp_prefix', default=None, type=str)
    parser.add_argument('--work_dir', default='tmp', type=str)
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--model_path', default=None, type=str)

    args = parser.parse_args()
    args.cooldir = generate_slug(2)

    base_dir = 'runs'
    utils.make_dir(base_dir)
    base_dir = os.path.join(base_dir, args.work_dir)
    utils.make_dir(base_dir)
    args.work_dir = os.path.join(base_dir, args.env + '_' + args.dataset)
    utils.make_dir(args.work_dir)

    utils.set_seed_everywhere(args.seed)

    experiment(variant=vars(args))
