import gym
import numpy as np
import torch
import pickle
import os
from collections import defaultdict

from envs import make_vec_envs
from a2c_ppo_acktr.algo.ppo import PPO
from a2c_ppo_acktr.model import Policy

from train import train_loop
from curiosity import ForwardModel, InverseModel, ICM, RND, random_observations, CuriosityModule
from plot_utils import plot_stats, plot_stats_with_curiosity, plot_all_train_reward_history
from ppo_config import *


def run_all_train_loops(env_name, cur_models, num_env_steps, device):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Train loop params
    num_updates = int(num_env_steps) // num_steps // num_processes
    lr_decay_horizon = int(10e6) // num_steps // num_processes
    log_interval = 10
    eval_interval = 10
    time_limit = gym.make(
        env_name).spec.tags['wrapper_config.TimeLimit.max_episode_steps']

    # All methods to be tested
    ppo_variants = ['PPO', 'PPONormObs', 'PPONormObsRew']
    all_methods = list(cur_models.keys()) + ppo_variants

    n_launches = 5
    path = './'
    for name in all_methods:
        np.random.seed(42)
        torch.manual_seed(42)

        if not os.path.exists(os.path.join(path, env_name, name)):
            os.makedirs(os.path.join(path, env_name, name))

        for i in range(n_launches):
            if name == 'PPONormObs':
                envs = make_vec_envs(env_name, 1, num_processes, None,
                                     device, False, normalize_obs=True)
            elif name == 'PPONormObsRew':
                envs = make_vec_envs(env_name, 1, num_processes, gamma,
                                     device, False, normalize_obs=True)
            else:
                envs = make_vec_envs(env_name, 1, num_processes, gamma,
                                     device, False, normalize_obs=False)

            if name in ['ForwardDynLoss', 'InverseDynLoss', 'ICM', 'RND']:
                if name == 'RND':
                    cur_model = cur_models[name][0](
                        envs.observation_space.shape[0], num_processes)
                    cur_model.to(device)
                    cur_model.init_obs_norm(random_observations(
                        env_name, size=2000, device=device))
                else:
                    cur_model = cur_models[name][0](
                        envs.observation_space.shape[0], envs.action_space)
                    cur_model.to(device)
                curiosity_module = CuriosityModule(
                    cur_model, rew_coef=cur_models[name][1])
            else:
                curiosity_module = None

            print('Environment: {}, method: {}, {}'.format(env_name, name, i))
            actor_critic = Policy(envs.observation_space.shape, envs.action_space,
                                  base_kwargs={'recurrent': False}).to(device)

            agent = PPO(actor_critic,
                        clip_param,
                        ppo_epochs,
                        num_mini_batch,
                        value_loss_coef,
                        entropy_coef,
                        lr,
                        eps,
                        max_grad_norm)

            stats = train_loop(agent, envs, env_name, num_updates, num_steps, curiosity_module=curiosity_module,
                               save_interval=save_interval, eval_interval=eval_interval, log_interval=log_interval,
                               time_limit=time_limit, curiosity_rew_after=0, curiosity_rew_before=None,
                               use_linear_lr_decay=True, lr_decay_horizon=lr_decay_horizon,
                               callbacks=None)
            with open(os.path.join(path, env_name, name, str(i)), 'wb') as f:
                pickle.dump(stats, f)


def read_all_train_results(env_name):
    methods_stats = defaultdict(lambda: list())
    n_launches = 5
    path = './'
    for name in os.listdir(os.path.join(path, env_name)):
        if not name.startswith(".") :
            for i in range(n_launches):
                with open(os.path.join(path, env_name, name, str(i)), 'rb') as f:
                    stats = pickle.load(f)
                    methods_stats[name].append(stats)
    return methods_stats


def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    env_parameters = {'MountainCar-v0': {'cur_models': {'ForwardDynLoss': [ForwardModel, 10.],
                                                        'InverseDynLoss': [InverseModel, 0.5],
                                                        'RND': [RND, 10.],
                                                        'ICM': [ICM, 10.]},
                                         'num_env_steps': int(2e6)},
                      'MountainCarContinuous-v0': {'cur_models': {'ForwardDynLoss': [ForwardModel, 20.],
                                                                  'InverseDynLoss': [InverseModel, 0.5],
                                                                  'RND': [RND, 10.],
                                                                  'ICM': [ICM, 100.]},
                                                   'num_env_steps': int(1.5e6)}}
    os.mkdir('pictures')
    save_path = 'pictures'
    for env_name in env_parameters.keys():
        run_all_train_loops(
            env_name, env_parameters[env_name]['cur_models'], env_parameters[env_name]['num_env_steps'], device)
        methods_stats = read_all_train_results(env_name)
        plot_all_train_reward_history(save_path, "train_rew_all_methods",
                                      methods_stats, env_name, num_processes, num_steps, save=True)


if __name__ == '__main__':
    main()
