import numpy as np
import torch
import gym
import time
import os

from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr import utils
from log_utils import TrainStatistics, CuriosityStatistics
from ppo_config import *


def evaluate(actor_critic, env_name, device, ob_rms=None):
    env = gym.make(env_name)
    reward = 0

    obs = torch.as_tensor([env.reset()], dtype=torch.float32, device=device)
    if ob_rms is not None:
        obs = scale_obs(obs, ob_rms)

    eval_recurrent_hidden_states = torch.zeros(
        1, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(1, 1, device=device)

    while True:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        if isinstance(env.action_space, gym.spaces.discrete.Discrete):
            action = action.data.cpu().item()
        else:
            action = action.data.cpu().numpy()[0]

        obs, r, done, info = env.step(action)
        obs = torch.as_tensor([obs], dtype=torch.float32, device=device)
        if ob_rms is not None:
            obs = scale_obs(obs, ob_rms)
        reward += r
        if done:
            break

    return reward


def scale_obs(obs, ob_rms, epsilon=1e-8, clipob=10.):
    return np.clip((obs - ob_rms.mean) / np.sqrt(ob_rms.var + epsilon),
                   -clipob, clipob)


def train_loop(agent, envs, env_name, num_updates, num_steps, curiosity_module=None,
               save_interval=None, eval_interval=None, log_interval=None,
               time_limit=1000, curiosity_rew_after=0, curiosity_rew_before=None,
               use_linear_lr_decay=True, lr_decay_horizon=None,
               callbacks=None):
    # Create rollout storage
    num_processes = envs.num_envs
    rollouts = RolloutStorage(num_steps, num_processes,
                              envs.observation_space.shape, envs.action_space,
                              agent.actor_critic.recurrent_hidden_state_size)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    device = next(agent.actor_critic.parameters()).device
    rollouts.to(device)

    # Create curiosity statistics saver
    if curiosity_module is not None:
        curiosity_stats = CuriosityStatistics(num_processes=num_processes,
                                              time_limit=time_limit)
    else:
        curiosity_stats = None
    # Create train statistics saver
    stats = TrainStatistics(
        log_interval, with_curiosity=(curiosity_module is not None))

    # Possibility to use curiosity only for a few epochs
    if curiosity_rew_before is None:
        curiosity_rew_before = num_updates

    # Train loop
    start = time.time()
    for j in range(num_updates):
        if use_linear_lr_decay:
            if lr_decay_horizon is None:
                lr_decay_horizon = num_updates
            utils.update_linear_schedule(
                agent.optimizer, j, lr_decay_horizon, agent.optimizer.defaults['lr'])

        curiosity_loss = 0
        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = agent.actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Observe extrinsic reward and next obs
            obs, reward, done, infos = envs.step(action)

            # Compute intrinsic rewards
            if curiosity_module is not None:
                time_limit_mask = np.array(
                    [0 if 'bad_transition' in info.keys() else 1 for info in infos])
                if use_proper_time_limits:
                    curiosity_done = done * time_limit_mask
                else:
                    curiosity_done = done
                curiosity_reward = curiosity_module.get_reward(
                    rollouts.obs[step], action, obs, curiosity_done)
                curiosity_loss += curiosity_module.update(
                    rollouts.obs[step], action, obs, curiosity_done)

            # Update current reward statistics
            stats.update_extrinsic_reward(infos)
            if curiosity_module is not None:
                curiosity_stats.update(
                    curiosity_reward.cpu().numpy().ravel(), done)

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

            if (curiosity_module is not None) and (j >= curiosity_rew_after and j <= curiosity_rew_before):
                reward = reward + curiosity_reward
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = agent.actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(
            next_value, use_gae, gamma, gae_lambda, use_proper_time_limits)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        if curiosity_module is not None:
            curiosity_loss /= num_steps
        else:
            curiosity_loss = None
        stats.update_losses(value_loss, action_loss,
                            dist_entropy, curiosity_loss)

        # Save for every interval-th episode or for the last epoch
        if (j % save_interval == 0 or j == num_updates - 1) and save_dir != "":
            save_path = os.path.join(save_dir, "ppo")
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([agent.actor_critic, getattr(utils.get_vec_normalize(envs), 'ob_rms', None)],
                       os.path.join(save_path, env_name + ".pt"))

        if j % eval_interval == 0:
            try:
                ob_rms = utils.get_vec_normalize(envs).ob_rms
            except:
                ob_rms = None
            stats.eval_episode_rewards.append(
                evaluate(agent.actor_critic, env_name, device, ob_rms))

        if j % log_interval == 0 and len(stats.episode_rewards) > 1:
            total_num_steps = (j + 1) * num_processes * num_steps
            end = time.time()
            stats.update_log(total_num_steps, curiosity_stats)
            if callbacks is not None:
                for callback in callbacks:
                    callback(stats, agent, n_updates=j, total_n_steps=total_num_steps,
                             fps=int(total_num_steps / (end - start)))

    return stats
