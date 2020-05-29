import numpy as np
from collections import deque


class CuriosityStatistics:
    def __init__(self, num_processes, time_limit, deque_max_len=50):
        self.curr_rewards = np.zeros((num_processes, time_limit + 1))
        self.curr_episode_len = np.zeros(num_processes, dtype=np.int)
        
        self.total_rewards = deque(maxlen=deque_max_len)
        self.mean_ep_rewards = deque(maxlen=deque_max_len)
        self.median_ep_rewards = deque(maxlen=deque_max_len)
        self.max_ep_rewards = deque(maxlen=deque_max_len)
        self.min_ep_rewards = deque(maxlen=deque_max_len)

    def update(self, rewards, dones):
        num_processes = len(self.curr_episode_len)
        
        self.curr_rewards[np.arange(num_processes), self.curr_episode_len] = rewards
        self.curr_episode_len += 1
        
        for i in range(num_processes):
            if dones[i]:
                self.total_rewards.append(np.sum(self.curr_rewards[i, :self.curr_episode_len[i]]))
                self.mean_ep_rewards.append(np.mean(self.curr_rewards[i, :self.curr_episode_len[i]]))
                self.median_ep_rewards.append(np.median(self.curr_rewards[i, :self.curr_episode_len[i]]))
                self.min_ep_rewards.append(np.min(self.curr_rewards[i, :self.curr_episode_len[i]]))
                self.max_ep_rewards.append(np.max(self.curr_rewards[i, :self.curr_episode_len[i]]))
                
                self.curr_episode_len[i] = 0  


class TrainStatistics:
    def __init__(self, log_interval, with_curiosity=False):
        self.with_curiosity = with_curiosity
        
        self.v_loss = deque(maxlen=log_interval)
        self.pi_loss = deque(maxlen=log_interval)
        self.entropy = deque(maxlen=log_interval)

        self.episode_rewards = deque(maxlen=50)
        self.eval_episode_rewards = deque(maxlen=10)
        
        self.env_step = []
        
        self.entropy_history = []
        self.v_loss_history = []
        self.pi_loss_history = []

        self.mean_rw_history = []
        self.median_rw_history = []
        self.max_rw_history = []
        self.min_rw_history = []

        self.eval_rw_history = []

        if with_curiosity:
            self.cur_loss = deque(maxlen=log_interval)

            # Curiosity statistics
            self.mean_curiosity_history = []
            self.median_curiosity_history = []
            self.max_curiosity_history = []
            self.min_curiosity_history = []

            # Value of the curiosity reward during an episode
            self.mean_curiosity_r_history = []
            self.median_curiosity_r_history = []
            self.max_curiosity_r_history = []
            self.min_curiosity_r_history = []

            self.curiosity_loss_history = []
        
    def update_extrinsic_reward(self, infos):
        for info in infos:
            if 'episode' in info.keys():
                self.episode_rewards.append(info['episode']['r'])

    def update_losses(self, value_loss, action_loss, dist_entropy, curiosity_loss=None):
        self.v_loss.append(value_loss)
        self.pi_loss.append(action_loss)
        self.entropy.append(dist_entropy)
        if self.with_curiosity:
            self.cur_loss.append(curiosity_loss)

    def update_log(self, total_num_steps, curiosity_stats=None):
        self.env_step.append(total_num_steps)

        self.entropy_history.append(np.mean(self.entropy))
        self.v_loss_history.append(np.mean(self.v_loss))
        self.pi_loss_history.append(np.mean(self.pi_loss))
        
        self.eval_rw_history.append(np.mean(self.eval_episode_rewards))

        self.mean_rw_history.append(np.mean(self.episode_rewards))
        self.median_rw_history.append(np.median(self.episode_rewards))
        self.max_rw_history.append(np.max(self.episode_rewards))
        self.min_rw_history.append(np.min(self.episode_rewards))
        
        if self.with_curiosity:
            self.curiosity_loss_history.append(np.mean(self.cur_loss))

            self.mean_curiosity_history.append(np.mean(curiosity_stats.total_rewards))
            self.median_curiosity_history.append(np.median(curiosity_stats.total_rewards))
            self.max_curiosity_history.append(np.max(curiosity_stats.total_rewards))
            self.min_curiosity_history.append(np.min(curiosity_stats.total_rewards))

            self.mean_curiosity_r_history.append(np.mean(curiosity_stats.mean_ep_rewards))
            self.median_curiosity_r_history.append(np.mean(curiosity_stats.median_ep_rewards))
            self.max_curiosity_r_history.append(np.mean(curiosity_stats.max_ep_rewards))
            self.min_curiosity_r_history.append(np.mean(curiosity_stats.min_ep_rewards))
