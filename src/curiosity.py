import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import copy


class ForwardModel(nn.Module):
    def __init__(self, state_dim, action_space, hidden_size=64):
        super(ForwardModel, self).__init__()
        
        self.state_dim = state_dim
        self.action_space = action_space
        
        if isinstance(action_space, gym.spaces.discrete.Discrete):
            self.n_actions = action_space.n
        else:
            assert isinstance(action_space, gym.spaces.box.Box), "Not supported action space"
            self.n_actions = action_space.shape[0]
            
        self.hidden_size = hidden_size
        self.model = nn.Sequential(nn.Linear(self.state_dim + self.n_actions, self.hidden_size),
                                   nn.ReLU(), 
                                   nn.Linear(self.hidden_size, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, self.state_dim))
        
    def forward(self, states, actions):
        if self.action_space.__class__.__name__ == "Discrete":
            actions = to_one_hot(actions, self.n_actions, next(self.model.parameters()).device)
        return self.model(torch.cat([states, actions], dim=1))
    
    def reward(self, states, actions, next_states, dones=None):
        with torch.no_grad():
            return torch.sum((self.forward(states, actions) - next_states) ** 2, dim=1, keepdim=True)
        
    def loss(self, states, actions, next_states, dones=None):
        return torch.sum((self.forward(states, actions) - next_states) ** 2, dim=1).mean()


class InverseModel(nn.Module):
    def __init__(self, state_dim, action_space, hidden_size=64):
        super(InverseModel, self).__init__()
        
        self.state_dim = state_dim
        self.action_space = action_space
        
        if isinstance(action_space, gym.spaces.discrete.Discrete):
            self.n_actions = action_space.n
        else:
            assert isinstance(action_space, gym.spaces.box.Box), "Not supported action space"
            self.n_actions = action_space.shape[0]
            
        self.hidden_size = hidden_size
        self.model = nn.Sequential(nn.Linear(self.state_dim * 2, self.hidden_size),
                                   nn.ReLU(), 
                                   nn.Linear(self.hidden_size, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, self.n_actions))
        
    def forward(self, states, next_states):
        return self.model(torch.cat([states, next_states], dim=1))
    
    def reward(self, states, actions, next_states, dones=None):
        with torch.no_grad():
            if self.action_space.__class__.__name__ == "Discrete":
                return F.cross_entropy(self.forward(states, next_states), actions.view(-1), reduction='none').view(-1, 1)
            else:
                return torch.sum((self.forward(states, next_states) - actions) ** 2, dim=1, keepdim=True)
    
    def loss(self, states, actions, next_states, dones=None):
        if self.action_space.__class__.__name__ == "Discrete":
            return F.cross_entropy(self.forward(states, next_states), actions.view(-1))
        else:
            return torch.sum((self.forward(states, next_states) - actions) ** 2, dim=1).mean()


class ICM(nn.Module):
    def __init__(self, state_dim, action_space, emb_size=64, hidden_size=64, beta=0.2):
        super(ICM, self).__init__()
        
        self.state_dim = state_dim
        self.action_space = action_space
        
        if isinstance(action_space, gym.spaces.discrete.Discrete):
            self.n_actions = action_space.n
        else:
            assert isinstance(action_space, gym.spaces.box.Box), "Not supported action space"
            self.n_actions = action_space.shape[0]
            
        self.beta = beta
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        
        self.encoder = nn.Sequential(nn.Linear(self.state_dim, self.hidden_size),
                                     nn.ELU(),
                                     nn.Linear(self.hidden_size, self.emb_size),
                                     nn.ELU())
        
        self.forward_model = ForwardModel(self.emb_size, action_space, self.hidden_size)
        self.inv_model = InverseModel(self.emb_size, action_space, self.hidden_size)
    
    def reward(self, states, actions, next_states, dones=None):
        with torch.no_grad():
            states_emb = self.encoder(states)
            next_states_emb = self.encoder(next_states)
        return self.forward_model.reward(states_emb, actions, next_states_emb, dones)
    
    def loss(self, states, actions, next_states, dones=None):
        states_emb = self.encoder(states)
        next_states_emb = self.encoder(next_states)
        loss = (1 - self.beta) * self.inv_model.loss(states_emb, actions, next_states_emb, dones)\
               + self.beta * self.forward_model.loss(states_emb, actions, next_states_emb, dones)
        return loss

            
class RND(nn.Module):
    def __init__(self, state_dim, num_envs, hidden_size=64,
                 clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        super(RND, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        
        self.random_network = nn.Sequential(nn.Linear(self.state_dim, self.hidden_size),
                                            nn.ReLU(), 
                                            nn.Linear(self.hidden_size, self.hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size, 1))
        self.model = copy.deepcopy(self.random_network)
        self.model.apply(init_weights)
        
        for p in self.random_network.parameters():
            p.requires_grad=False
        
        # normalization stuff
        self.ob_rms = RunningMeanStd(shape=(state_dim, ))
        self.ret_rms = RunningMeanStd(shape=(1, ))
        
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = torch.zeros(num_envs, 1, dtype=torch.float32)
        self.gamma = gamma
        self.epsilon = epsilon
            
    def init_obs_norm(self, obs):
        self.ob_rms.update(obs)

    def apply_obs_norm(self, obs):
        return torch.clamp((obs - self.ob_rms.mean) / torch.sqrt(self.ob_rms.var + self.epsilon),
                           -self.clipob, self.clipob)
    
    def apply_rew_norm(self, rews, dones):
        self.ret = self.ret * self.gamma + rews
        self.ret_rms.update(self.ret)
        self.ret[dones] = 0.

        return torch.clamp(rews / torch.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
    
    def reward(self, states, actions, next_states, dones):
        with torch.no_grad():
            rews = (self.model(self.apply_obs_norm(states)) - self.random_network(self.apply_obs_norm(states))) ** 2
            return self.apply_rew_norm(rews, dones)
        
    def loss(self, states, actions, next_states, dones):
        return torch.mean((self.model(self.apply_obs_norm(states)) - self.random_network(self.apply_obs_norm(states))) ** 2)
    
    def to(self, device):
        self.model.to(device)
        self.random_network.to(device)
        self.ob_rms.to(device)
        self.ret_rms.to(device)
        self.ret = self.ret.to(device)


class CuriosityModule():
    def __init__(self, model, rew_coef, curiosity_lr=1e-3):
        self.model = model
        self.device = next(model.parameters()).device
        self.rew_coef = rew_coef
        self.optimizer = optim.Adam(self.model.parameters(), lr=curiosity_lr)
    
    def get_reward(self, states, actions, next_states, dones):
        return self.rew_coef * self.model.reward(states, actions, next_states, dones)
    
    def update(self, states, actions, next_states, dones):
        self.optimizer.zero_grad()
        loss = self.model.loss(states, actions, next_states, dones)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().item()


# Helpers
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.zeros(shape, dtype=torch.float32)
        self.count = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)
   
    def to(self, device):
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)

    
def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + (delta ** 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

def to_one_hot(y_tensor, ndims, device):
    """ helper: take an integer vector and convert it to 1-hot matrix. """
    y_tensor = y_tensor.view(-1, 1)
    y_one_hot = torch.zeros(y_tensor.size()[0], ndims, dtype=torch.float32, device=device).scatter_(1, y_tensor, 1)
    return y_one_hot

def random_observations(env_name, device, size=1000):
    """
    Used for RND initialization
    """
    env = gym.make(env_name)
    observations = torch.empty(size, *env.observation_space.shape, dtype=torch.float32, device=device)
    
    obs = env.reset()
    for i in range(size):
        observations[i].copy_(torch.as_tensor(obs, dtype=torch.float32))
        obs, r, d, info = env.step(env.action_space.sample())
        if d:
            obs = env.reset()
    return observations


def init_weights(m):
    """
    Default PyTorch initialization for linear layer
    """
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
