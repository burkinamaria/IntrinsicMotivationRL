# PPO parameters
gamma = 0.99
gae_lambda = 0.95
use_gae = True

num_processes = 16
num_steps = 128
num_mini_batch = 16
ppo_epochs = 10

value_loss_coef = 0.5
entropy_coef = 0.01
clip_param = 0.1
max_grad_norm = 5
use_proper_time_limits = True

# Optimization parameters
lr = 7e-4
eps = 1e-5

# Save params
save_dir = ''
save_interval = 100
