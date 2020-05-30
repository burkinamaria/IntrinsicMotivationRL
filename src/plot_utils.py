import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.display import clear_output


def plot_stats(stats, agent, n_updates, total_n_steps, fps):
    clear_output(True)
    print("Updates {}, num timesteps {}, FPS {}".format(n_updates, total_n_steps, fps))
    for param_group in agent.optimizer.param_groups:
        print("Current lr: {:.5f}".format(param_group['lr']))
        break

    plt.figure(figsize=[16, 16])

    plt.subplot(3, 2, 1)
    plt.title("Train episode reward history")
    plt.plot(stats.env_step, stats.mean_rw_history, label='Mean')
    plt.plot(stats.env_step, stats.median_rw_history, label='Median')
    plt.fill_between(stats.env_step, stats.min_rw_history, stats.max_rw_history, alpha=0.3)
    plt.legend()
    plt.grid()
    
    plt.subplot(3, 2, 2)
    plt.title("Evaluation reward history")
    plt.plot(stats.env_step, stats.eval_rw_history)
    plt.grid()

    plt.subplot(3, 2, 3)
    plt.title("Value loss history")
    plt.plot(stats.env_step, stats.v_loss_history)
    plt.grid()

    plt.subplot(3, 2, 4)
    plt.title("Policy loss history")
    plt.plot(stats.env_step, stats.pi_loss_history)
    plt.grid()
    
    plt.subplot(3, 2, 5)
    plt.title("Entropy history")
    plt.plot(stats.env_step, stats.entropy_history)
    plt.grid()
    
    plt.show()

def plot_stats_with_curiosity(stats, agent, n_updates, total_n_steps, fps):
    clear_output(True)
    print("Updates {}, num timesteps {}, FPS {}".format(n_updates, total_n_steps, fps))
    for param_group in agent.optimizer.param_groups:
        print("Current lr: {:.5f}".format(param_group['lr']))
        break

    plt.figure(figsize=[16, 17])

    plt.subplot(4, 2, 1)
    plt.title("Extrinsic last {} episodes reward".format(len(stats.episode_rewards)))
    plt.plot(stats.env_step, stats.mean_rw_history, label='Mean')
    plt.plot(stats.env_step, stats.median_rw_history, label='Median')
    plt.fill_between(stats.env_step, stats.min_rw_history, stats.max_rw_history, alpha=0.3)
    plt.legend()
    plt.grid()

    plt.subplot(4, 2, 2)
    plt.title("Intrinsic last {} episodes reward".format(len(stats.episode_rewards)))
    plt.plot(stats.env_step, stats.mean_curiosity_history, label='Mean')
    plt.plot(stats.env_step, stats.median_curiosity_history, label='Median')
    plt.fill_between(stats.env_step, stats.min_curiosity_history, stats.max_curiosity_history, alpha=0.3)
    plt.legend()
    plt.grid()
    
    plt.subplot(4, 2, 3)
    plt.title("Intrinsic reward during an episode")
    plt.plot(stats.env_step, stats.mean_curiosity_r_history, label='Mean')
    plt.plot(stats.env_step, stats.median_curiosity_r_history, label='Median')
    plt.fill_between(stats.env_step, stats.min_curiosity_r_history, stats.max_curiosity_r_history, alpha=0.3)
    plt.legend()
    plt.grid()
    
    plt.subplot(4, 2, 4)
    plt.title("Curiosity module loss")
    plt.plot(stats.env_step, stats.curiosity_loss_history)
    plt.grid()

    plt.subplot(4, 2, 5)
    plt.title("Entropy history")
    plt.plot(stats.env_step, stats.entropy_history)
    plt.grid()
    
    plt.subplot(4, 2, 6)
    plt.title("Evaluation reward history")
    plt.plot(stats.env_step, stats.eval_rw_history)
    plt.grid()

    plt.subplot(4, 2, 7)
    plt.title("Value loss history")
    plt.plot(stats.env_step, stats.v_loss_history)
    plt.grid()

    plt.subplot(4, 2, 8)
    plt.title("Policy loss history")
    plt.plot(stats.env_step, stats.pi_loss_history)
    plt.grid()

    plt.show()


def plot_all_train_reward_history(save_path, pic_prefix, methods_stats, env_name, num_processes, num_steps, 
                                  env_steps=False, length=-1, save=False):
    plt.figure(figsize=[12, 7])
    plt.title("Extrinsic reward during training\n{}".format(env_name), fontsize='x-large')
    for name in methods_stats.keys():
        stats_list = methods_stats[name]
        lengths = [len(stats.mean_rw_history) for stats in stats_list]
        rewards = np.array([stats.mean_rw_history[lengths[i]-min(lengths):] for i, stats in enumerate(stats_list)])
        if env_steps:
            log_steps = stats_list[0].env_step[lengths[0]-min(lengths):]
        else:
            log_steps = np.array(stats_list[0].env_step[lengths[0]-min(lengths):]) // (num_processes * num_steps)

        log_steps = log_steps[:length]
        rewards = rewards[:, :length]
        plt.plot(log_steps, np.mean(rewards, axis=0), label=name)
        plt.fill_between(log_steps, np.min(rewards, axis=0), np.max(rewards, axis=0), alpha=0.3)
    
    if env_steps:
        plt.xlabel('#env steps', fontsize='large')
    else:
        plt.xlabel('#ppo updates', fontsize='large')
    plt.ylabel('episode reward', fontsize='large')
    plt.grid()
    plt.legend(fontsize='large')
    if save:
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "{}_{}.png".format(pic_prefix, env_name)), format='png', pad_inches=0)
    else:
        plt.show()


def plot_curiosity_history(save_path, pic_prefix, stats, method_name, env_name, num_processes, num_steps,
                           env_steps=False, save=False, length=-1):
    plt.figure(figsize=[10, 7])
    plt.title("{} intrinsic reward during training\n{}".format(method_name, env_name), fontsize='x-large')
    if env_steps:
        log_steps = stats.env_step
    else:
        log_steps = np.array(stats.env_step) // (num_processes * num_steps)
    
    skip_first = 5 # sometimes the loss on the first steps is too big compared with the next loss we want to look at
    plt.subplot(2, 2, 1)
    plt.title("Extrinsic episode reward (averaged by {} last episodes)".format(len(stats.episode_rewards)))
    plt.plot(log_steps[skip_first:length], stats.mean_rw_history[skip_first:length])
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.title("Intrinsic episode reward (averaged by {} last episodes)".format(len(stats.episode_rewards)))
    plt.plot(log_steps[skip_first:length], stats.mean_curiosity_history[skip_first:length])
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.title("Max intrinsic reward of one step during an episode")
    plt.plot(log_steps[skip_first:length], stats.max_curiosity_r_history[skip_first:length])
    if env_steps:
        plt.xlabel('#env steps', fontsize='large')
    else:
        plt.xlabel('#ppo updates', fontsize='large')
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.title("Curiosity module loss")
    plt.plot(log_steps[5:length], stats.curiosity_loss_history[5:length])
    if env_steps:
        plt.xlabel('#env steps', fontsize='large')
    else:
        plt.xlabel('#ppo updates', fontsize='large')
    plt.grid()

    if save:
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "{}_{}_{}.png".format(pic_prefix, method_name, env_name)),
                    format='png', pad_inches=0)
    else:
        plt.show()