
"""
 Install before first run with:
 1. Go to main project dir
 2. $ python3 setup.py install
"""

from __future__ import division, absolute_import, print_function
import sys, os
from os.path import dirname, join, abspath
project_dir = abspath(join(dirname(__file__), '..'))
sys.path.insert(0, project_dir)
import ray
from ray import tune
import argparse
import timeit, time
import numpy as np
import matplotlib.pyplot as plt
try:
    from common.utils import set_random_seed, create_result_dir, save_run_data, load_run_data, write_to_log, time_now, get_grid
    from rllib.agents.ddpg.td3 import TD3Trainer
    from  rllib.agents.ddpg.ddpg_policy import DDPGTFPolicy
except ImportError as error:
    raise ImportError(str(error) + '\n Try this:  1. Go to main project dir 2. $ python3 setup.py install ')

CustomTrainer = TD3Trainer.with_updates(
    default_policy=DDPGTFPolicy)

params = {'font.size': 10, 'lines.linewidth': 2, 'legend.fontsize': 10, 'legend.handlelength': 2, 'pdf.fonttype':42, 'ps.fonttype':42}
plt.rcParams.update(params)

# ---------------------------------------------------------------------------------------------------------------------------------#
#  Set parameters
# ---------------------------------------------------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--run-name', type=str, help='Name of dir to save results in (if empty, name by time)', default='')
parser.add_argument('--seed', type=int,  help='random seed', default=1)
parser.add_argument("--env", default="HalfCheetah-v2")  # OpenAI gym environment name
parser.add_argument("--default_discount", default=0.999)  # Default Discount factor
parser.add_argument("--default_L2", default=0.)  # Default Discount factor
parser.add_argument('--timesteps_total', type=int,  default=1e5)
parser.add_argument('--learning_starts', type=int,  default=1e4)
parser.add_argument('--pure_exploration_steps', type=int,  default=1e4)
args, _ = parser.parse_known_args()


smoke_test = False  # True/False - short  run for debug

local_mode = False   # True/False - run non-parallel to get error messages and debugging

save_PDF = False  # False/True - save figures as PDF file

# Option to load previous run results or continue unfinished run or start a new run:
run_mode = 'New'   # 'New' / 'Load' / 'Continue'
# If run_mode ==  'Load' / 'Continue' use this results dir:
result_dir_to_load = './saved/2020_01_03_15_27_19'

args.n_reps = 100   # 100 # number of experiment repetitions for each point in grid


net_depth_grid = [1, 2, 3, 4]
param_configs_grid = [{'name': 'No Regularizer', 'gamma': 0.999, 'L2': 0.0},
                      {'name': 'Discount Regularizer', 'gamma': 0.985, 'L2': 0.0},
                      {'name': 'L2 Regularizer', 'gamma': 0.999, 'L2': 0.006}]


if smoke_test:
    print('Smoke Test !!!!!!!')
    args.n_reps = 2
    args.timesteps_total = 1e2
    args.learning_starts = 1e1
    args.pure_exploration_steps =1e1

# ---------------------------------------------------------------------------------------------------------------------------------#
#  Get results
# ---------------------------------------------------------------------------------------------------------------------------------#

if run_mode in {'Load', 'Continue'}:
    #  Load previous run
    args, info_dict = load_run_data(result_dir_to_load)
    mean_R = info_dict['mean_R']
    std_R = info_dict['std_R']
    net_depth_grid = info_dict['net_depth_grid']
    param_configs_grid = info_dict['param_configs_grid']
    print('Loaded parameters: \n', args, '\n', '-'*20)
else:
    # Start from scratch
    create_result_dir(args)

    run_grid_shape = (len(net_depth_grid), len(param_configs_grid))

    mean_R = np.full(run_grid_shape, np.nan)
    std_R = np.full(run_grid_shape, np.nan)

if run_mode in {'New', 'Continue'}:
    # Run grid
    ray.init(local_mode=local_mode)
    start_time = timeit.default_timer()

    for i_net_depth,  net_depth in enumerate(net_depth_grid):
        for i_param_config, param_config in enumerate(param_configs_grid):

            if not np.isnan(mean_R[i_net_depth, i_param_config]):
                continue  # this index already completed

            set_random_seed(args.seed)
            gamma_guidance = param_config['gamma']
            l2_factor = param_config['L2']
            run_name = 'net-depth: {}, Config: {}'.format(net_depth, param_config['name'])
            write_to_log('Starting: ' + run_name + 'time: {}'.format(time_now()),  args)

            critic_hiddens = [400] + [300] * (net_depth - 1)

            # Training
            analysis = tune.run(
                CustomTrainer,
                name=run_name,
                num_samples=args.n_reps,
                stop={"timesteps_total": args.timesteps_total},
                config={
                    "env": args.env,
                    "num_gpus": 0.1,
                    # === Algorithm ===
                    "gamma": gamma_guidance,
                    "l2_reg_critic": l2_factor,
                    "l2_reg_actor": None,
                    # === Exploration ===
                    "learning_starts": args.learning_starts,
                    "pure_exploration_steps": args.pure_exploration_steps,
                    "critic_hiddens": critic_hiddens,
                })
            # Evaluation
            # Get a dataframe for the last reported results of all of the trials:
            df = analysis.dataframe(metric="episode_reward_mean")
            mean_reward = df['episode_reward_mean'].mean()
            std_reward = df['episode_reward_mean'].std()
            mean_R[i_net_depth, i_param_config] = mean_reward
            std_R[i_net_depth, i_param_config] = std_reward
            # Save results so far:
            info_dict = {'mean_R': mean_R,
                         'std_R': std_R, 'net_depth_grid': net_depth_grid, 'param_configs_grid':param_configs_grid}
            write_to_log('Finished: ' + run_name + 'time: {}'.format(time_now()), args)
            write_to_log('mean_R: {}, std_R: {}'.format(mean_reward, std_reward), args)
            save_run_data(args, info_dict)
        # end for i_param_config
    # end for i_net_depth_grid
    stop_time = timeit.default_timer()
    write_to_log('Total runtime: ' +
                 time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)), args)

ci_factor = 1.96/np.sqrt(args.n_reps)  # 95% confidence interval factor
plt.figure()
for i_param_config, param_config in enumerate(param_configs_grid):
    plt.errorbar(net_depth_grid, mean_R[:, i_param_config], yerr=std_R[:, i_param_config] * ci_factor,
                 marker='.', label=param_config['name'])
plt.grid(True)
plt.xlabel('Network Depth')
# plt.ylim([2200, 3000])

plt.ylabel('Average Episode Return')
if save_PDF:
    plt.savefig(args.run_name + '.pdf', format='pdf', bbox_inches='tight')
else:
    plt.title('Episode Reward Mean +- 95% CI, ' + ' \n ' + str(args.result_dir))
plt.show()
