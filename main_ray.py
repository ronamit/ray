
"""
 Install before first run with:
 1. Go to main project dir
 2. $ python3 setup.py install
"""

from __future__ import division, absolute_import, print_function
import sys, os
from copy import deepcopy
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
    from common.utils import set_random_seed, create_result_dir, save_run_data, load_run_data, write_to_log, time_now, get_grid, create_results_backup
    from rllib.agents.ddpg.td3 import TD3Trainer
    from rllib.agents.ddpg.ddpg_policy import DDPGTFPolicy
except ImportError as error:
    raise ImportError(str(error) + '\n Try this:  1. Go to main project dir 2. $ python3 setup.py install ')

CustomTrainer = TD3Trainer.with_updates(
    default_policy=DDPGTFPolicy)

plt_params = {'font.size': 10,
          'lines.linewidth': 2, 'legend.fontsize': 10, 'legend.handlelength': 2,
          'pdf.fonttype':42, 'ps.fonttype':42,
          'axes.labelsize': 16, 'axes.titlesize': 16,
          'xtick.labelsize': 12, 'ytick.labelsize': 12}
plt.rcParams.update(plt_params)
# ---------------------------------------------------------------------------------------------------------------------------------#
#  Set parameters
# ---------------------------------------------------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--run-name', type=str, help='Name of dir to save results in (if empty, name by time)', default='')
parser.add_argument('--seed', type=int,  help='random seed', default=1)
parser.add_argument("--env", default="Swimmer-v2")  # OpenAI gym environment name
parser.add_argument("--default_discount", default=0.999)  # Default Discount factor
parser.add_argument('--timesteps_total', type=int,  default=1e5)
parser.add_argument('--learning_starts', type=int,  default=1e4)
parser.add_argument('--pure_exploration_steps', type=int,  default=1e4)
args, _ = parser.parse_known_args()


smoke_test = False  # True/False - short  run for debug

local_mode = False   # True/False - run non-parallel to get error messages and debugging

save_PDF = False  # False/True - save figures as PDF file

# Option to load previous run results (even unfinished) or continue unfinished run or start a new run:
run_mode = 'New'   # 'New' / 'Load' / 'Continue' / 'ContinueNewGrid' / 'ContinueAddGrid'
# If run_mode ==  'Load' / 'Continue' use this results dir:
result_dir_to_load = './saved/2020_01_19_10_56_15'

args.n_reps = 100   # 100 # number of experiment repetitions for each point in grid

#  how to create parameter grid:
# args.param_grid_def = {'type': 'gamma_guidance', 'spacing': 'linspace', 'start': 0.99, 'stop': 0.999, 'num': 10}
args.param_grid_def = {'type': 'gamma_guidance', 'spacing': 'list',
                       'list': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999][::-1]}
# args.param_grid_def = {'type': 'L2_factor', 'spacing': 'linspace', 'start': 2.25e-2, 'stop': 5e-2, 'num': 10}
# args.param_grid_def = {'type': 'L2_factor', 'spacing': 'endpoints', 'start': 2.25e-2, 'end': 5e-2, 'num': 12}
# args.param_grid_def = {'type': 'L2_factor', 'spacing': 'list', 'list': [0, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4]}

gamma_guidance = args.default_discount # default discount factor for algorithm
l2_factor = None   # default L2 regularization factor for the Q-networks


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
    args.result_dir = args.result_dir.replace("linux2", "linux4")
    mean_R = info_dict['mean_R']
    std_R = info_dict['std_R']
    alg_param_grid = info_dict['alg_param_grid']
    print('Loaded parameters: \n', args, '\n', '-'*20)

elif run_mode in {'ContinueNewGrid', 'ContinueAddGrid'}:
    # Create a new gird according to param_grid_def defined above, and use the loaded results if compatible.
    # all the other run  args (besides param_grid_def) are according to the loaded file
    loaded_args, info_dict = load_run_data(result_dir_to_load)
    assert loaded_args.param_grid_def['type'] == args.param_grid_def['type']
    create_results_backup(result_dir_to_load)
    loaded_alg_param_grid = info_dict['alg_param_grid']
    new_param_grid_def = args.param_grid_def
    new_alg_param_grid = np.around(get_grid(new_param_grid_def), decimals=10)
    args = deepcopy(loaded_args)
    if run_mode == 'ContinueAddGrid':
        new_alg_param_grid = np.union1d(loaded_alg_param_grid, new_alg_param_grid)
        args.param_grid_def['spacing'] = 'list',
        args.param_grid_def['list'] = new_alg_param_grid
        if args.param_grid_def['type'] == 'gamma_guidance':
            new_alg_param_grid[::-1].sort()
    else:
        args.param_grid_def = new_param_grid_def
    n_grid = len(new_alg_param_grid)
    mean_R = np.full(n_grid, np.nan)
    std_R = np.full(n_grid, np.nan)
    # now take completed results from loaded data:
    for i_grid, alg_param in enumerate(new_alg_param_grid):
        if alg_param in loaded_alg_param_grid:
            load_idx = np.nonzero(loaded_alg_param_grid == alg_param)
            mean_R[i_grid] = info_dict['mean_R'][load_idx]
            std_R[i_grid] = info_dict['std_R'][load_idx]
    if np.all(np.isnan(mean_R)):
        raise Warning('Loaded file  {} did not complete any of the desired grid points'.format(result_dir_to_load))
    write_to_log('Continue run with new grid def {}, {}'.format(new_param_grid_def, time_now()), args)
    write_to_log('Run parameters: \n' + str(args) + '\n' + '-'*20, args)
    alg_param_grid = new_alg_param_grid

else:
    # Start from scratch
    create_result_dir(args)
    alg_param_grid = np.around(get_grid(args.param_grid_def), decimals=10)
    n_gammas = len(alg_param_grid)
    mean_R = np.full(n_gammas, np.nan)
    std_R = np.full(n_gammas, np.nan)

if run_mode in {'New', 'Continue', 'ContinueNewGrid', 'ContinueAddGrid'}:
    # Run grid
    ray.init(local_mode=local_mode)
    start_time = timeit.default_timer()

    for i_grid, alg_param in enumerate(alg_param_grid):
        if not np.isnan(mean_R[i_grid]):
            continue  # this index already completed
        set_random_seed(args.seed)

        if args.param_grid_def['type'] == 'L2_factor':
            l2_factor = alg_param
            run_name = 'L2_' + str(l2_factor)
        elif args.param_grid_def['type'] == 'gamma_guidance':
            gamma_guidance = alg_param
            run_name = 'Gamma_' + str(alg_param)
        else:
            raise ValueError('Unrecognized args.grid_type')

        write_to_log('Starting: {}, time: {}'.format(run_name, time_now()), args)

        # Training
        analysis = tune.run(
            CustomTrainer,
            name=run_name,
            num_samples=args.n_reps,
            stop={"timesteps_total": args.timesteps_total},
            config={
                "env":args.env,
                "num_gpus": 0.1,
                # === Algorithm ===
                "gamma": gamma_guidance,
                "l2_reg_critic": l2_factor,
                "l2_reg_actor": None,
                # === Exploration ===
                "learning_starts": args.learning_starts,
                "pure_exploration_steps": args.pure_exploration_steps,
                # # === Evaluation ===
                # "evaluation_interval": 1 if args.smoke_test else 5,
                # "evaluation_num_episodes": 1 if args.smoke_test else 10,
            })
        # Evaluation

        # Get a dataframe for the last reported results of all of the trials:
        df = analysis.dataframe(metric="episode_reward_mean")
        mean_R[i_grid] = df['episode_reward_mean'].mean()
        std_R[i_grid] = df['episode_reward_mean'].std()
        # Save results so far:
        info_dict = {'mean_R': mean_R,
                     'std_R': std_R, 'alg_param_grid': alg_param_grid}
        write_to_log('Finished: {}, time: {}'.format(run_name, time_now()), args)
        write_to_log('mean_R: {}, std_R: {}'.format(mean_R[i_grid], std_R[i_grid]), args)
        save_run_data(args, info_dict)

    stop_time = timeit.default_timer()
    write_to_log('Total runtime: ' +
                 time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)), args)


if args.param_grid_def['type'] == 'L2_factor':
    alg_param_grid *= 1e2
    xlabel = r'$L_2$ Factor (1e-2)'
    title_prefix = args.env + r', $L_2$ Regularization'
elif args.param_grid_def['type'] == 'gamma_guidance':
    xlabel = r'Guidance Discount Factor $\gamma$'
    title_prefix = args.env + r', Discount Regularization'
else:
    raise ValueError('Unrecognized args.grid_type')


ci_factor = 1.96/np.sqrt(args.n_reps)  # 95% confidence interval factor
plt.figure()
# plt.errorbar(alg_param_grid, mean_R, yerr=std_R * ci_factor,
             # marker='.')
plt.plot(alg_param_grid, mean_R)

plt.fill_between(alg_param_grid, mean_R - std_R * ci_factor, mean_R + std_R * ci_factor,
                 color='blue', alpha=0.2)
plt.grid(True)
plt.xlabel(xlabel)
# plt.ylim([100, 900])
# plt.xlim([0, 4])

plt.ylabel('Average Episode Return')
if save_PDF:
    plt.title(title_prefix)
    plt.savefig(args.run_name + '.pdf', format='pdf', bbox_inches='tight')
else:
    plt.title(title_prefix + ' \n ' + str(args.result_dir), fontsize=12)
    # + 'Episode Reward Mean +- 95% CI, ' + ' \n '
plt.show()
