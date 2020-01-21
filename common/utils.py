
from __future__ import absolute_import, division, print_function

from datetime import datetime
import os
# import torch.nn as nn
# import torch
import numpy as np
import random
import sys
import pickle
from functools import reduce
import shutil, glob


# -----------------------------------------------------------------------------------------------------------#
#  Useful functions
# -----------------------------------------------------------------------------------------------------------#

def set_random_seed(seed):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_grid(param_grid_def):
    if param_grid_def['spacing'] == 'linspace':
        alg_param_grid = np.linspace(param_grid_def['start'], param_grid_def['stop'],
                                     num=int(param_grid_def['num']))
    elif param_grid_def['spacing'] == 'endpoints':
        alg_param_grid = np.linspace(param_grid_def['start'], param_grid_def['end'],
                                    num=int(param_grid_def['num']), endpoint=True)
    elif param_grid_def['spacing'] == 'list':
        alg_param_grid = np.arange(param_grid_def['start'], )

    else:
        raise ValueError('Invalid param_grid_def')
    return alg_param_grid

# -----------------------------------------------------------------------------------------------------------#
# Result saving
# -----------------------------------------------------------------------------------------------------------#
def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def create_result_dir(args, run_experiments=True):

    if run_experiments:
        # If run_name empty, set according to time
        time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        if args.run_name == '':
            args.run_name = time_str
        run_file = sys.argv[0]
        dir_path = os.path.dirname(os.path.realpath(run_file))
        args.result_dir = os.path.join(dir_path, 'saved', args.run_name)
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        message = [
                   'Run script: ' + run_file,
                   'Log file created at ' + time_str,
                   'Parameters:', str(args), '-' * 70]
        write_to_log(message, args, mode='w') # create new log file
        write_to_log('Results dir: ' + args.result_dir, args)
        write_to_log('-' * 50, args)
        # set the path to pre-trained model, in case it is loaded (if empty - set according to run_name)
        # if not hasattr(args, 'load_model_path') or args.load_model_path == '':
        #     args.load_model_path = os.path.join(args.result_dir, 'model.pt')

        save_code(args.result_dir)
    else:
        # In this case just check if result dir exists and print the loaded parameters
        dir_path = os.path.dirname(os.path.realpath(__file__))
        args.result_dir = os.path.join(dir_path, 'saved', args.run_name)
        if not os.path.exists(args.result_dir):
            raise ValueError('Results dir not found:  ' + args.result_dir)
        else:
            print('Run script: ' + sys.argv[0])
            print( 'Data loaded from: ' + args.result_dir)
            print('-' * 70)

def write_to_log(message, args, mode='a', update_file=True):
    # mode='a' is append
    # mode = 'w' is write new file
    if not isinstance(message, list):
        message = [message]
    # update log file:
    if update_file:
        log_file_path = os.path.join(args.result_dir, 'log') + '.out'
        with open(log_file_path, mode) as f:
            for string in message:
                print(string, file=f)
    # print to console:
    for string in message:
        print(string)

def time_now():
    return datetime.now().strftime('%Y\%m\%d, %H:%M:%S')


def save_run_data(args, info_dict, verbose=1):
    run_data_file_path = os.path.join(args.result_dir, 'run_data.pkl')
    with open(run_data_file_path, 'wb') as f:
        pickle.dump([args, info_dict], f)
    if verbose == 1:
        write_to_log('Results saved in ' + run_data_file_path, args)


def load_run_data(result_dir):
    run_data_file_path = os.path.join(result_dir, 'run_data.pkl')
    with open(run_data_file_path, 'rb') as f:
       args, info_dict = pickle.load(f)
    print('Data loaded from ', run_data_file_path)
    return args, info_dict


def load_saved_vars(result_dir):
    run_data_file_path = os.path.join(result_dir, 'run_data.pkl')
    with open(run_data_file_path, 'rb') as f:
        loaded_args, loaded_dict = pickle.load(f)
    print('Loaded run parameters: ' + str(loaded_args))
    print('-' * 70)
    return loaded_args, loaded_dict

def create_results_backup(result_dir):
    src = os.path.join(result_dir, 'run_data.pkl')
    dst = os.path.join(result_dir, 'backup_run_data.pkl')
    shutil.copyfile(src, dst)
    print('Backuo of run data with original grid was saved in ', dst)

def save_code(save_dir):
    # Create backup of code
    source_dir = os.getcwd()
    dest_dir = save_dir + '/Code_Archive/'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for filename in glob.glob(os.path.join(source_dir, '*.*')):
        if ".egg-info" not in filename:
            shutil.copy(filename, dest_dir)

