import sys
import os
import torch
import json
import multiprocessing
from Model.aSN2N import aSN2N


# Set working directory to the script's parent directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Current working directory:", os.getcwd())

test_mode = False  # If True, only load model and test without training
debug_mode = False  # If True, run experiments sequentially without multiprocessing


def run_experiment(config):
    # Set GPU device
    device_index = config['device_index']
    torch.cuda.set_device(device_index)
    print(f"Using GPU: {device_index}")

    torch.cuda.empty_cache()

    # Extract configuration parameters
    dataset_name = config['dataset_name']
    train_data_path = config['train_data_path']
    test_path = config['test_path']
    epochs = config['epochs']
    train_batch_size = config['train_batch_size']
    test_batch_size = config['test_batch_size']
    reg = config['reg']
    reg_sparse = config['reg_sparse']
    work_mode = config['work_mode']

    # Inference mode with overlapping blocks, defaults to work_mode
    inference_mode = config.get('inference_mode', work_mode)

    # Prepare output directories
    prefix = '2D_'
    tests_name = (prefix + ('%s_EPOCH%d_BS%d_LOSSconsis_%.2f_LOSSsparse%.2f_' %
                (dataset_name, epochs, train_batch_size, reg, reg_sparse)))
    os.makedirs((('./images/%s' % (tests_name))), exist_ok=True)
    os.makedirs((('./images/%s/images' % (tests_name))), exist_ok=True)
    os.makedirs((('./images/%s/checkpoints' % (tests_name))), exist_ok=True)

    # Initialize and train aSN2N model
    aSN2N_Net = aSN2N(dataset_name=dataset_name, tests_name=tests_name, reg=reg, reg_sparse=reg_sparse,
                    constrained_type='L1', epochs=epochs, train_batch_size=train_batch_size,
                    ifadaptive_lr=False, test_batch_size=test_batch_size, train_data_path=train_data_path,
                    test_path=test_path, test_mode=test_mode, work_mode=work_mode, img_res=(64, 64), 
                    inference_mode=inference_mode)
    aSN2N_Net.train()


def main():
    # Check available GPU devices
    num_gpus = torch.cuda.device_count()
    print("Number of available GPUs:", num_gpus)

    # Set the number of experiments per GPU
    experiments_per_gpu = 1  # Adjust based on your GPU memory

    # Load configuration file
    # config_path = './Config/your_config.json'  # Specify your config file path
    config_path = './Config/test.json'  # Specify your config file path
    with open(config_path, 'r') as f:
        configs = json.load(f)

    if debug_mode:
        print("Debug mode: running experiments sequentially on a single GPU")
        for config in configs:
            config['device_index'] = 0  # Use first GPU in debug mode
            run_experiment(config)
        return

    # Assign GPU to each configuration
    for i, config in enumerate(configs):
        config['device_index'] = i // experiments_per_gpu % num_gpus

    # Create process pool
    pool = multiprocessing.Pool(processes=num_gpus * experiments_per_gpu)

    # Run experiments in parallel
    pool.map(run_experiment, configs)

    # Close process pool
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
  