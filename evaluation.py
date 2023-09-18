import sys
sys.path.append('./../')
import matplotlib
import visualization

# exp_dir = './../experiments/mnist/test_mnist/'
exp_dir = '/home/hongyu/Projects/SubspaceLearningMachine_SoftTree/experiments/mnist/test_ant_mnist/'

# /home/dummy/Project/AdaptiveNeuralTrees/experiments/cifar100/test_ant_cifar100/standard_seed1/checkpoints/records.pkl

models_list = ["debug"]
# models_list = ['batch512_depth10','batch512_depth6', 'patience2', 'standard']
# models_list = ['MLP_2H_IDENTITY_MLP_test']

# cifar10
# models_list = ['standard_seed1']

# fashion mnist
# models_list = ['batch1024_depth5', 'batch1024_maxdepth4_seed1', 'batch1024_maxdepth5_seed42']

# cifar100
# models_list = ["standard_seed1", "depth_10_r_256_t_256"]

# tiny-imagenet
# models_list = ["run0"]

records_file_list = [exp_dir + model_name + '/checkpoints/records.pkl' for model_name in models_list]
model_files = [exp_dir + model_name + '/checkpoints/model.pth' for model_name in models_list]

visualization.plot_performance(records_file_list, models_list, ymax = 3.0, figsize=(10,7), finetune_position=True)
visualization.plot_accuracy(records_file_list, models_list, figsize=(10,7), ymin=0, ymax=98, finetune_position=True)

_ = visualization.compute_number_of_params(model_files, models_list, is_gpu=False)
