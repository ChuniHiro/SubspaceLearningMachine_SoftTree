# python train.py --experiment test_ant_cifar10     \
#                --subexperiment MLP_2H_IDENTITY_MLP_TEST2 \
#                --dataset cifar10   \
#                --lr 0.001 \
#                --router_ver 7      \
#                --router_ngf 128     \
#                --router_k 3        \
#                --transformer_ver 1    \
#                --transformer_ngf 128  \
#                --transformer_k 3     \
#                --solver_ver 5       \
#                --batch_norm         \
#                --maxdepth 6       \
#                --batch-size 1024   \
#                --augmentation_on  \
#                --scheduler step_lr  \
#                --criteria avg_valid_loss  \
#                --epochs_patience 10 \
#                --epochs_node 100   \
#                --epochs_finetune 200 \
#                --seed 0    \
#                --num_workers 0    \
#                --visualise_split \
#                --finetune_during_growth \
#                --criteria always


# python train.py --experiment cifar10     \
#                --subexperiment run0 \
#                --dataset cifar10   \
#                --lr 0.002 \
#                --router_ver 2      \
#                --router_ngf 64     \
#                --router_k 3        \
#                --transformer_ver 6   \
#                --transformer_ngf 64  \
#                --transformer_k 3     \
#                --solver_ver 5       \
#                --batch_norm         \
#                --maxdepth 2       \
#                --batch-size 1024   \
#                --scheduler step_lr  \
#                --criteria avg_valid_loss  \
#                --epochs_patience 100 \
#                --epochs_node 100   \
#                --epochs_finetune 200 \
#                --seed 0    \
#                --num_workers 0 \
#                --visualize_split \
#                --finetune_during_growth \
#                --criteria always \
#                --transformer_expansion_rate 2

# python train.py --experiment cifar10     \
#                --subexperiment inverse_residual_run0 \
#                --dataset cifar10   \
#                --lr 0.001 \
#                --router_ver 8      \
#                --router_ngf 64     \
#                --router_k 3        \
#                --transformer_ver 6   \
#                --transformer_ngf 64  \
#                --transformer_k 3     \
#                --solver_ver 7       \
#                --batch_norm         \
#                --maxdepth 5       \
#                --batch-size 1024   \
#                --scheduler step_lr  \
#                --criteria avg_valid_loss  \
#                --epochs_patience 50 \
#                --epochs_node 100   \
#                --epochs_finetune 200 \
#                --seed 0    \
#                --num_workers 0 \
#                --visualize_split \
#                --finetune_during_growth \
#                --transformer_expansion_rate 2


# python train.py --experiment mbv2light     \
#                --subexperiment run0 \
#                --dataset cifar10   \
#                --lr 0.001 \
#                --router_ver 4      \
#                --router_ngf 64     \
#                --router_k 3        \
#                --transformer_ver_root 10\
#                --transformer_ver 1   \
#                --transformer_ngf 64  \
#                --transformer_k 3     \
#                --transformer_expansion_rate 2 \
#                --solver_ver 4       \
#                --batch_norm         \
#                --maxdepth 6       \
#                --batch-size 64  \
#                --scheduler step_lr  \
#                --criteria avg_valid_loss  \
#                --epochs_patience 50 \
#                --epochs_node 50  \
#                --epochs_finetune 100 \
#                --seed 0    \
#                --num_workers 0 \
#                --visualize_split \
#                --finetune_during_growth \
#                --augmentation_on
               

python train.py --experiment mbv2    \
               --subexperiment run1_with_1.4_expansion_8 \
               --dataset cifar10   \
               --lr 0.001 \
               --router_ver 4      \
               --router_ngf 64     \
               --router_k 3        \
               --transformer_ver_root 9\
               --transformer_ver 1   \
               --transformer_ngf 64  \
               --transformer_k 3     \
               --transformer_expansion_rate 8 \
               --transformer_width_mult 1.4 \
               --solver_ver 4       \
               --batch_norm         \
               --maxdepth 6       \
               --batch-size 128  \
               --scheduler step_lr  \
               --criteria avg_valid_loss  \
               --epochs_patience 50 \
               --epochs_node 50  \
               --epochs_finetune 200 \
               --seed 0    \
               --num_workers 0 \
               --visualize_split \
               --finetune_during_growth \
               --augmentation_on