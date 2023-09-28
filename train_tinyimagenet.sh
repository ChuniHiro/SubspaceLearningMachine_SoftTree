# python train.py --experiment mbv3   \
#                --subexperiment run0 \
#                --dataset tiny-imagenet   \
#                --lr 0.001 \
#                --router_ver 4      \
#                --router_ngf 64     \
#                --router_k 3        \
#                --transformer_ver_root 8\
#                --transformer_ver 1   \
#                --transformer_ngf 64  \
#                --transformer_k 3     \
#                --transformer_expansion_rate 2 \
#                --solver_ver 4       \
#                --batch_norm         \
#                --maxdepth 5       \
#                --batch-size 64   \
#                --scheduler step_lr  \
#                --criteria avg_valid_loss  \
#                --epochs_patience 50 \
#                --epochs_node 1  \
#                --epochs_finetune 1 \
#                --seed 0    \
#                --num_workers 0 \
#                --finetune_during_growth \
#                --num-classes 200 
               

# python train.py --experiment mbv2   \
#                --subexperiment run0_width_1.4_t_6git  \
#                --dataset tiny-imagenet   \
#                --lr 0.001 \
#                --router_ver 4      \
#                --router_ngf 64     \
#                --router_k 3        \
#                --transformer_ver_root 9 \
#                --transformer_ver 1   \
#                --transformer_ngf 64  \
#                --transformer_k 3     \
#                --transformer_expansion_rate 6 \
#                --transformer_width_mult 1.4 \
#                --solver_ver 4       \
#                --batch_norm         \
#                --maxdepth 8       \
#                --batch-size 128   \
#                --scheduler step_lr  \
#                --criteria avg_valid_loss  \
#                --epochs_patience 50 \
#                --epochs_node 100  \
#                --epochs_finetune 200 \
#                --seed 0    \
#                --num_workers 0 \
#                --finetune_during_growth \
#                --num-classes 200 

# python train.py --experiment mbv2   \
#                --subexperiment debug  \
#                --dataset tiny-imagenet   \
#                --lr 0.0005 \
#                --router_ver 4      \
#                --router_ngf 64     \
#                --router_k 3        \
#                --transformer_ver_root 11 \
#                --transformer_ver 2   \
#                --transformer_ngf 64  \
#                --transformer_k 3     \
#                --transformer_expansion_rate 6 \
#                --transformer_width_mult 1.4 \
#                --solver_ver 4       \
#                --batch_norm         \
#                --maxdepth 2       \
#                --batch-size 256   \
#                --scheduler step_lr  \
#                --criteria avg_valid_loss  \
#                --epochs_patience 50 \
#                --epochs_node 1  \
#                --epochs_finetune 1 \
#                --seed 0    \
#                --num_workers 0 \
#                --finetune_during_growth \
#                --num-classes 200 

# python train.py --experiment mbv2   \
#                --subexperiment run0_width_2_t_8  \
#                --dataset tiny-imagenet   \
#                --lr 0.0005 \
#                --router_ver 4      \
#                --router_ngf 64     \
#                --router_k 3        \
#                --transformer_ver_root 9 \
#                --transformer_ver 2   \
#                --transformer_ngf 64  \
#                --transformer_k 3     \
#                --transformer_expansion_rate 8 \
#                --transformer_width_mult 2.0 \
#                --solver_ver 4       \
#                --batch_norm         \
#                --maxdepth 8       \
#                --batch-size 128   \
#                --scheduler step_lr  \
#                --criteria avg_valid_loss  \
#                --epochs_patience 50 \
#                --epochs_node 50  \
#                --epochs_finetune 200 \
#                --seed 0    \
#                --num_workers 0 \
#                --finetune_during_growth \
#                --num-classes 200 


# python train.py --experiment mbv2tiny   \
#                --subexperiment run0_width_1.4_t_8  \
#                --dataset tiny-imagenet   \
#                --lr 0.0005 \
#                --router_ver 4      \
#                --router_ngf 64     \
#                --router_k 3        \
#                --transformer_ver_root 11 \
#                --transformer_ver 2   \
#                --transformer_ngf 64  \
#                --transformer_k 3     \
#                --transformer_expansion_rate 8 \
#                --transformer_width_mult 1.4 \
#                --solver_ver 4       \
#                --batch_norm         \
#                --maxdepth 8       \
#                --batch-size 128   \
#                --scheduler step_lr  \
#                --criteria avg_valid_loss  \
#                --epochs_patience 50 \
#                --epochs_node 50  \
#                --epochs_finetune 200 \
#                --seed 0    \
#                --num_workers 0 \
#                --finetune_during_growth \
#                --num-classes 200 

python train.py --experiment mbv2tiny   \
               --subexperiment run0_width_1.4_t_6  \
               --dataset tiny-imagenet   \
               --lr 0.0005 \
               --router_ver 4      \
               --router_ngf 64     \
               --router_k 3        \
               --transformer_ver_root 11 \
               --transformer_ver 2   \
               --transformer_ngf 64  \
               --transformer_k 3     \
               --transformer_expansion_rate 6 \
               --transformer_width_mult 1.4 \
               --solver_ver 4       \
               --batch_norm         \
               --maxdepth 8       \
               --batch-size 128   \
               --scheduler step_lr  \
               --criteria avg_valid_loss  \
               --epochs_patience 50 \
               --epochs_node 50  \
               --epochs_finetune 200 \
               --seed 0    \
               --num_workers 0 \
               --finetune_during_growth \
               --num-classes 200 