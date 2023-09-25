# python train.py --experiment test_ant_mnist    \
#                --subexperiment run0 \
#                --dataset mnist   \
#                --router_ver 1      \
#                --router_ngf 32     \
#                --router_k 3        \
#                --transformer_ver 1    \
#                --transformer_ngf 32  \
#                --transformer_k 3     \
#                --solver_ver 1        \
#                --batch_norm         \
#                --maxdepth 2       \
#                --batch-size 2048    \
#                --augmentation_on  \
#                --scheduler step_lr  \
#                --criteria avg_valid_loss  \
#                --epochs_patience 5 \
#                --epochs_node 5   \
#                --epochs_finetune 5  \
#                --seed 0    \
#                --num_workers 0    \
#                --visualise_split 

# python train.py --dataset mnist \
#                --experiment mbv2light \
#                --subexperiment  run1_width_0.2 \
#                --lr 0.001 \
#                --batch-size 64  \
#                --epochs_patience 20\
#                --epochs_node 20 \
# 	           --epochs_finetune 20\
#                -t_ver_root 10 \
#                -t_ver 1 -t_ngf 64 -t_k 3\
#                -t_wm 0.2 \
#                -r_ver 4 -r_ngf 64 -r_k 3 \
#                -s_ver 4 \
#                -ds_int 1 \
#                --maxdepth 5 \
#                --visualize_split --num_workers 0\
#                --seed 0\
#                --augmentation_on

python train.py --dataset mnist \
               --experiment mbv2light \
               --subexperiment  run1_width_0.5_expand_4 \
               --lr 0.001 \
               --batch-size 64  \
               --epochs_patience 20\
               --epochs_node 20 \
	           --epochs_finetune 20\
               -t_ver_root 10 \
               -t_ver 1 -t_ngf 64 -t_k 3\
               -t_wm 0.5 \
               -transformer_expansion_rate 4 \
               -r_ver 4 -r_ngf 64 -r_k 3 \
               -s_ver 4 \
               -ds_int 1 \
               --maxdepth 1 \
               --visualize_split --num_workers 0\
               --seed 0\
               --augmentation_on