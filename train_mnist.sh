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

python train.py --dataset mnist \
               --experiment batch512_depth6 \
               --subexperiment  run0 \
               --batch-size 1024  \
               --epochs_patience 2\
               --epochs_node 50\
	           --epochs_finetune 100\
               -t_ver 3 -t_ngf 128 -t_k 3\
               -r_ver 6 -r_ngf 256 -r_k 3 \
               -s_ver 4 \
               -ds_int 1 \
               --maxdepth 5 \
               --visualize_split --num_workers 0\
               --seed 1