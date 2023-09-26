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
               

python train.py --experiment mbv2   \
               --subexperiment run0 \
               --dataset tiny-imagenet   \
               --lr 0.001 \
               --router_ver 4      \
               --router_ngf 64     \
               --router_k 3        \
               --transformer_ver_root 9 \
               --transformer_ver 1   \
               --transformer_ngf 64  \
               --transformer_k 3     \
               --transformer_expansion_rate 6 \
               --solver_ver 4       \
               --batch_norm         \
               --maxdepth 8       \
               --batch-size 64   \
               --scheduler step_lr  \
               --criteria avg_valid_loss  \
               --epochs_patience 50 \
               --epochs_node 100  \
               --epochs_finetune 200 \
               --seed 0    \
               --num_workers 0 \
               --finetune_during_growth \
               --num-classes 200 