# python train.py --experiment mbv2   \
#                --subexperiment debug \
#                --dataset cifar100   \
#                --lr 0.001 \
#                --router_ver 4      \
#                --router_ngf 64     \
#                --router_k 3        \
#                --router_dropout_prob 0.2 \
#                --transformer_ver_root 10\
#                --transformer_ver 1   \
#                --transformer_ngf 64  \
#                --transformer_k 3     \
#                --transformer_expansion_rate 2 \
#                --solver_ver 4       \
#                --solver_dropout_prob 0.2 \
#                --batch_norm         \
#                --maxdepth 8       \
#                --batch-size 512   \
#                --scheduler step_lr  \
#                --criteria avg_valid_loss  \
#                --epochs_patience 50 \
#                --epochs_node 5  \
#                --epochs_finetune 1 \
#                --seed 0    \
#                --num_workers 0 \
#                --finetune_during_growth \
#                --num-classes 100 \
#                --augmentation_on
               


# python train.py --experiment mbv2  \
#                --subexperiment run0_widthmult_1.4_t_6 \
#                --dataset cifar100   \
#                --lr 0.001 \
#                --router_ver 4      \
#                --router_ngf 64     \
#                --router_k 3        \
#                --router_dropout_prob 0.2 \
#                --transformer_ver_root 9\
#                --transformer_width_mult 1.4 \
#                --transformer_ver 1   \
#                --transformer_ngf 64  \
#                --transformer_k 3     \
#                --transformer_expansion_rate 8 \
#                --solver_ver 4       \
#                --solver_dropout_prob 0.2 \
#                --batch_norm         \
#                --maxdepth 8       \
#                --batch-size 256   \
#                --scheduler step_lr  \
#                --criteria avg_valid_loss  \
#                --epochs_patience 50 \
#                --epochs_node 100  \
#                --epochs_finetune 200 \
#                --seed 0    \
#                --num_workers 0 \
#                --finetune_during_growth \
#                --num-classes 100 \
#                --augmentation_on
               
python train.py --experiment mbv2tiny_conv  \
               --subexperiment run0_widthmult_1.4_t_6 \
               --dataset cifar100   \
               --lr 0.001 \
               --router_ver 4      \
               --router_ngf 64     \
               --router_k 3        \
               --router_dropout_prob 0.2 \
               --transformer_ver_root 11\
               --transformer_width_mult 1.4 \
               --transformer_ver 2   \
               --transformer_ngf 64  \
               --transformer_k 3     \
               --transformer_expansion_rate 8 \
               --solver_ver 4       \
               --solver_dropout_prob 0.2 \
               --batch_norm         \
               --maxdepth 8       \
               --batch-size 256   \
               --scheduler step_lr  \
               --criteria avg_valid_loss  \
               --epochs_patience 50 \
               --epochs_node 100  \
               --epochs_finetune 200 \
               --seed 0    \
               --num_workers 0 \
               --finetune_during_growth \
               --num-classes 100 \
               --augmentation_on