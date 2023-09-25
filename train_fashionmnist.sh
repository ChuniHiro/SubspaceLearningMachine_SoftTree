# python tree.py --experiment fashion_mnist     \
#                --subexperiment batch1024_maxdepth5_seed42\
#                --dataset fashion_mnist   \
#                --lr 0.005 \
#                --batch-size 1024  \
#                --epochs_patience 100\
#                --epochs_node 50\
# 	           --epochs_finetune 100\
#                -t_ver 3 -t_ngf 128 -t_k 3 \
#                -r_ver 6 -r_ngf 64 -r_k 3 \
#                -s_ver 4 \
#                -ds_int 1 \
#                --maxdepth 5 \
#                --visualise_split \
#                --num_workers 0\
#                --seed 42 \
# 91.01

# python tree.py --experiment fashion_mnist     \
#                --subexperiment batch1024_maxdepth6_seed1_t_64_r_64\
#                --dataset fashion_mnist   \
#                --lr 0.001 \
#                --batch-size 512  \
#                --epochs_patience 2\
#                --epochs_node 100\
# 	           --epochs_finetune 200\
#                -t_ver 3 -t_ngf 128 -t_k 3\
#                -r_ver 6 -r_ngf 256 -r_k 3 \
#                -s_ver 4 \
#                -ds_int 1 \
#                --maxdepth 6 \
#                --visualise_split --num_workers 0\
#                --seed 1
# 91.5699

# python tree.py --experiment fashion_mnist     \
#                --subexperiment batch1024_maxdepth10_seed1_t_64_r_64\
#                --dataset fashion_mnist   \
#                --lr 0.001 \
#                --batch-size 512  \
#                --epochs_patience 2\
#                --epochs_node 100\
# 	           --epochs_finetune 200\
#                -t_ver 3 -t_ngf 128 -t_k 3\
#                -r_ver 6 -r_ngf 256 -r_k 3 \
#                -s_ver 4 \
#                -ds_int 1 \
#                --maxdepth 10 \
#                --visualise_split --num_workers 0\
#                --seed 1
# STILL 91.5699 tree same as above

# python tree.py --experiment fashion_mnist     \
#                --subexperiment batch1024_maxdepth10_seed1_t_64_r_64\
#                --dataset fashion_mnist   \
#                --lr 0.001 \
#                --batch-size 1024  \
#                --epochs_patience 2\
#                --epochs_node 100\
# 	           --epochs_finetune 200\
#                -t_ver 3 -t_ngf 64 -t_k 3\
#                -r_ver 6 -r_ngf 64 -r_k 3 \
#                -s_ver 4 \
#                -ds_int 1 \
#                --maxdepth 10 \
#                --visualise_split \
#                --num_workers 0\
#                --seed 1
# 90.58

# python tree.py --experiment fashion_mnist     \
#                --subexperiment batch1024_maxdepth10_seed1_t_256_r_512\
#                --dataset fashion_mnist   \
#                --lr 0.001 \
#                --batch-size 512  \
#                --epochs_patience 5 \
#                --epochs_node 100\
# 	           --epochs_finetune 200\
#                -t_ver 3 -t_ngf 256 -t_k 3\
#                -r_ver 6 -r_ngf 512 -r_k 3 \
#                -s_ver 4 \
#                -ds_int 1 \
#                --maxdepth 10 \
#                --visualise_split \
#                --num_workers 0\
#                --seed 1

# python train.py --dataset fashion_mnist\
#                --experiment batch512_depth5 \
#                --subexperiment  run0 \
#                --batch-size 512  \
#                --epochs_patience 5\
#                --epochs_node 50\
# 	           --epochs_finetune 100\
#                -t_ver 3 -t_ngf 128 -t_k 3\
#                -r_ver 6 -r_ngf 64 -r_k 3 \
#                -s_ver 4 \
#                -ds_int 1 \
#                --maxdepth 5 \
#                --visualize_split --num_workers 0\
#                --seed 0

python train.py --dataset fashion_mnist \
               --experiment mbv2light \
               --subexperiment  run0_width_0.5 \
               --lr 0.001 \
               --batch-size 64  \
               --epochs_patience 20\
               --epochs_node 20 \
	           --epochs_finetune 20\
               -t_ver_root 10 \
               -t_ver 1 -t_ngf 64 -t_k 3\
               -transformer_expansion_rate 6 \
               -t_wm 0.5 \
               -r_ver 4 -r_ngf 64 -r_k 3 \
               -s_ver 4 \
               -ds_int 1 \
               --maxdepth 1 \
               --visualize_split --num_workers 0\
               --seed 0\
               --augmentation_on