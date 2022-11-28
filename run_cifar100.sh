
# CUDA_VISIBLE_DEVICES='0' python main.py --model vgg16 --save_dir ablation_cifar100_vgg16_seed42_init1_1.0expensemble --config_path ./configs/20220223_cifar100.yml --exp 1.0

# CUDA_VISIBLE_DEVICES='0' python main.py --model resnet32 --save_dir ablation_cifar100_resnet32_seed42_init2 --config_path ./configs/20220223_cifar100.yml --init 2

# CUDA_VISIBLE_DEVICES='0' python main.py --model resnet32 --save_dir ablation_cifar100_resnet32_seed42_init10 --config_path ./configs/20220223_cifar100.yml --init 10

# CUDA_VISIBLE_DEVICES='0' python main.py --model resnet32 --save_dir ablation_cifar100_resnet32_seed42_init20 --config_path ./configs/20220223_cifar100.yml --init 20

CUDA_VISIBLE_DEVICES='0' python main.py --model densenetd40k12 --save_dir ablation_cifar100_densenetd40k12_seed42_init2_0.5expensemble --config_path ./configs/20220223_cifar100.yml --init 2

sleep 5m

CUDA_VISIBLE_DEVICES='0' python main.py --model densenetd40k12 --save_dir ablation_cifar100_densenetd40k12_seed42_init5_0.5expensemble --config_path ./configs/20220223_cifar100.yml --init 5

sleep 5m

CUDA_VISIBLE_DEVICES='0' python main.py --model densenetd40k12 --save_dir ablation_cifar100_densenetd40k12_seed42_init10_0.5expensemble --config_path ./configs/20220223_cifar100.yml --init 10

sleep 5m

CUDA_VISIBLE_DEVICES='0' python main.py --model densenetd40k12 --save_dir ablation_cifar100_densenetd40k12_seed42_init20_0.5expensemble --config_path ./configs/20220223_cifar100.yml --init 20

# CUDA_VISIBLE_DEVICES='1' python main.py --model resnet110 --save_dir cifar100_resnet110_seed42_init1_0.5expensemble --config_path ./configs/20220223_cifar100.yml

# CUDA_VISIBLE_DEVICES='0' python main.py --model wide_resnet20_8 --save_dir cifar100_wide_resnet20_8_seed42_init1_0.5expensemble --config_path ./configs/20220223_cifar100.yml
