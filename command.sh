# diff 0.4
CUDA_VISIBLE_DEVICES=0 python3 main.py --project=mil_shz --datasets=camelyon16 --dataset_root=/data2/zhangxiaoxian/camelyon_all/r50_bioseg/ --model_path=/data/shihuazhan/output_wsi/ --cv_fold=3 --title=abmil_diff0.4_twh_call_2023 --model=diff --seed=2023 --k_ratio=0.4 --t_steps=2 --ifTrain=1 --ifrand=0 --wandb;
# baseline
CUDA_VISIBLE_DEVICES=1 python3 main.py --project=mil_shz --datasets=camelyon16 --dataset_root=/data2/zhangxiaoxian/camelyon_all/r50_bioseg/  -model_path=/data/shihuazhan/output_wsi/ --cv_fold=3 --title=abmil_twh_call_baseline --model=attmil --seed=2021 --wandb;
# random 0.4
CUDA_VISIBLE_DEVICES=1 python3 main.py --project=mil_shz --datasets=camelyon16 --dataset_root=/data2/zhangxiaoxian/camelyon_all/r50_bioseg/ --model_path=/data/shihuazhan/output_wsi/ --model=random  --k_ratio=0.4 --t_steps=2 --ifTrain=1 --ifrand=0 --cv_fold=3 --seed=2021 --title=ab_mil_random_drop_k_0.4_twh_c_all --wandb;
# with noise
