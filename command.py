import subprocess
import time

commands = [
    # "export NCCL_DEBUG=INFO",
    # "export TORCH_DISTRIBUTED_DEBUG=DETAIL",
    # "export NCCL_ASYNC_ERROR_HANDLING=1",

    # f"CUDA_VISIBLE_DEVICES=3 \
    # torchrun --nproc_per_node=2 \

    # f"CUDA_VISIBLE_DEVICES=5 \
    # python pretrain_main.py \
    # --pretrain_dataset Chisco,TUEG,Lee2019_SSVEP,Lee2019_MI,BI2015a,Cho2017,Zhou2016,AlexMI,BNCI2014_008,Kalunga2016,BNCI2015_001,Nakanishi2015 \
    # --epochs 500 --seed 3 \
    # --mask_ratio 0.5 \
    # --train_mode recon \
    # --use_channel_subset \
    # --lambda_codebook 0 \
    # --foundation_dir pretrained_weights_recon_3_6 \
    # --batch_size 16",
]

downstream_tasks = [
    "FACED",
    "PhysioNet-MI",
    "BCIC2020-3",
    "BCIC-IV-2a",
]

gpu_id = 2
seed = 3
batch_size = 64
foundation_ckpt = "pretrained_weights_recon_3_6/best_epoch150_tail0.537500.pth"

# pretrained_weights_both_ctx_contra/best_epoch7_tail31.693869.pth
# pretrained_weights_both/best_epoch197_tail0.008238.pth

for ds in downstream_tasks:
    cmd = f"""
    CUDA_VISIBLE_DEVICES={gpu_id} \
    python finetune_main.py \
    --downstream_dataset {ds} \
    --seed {seed} \
    --use_pretrained_weights \
    --foundation_dir {foundation_ckpt} \
    --batch_size {batch_size}
    """
    commands.append(cmd)

# 重试次数设置
max_retries = 1

# 依次执行每条指令
for cmd in commands:
    print(f"Running: {cmd}")
    success = False
    for attempt in range(1, max_retries + 1):
        try:
            # 注意：check=True，表示如果命令返回非0（错误），会抛异常
            subprocess.run(cmd, shell=True, check=True)
            success = True
            print(f"Success on attempt {attempt}: {cmd}")
            break  # 成功就跳出 retry 循环
        except subprocess.CalledProcessError:
            print(f"Attempt {attempt} failed for: {cmd}")
            if attempt < max_retries:
                time.sleep(5)  # 重试前等待几秒（可选）
            else:
                print(f"All {max_retries} attempts failed for: {cmd}")
    print()

print("All commands finished.")