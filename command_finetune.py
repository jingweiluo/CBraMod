import subprocess
import time

commands = []

downstream_tasks = [
    "FACED",
    "PhysioNet-MI",
    "BCIC2020-3",
    "BCIC-IV-2a",
    "Mumtaz2016",
    "MentalArithmetic",
    "SHU-MI",
    # "CHB-MIT",
]

gpu_id = 1
seed = 3
batch_size = 8
d_model = 400
log_file_name = "test_results_recon_325_125epochs.txt" # text_results.txt
foundation_ckpt = "pretrained_weights_recon_3_19/last.pth"

# pretrained_weights_both/best_epoch196_loss3.088265.pth
# pretrained_weights_both_ctx_contra/best_epoch7_tail31.693869.pth
# pretrained_weights_both/best_epoch197_tail0.008238.pth

for ds in downstream_tasks:
    cmd = f"""
    CUDA_VISIBLE_DEVICES={gpu_id} \
    python finetune_main.py \
    --downstream_dataset {ds} \
    --seed {seed} \
    --d_model {d_model} \
    --use_pretrained_weights \
    --foundation_dir {foundation_ckpt} \
    --batch_size {batch_size} \
    --log_file_name {log_file_name}
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