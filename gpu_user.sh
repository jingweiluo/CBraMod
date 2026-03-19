#!/usr/bin/env bash

declare -A gpu_map
declare -A user_mem_mib

# 记录 GPU uuid -> index
while IFS=',' read -r idx uuid; do
    idx=$(echo "$idx" | xargs)
    uuid=$(echo "$uuid" | xargs)
    gpu_map["$uuid"]="$idx"
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)

# 统计全部 GPU 总显存（MiB）
total_gpu_mem_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{sum+=$1} END{print sum}')

echo "================ Per-Process GPU Usage ================"
printf "%-6s %-8s %-12s %-10s %s\n" "GPU" "PID" "USER" "GPU_MEM" "CMD"
printf "%-6s %-8s %-12s %-10s %s\n" "------" "--------" "------------" "----------" "----------------"

tmpfile=$(mktemp)

nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null | \
while IFS=',' read -r gpu_uuid pid used_mem; do
    gpu_uuid=$(echo "$gpu_uuid" | xargs)
    pid=$(echo "$pid" | xargs)
    used_mem=$(echo "$used_mem" | xargs)

    user=$(ps -o user= -p "$pid" 2>/dev/null | xargs)
    cmd=$(ps -o comm= -p "$pid" 2>/dev/null | xargs)

    [ -z "$user" ] && user="N/A"
    [ -z "$cmd" ] && cmd="N/A"
    [ -z "$used_mem" ] && used_mem=0

    printf "%-6s %-8s %-12s %-10s %s\n" \
        "${gpu_map[$gpu_uuid]:-N/A}" "${pid:-N/A}" "${user}" "${used_mem}MiB" "${cmd}"

    echo "${user},${used_mem}" >> "$tmpfile"
done | sort -n

echo
echo "================ Per-User GPU Summary ================"
printf "%-12s %-12s %-12s\n" "USER" "GPU_MEM(GB)" "RATIO"
printf "%-12s %-12s %-12s\n" "------------" "------------" "------------"

if [ -s "$tmpfile" ]; then
    awk -F',' -v total="$total_gpu_mem_mib" '
    {
        mem[$1] += $2
    }
    END {
        for (u in mem) {
            gb = mem[u] / 1024
            ratio = mem[u] / total * 100
            printf "%-12s %-12.2f %-11.2f%%\n", u, gb, ratio
        }
    }' "$tmpfile" | sort -k2 -nr
else
    echo "No active GPU compute process found."
fi

rm -f "$tmpfile"