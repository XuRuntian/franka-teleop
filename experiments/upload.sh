#!/bin/bash

# 本地任务根目录
local_dir="/media/franka/2248-D4D5/hybridVLA/franka/spacemouse_teleop/pick_and_place"

# 远程主机名
remote_host="baai-79"

# 远程根目录
remote_dir="/mnt/hpfs/baaiei/lzy/pick_and_place"

# 遍历本地目录并上传所有的.npy文件
find "$local_dir" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
    echo $dir
    find "$dir" -maxdepth 1 -type f -name "*.npy" | while read -r file; do
        # 获取文件的相对路径
        echo $file
        relative_path="${file#$local_dir/}"

        # 计算文件的目标路径
        remote_file="$remote_dir/$relative_path"

        # 获取文件的父目录
        remote_dirname=$(dirname "$remote_file")

        # 在远程服务器上创建目录（如果不存在）
        ssh "$remote_host" "mkdir -p \"$remote_dirname\""

        # 使用scp上传文件
        scp "$file" "$remote_host:$remote_file"
    done
done
