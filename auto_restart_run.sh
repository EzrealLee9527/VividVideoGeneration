while true; do
    rlaunch -P 8 --charged-group=monet_video --gpu=8 --cpu=32 --memory=512000 --private-machine=yes -- bash run.sh configs/accelerate_accumulation_steps10.yaml configs/training/train_stage12_w_imageencoder_celebv.yaml 64
    # 检查进程退出状态
    status=$?
    if [ $status -eq 0 ]; then
        echo "Process completed successfully"
        break
    else
        echo "Process terminated with status $status, restarting..."
        sleep 1
    fi
done






