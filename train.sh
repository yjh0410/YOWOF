python train.py \
        --cuda \
        -d jhmdb \
        -v yowof-r50 \
        --num_workers 4 \
        --eval_epoch 2 \
        -p ./weights/pretrained/yolof-rt-R50/yolof-rt-R50_33.9.pth