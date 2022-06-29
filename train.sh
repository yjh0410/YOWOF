python train.py \
        --cuda \
        -d jhmdb \
        -v yowof-r50 \
        --num_workers 4 \
        --eval_epoch 2 \
        --fp16 \
        -p ./weights/pretrained/yolof-rt-R18/yolof-rt-R18_29.2.pth