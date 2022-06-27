python train.py \
        --cuda \
        -d ucf24 \
        -v yowof-r18 \
        --num_workers 4 \
        --eval_epoch 2 \
        -p ./weights/pretrained/yolof-rt-R18/yolof-rt-R18_29.2.pth