# Train YOWOF-R18
python train.py \
        --cuda \
        -d ucf24 \
        -v yowof-vgg16 \
        --num_workers 4 \
        --eval_epoch 1 \
        # --eval \
        # --fp16 \
