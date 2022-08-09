# Train YOWOF-R18
python train.py \
        --cuda \
        -d ucf24 \
        -v yowof-r18 \
        --num_workers 4 \
        --eval_epoch 2 \
        --eval \
        # --fp16 \
