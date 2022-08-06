# Train YOWOF-R18
python train.py \
        --cuda \
        -d jhmdb \
        -v yowof-d19 \
        --num_workers 4 \
        --eval_epoch 2 \
        --eval \
