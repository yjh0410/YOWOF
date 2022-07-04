# Train YOWOF-R18
python train.py \
        --cuda \
        -d jhmdb \
        -v yowof-r18 \
        --num_workers 4 \
        --eval \
        --eval_epoch 2 \
        --fp16 \
        # -p ./weights/pretrained/yolof-rt-R18/yolof-rt-R18_29.2.pth

# # Train YOWOF-R50
# python train.py \
#         --cuda \
#         -d jhmdb \
#         -v yowof-r50 \
#         --num_workers 4 \
#         --eval_epoch 2 \
#         --fp16 \
#         -p ./weights/pretrained/yolof-rt-R50/yolof-rt-R50_33.9.pth

# # Train YOWOF-R50-D
# python train.py \
#         --cuda \
#         -d jhmdb \
#         -v yowof-r50-D \
#         --num_workers 4 \
#         --eval_epoch 2 \
#         --fp16 \
#         -p ./weights/pretrained/yolof-rt-R50-DC5/yolof-rt-R50-DC5_35.4.pth
