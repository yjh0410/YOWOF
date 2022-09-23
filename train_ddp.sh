# Train YOWO-D19
python -m torch.distributed.run --nproc_per_node=2 train.py \
                                                    -dist \
                                                    --cuda \
                                                    -d ava_v2.2 \
                                                    -v yowof-rx101 \
                                                    --num_workers 8 \
                                                    --eval_epoch 1 \
                                                    --eval \
                                                    # --fp16 \
