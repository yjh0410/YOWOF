# YOWOF
You Only Watch One Frame (YOWOF) for Online Spatio-Temporal Action Detection

# Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n yowof python=3.6
```

- Then, activate the environment:
```Shell
conda activate yowof
```

- Requirements:
```Shell
pip install -r requirements.txt 
```

# Dataset
You can download **UCF24** and **JHMDB21** from the following links:

## Google Drive
## UCF101-24:
* Google drive

Link: https://drive.google.com/file/d/1Dwh90pRi7uGkH5qLRjQIFiEmMJrAog5J/view?usp=sharing

* BaiduYun Disk

Link: https://pan.baidu.com/s/11GZvbV0oAzBhNDVKXsVGKg

Password: hmu6 

## JHMDB21: 
* Google drive

Link: https://drive.google.com/file/d/15nAIGrWPD4eH3y5OTWHiUbjwsr-9VFKT/view?usp=sharing

* BaiduYun Disk

Link: https://pan.baidu.com/s/1HSDqKFWhx_vF_9x6-Hb8jA 

Password: tcjd 

## AVA
You can use instructions from [here](https://github.com/yjh0410/AVA_Dataset) to prepare **AVA** dataset.

# Visualization
* UCF101-24

Coming soon ...

* AVA

Coming soon ...


# Experiment
* Frame-mAP@0.5 IoU on UCF24

|    Model    |   Clip  |    FPS    |  FLOPs  |  mAP   |  Cls Accu  |  Recall  |   Weight   |   log   |
|-------------|---------|-----------|---------|--------|------------|----------|------------|---------|
|  YOWOF-R18  |    8    |     225   |  4.9 B  |  80.7  |    93.4    |   95.6   | [ckpt](https://github.com/yjh0410/YOWOF/releases/download/yowof-weight/yowof-r18_epoch_4_93.4_95.6_80.7.pth) | [log](https://github.com/yjh0410/YOWOF/releases/download/yowof-weight/YOWOF-R18-K-8.txt) |
|  YOWOF-R18  |   16    |     225   |  4.9 B  |  81.0  |    94.0    |   96.0   | [ckpt](https://github.com/yjh0410/YOWOF/releases/download/yowof-weight/yowof-r18_epoch_3_94.0_96.0_81.0.pth) | [log](https://github.com/yjh0410/YOWOF/releases/download/yowof-weight/YOWOF-R18-K-16.txt) |
|  YOWOF-R18  |   16    |     225   |  4.9 B  |  82.4* |    94.4*   |   96.1*  | [ckpt](https://github.com/yjh0410/YOWOF/releases/download/yowof-weight/yowof-r18_epoch_5_94.4_96.1_82.4.pth) | [log](https://github.com/yjh0410/YOWOF/releases/download/yowof-weight/YOWOF-R18-K16-UCF24.txt) |
|  YOWOF-R18  |   32    |     225   |  4.9 B  |  81.5  |    94.6    |   96.3   | [ckpt](https://github.com/yjh0410/YOWOF/releases/download/yowof-weight/yowof-r18_epoch_2_94.6_96.3_81.5.pth) | [log](https://github.com/yjh0410/YOWOF/releases/download/yowof-weight/YOWOF-R18-K-32.txt) |

\* indicates that we are temporarily unable to reproduce this result, although we provide the ```ckpt``` file
and ```log``` file. We are trying to fix this bug.

* Frame-mAP@0.5 IoU on AVA_v2.2

|     Model     |   Clip  |    FPS    |  FLOPs  |  mAP   |   Weight   |   log   |
|---------------|---------|-----------|---------|--------|------------|---------|
|   YOWOF-R18   |   16    |    220    |         |        | [ckpt]() | [log]() |
|   YOWOF-R18   |   32    |    220    |         |        | [ckpt]() | [log]() |
|   YOWOF-R50   |   16    |    125    |         |        | [ckpt]() | [log]() |
|   YOWOF-R50   |   32    |    125    |         |        | [ckpt]() | [log]() |
| YOWOF-R50-DC5 |   16    |           |         |        | [ckpt]() | [log]() |
|  YOWOF-RX101  |   16    |           |         |        | [ckpt]() | [log]() |

## Train YOWOF
### Train yowof-r18 on UCF24

```Shell
python train.py --cuda -d ucf24 -v yowof-r18 --num_workers 4 --eval_epoch 1 --eval
```

or you can just run the script:

```Shell
sh train_ucf.sh
```

### Train yowof-r50 on AVA_v2.2

```Shell
python train.py --cuda -d ava_v2.2 -v yowof-r50 --num_workers 4 --eval_epoch 1 --eval
```

or you can just run the script:

```Shell
sh train_ava.sh
```

## Test YOWOF
### Test yowof-r18 on UCF24

* run yowof with *clip* inference mode

```Shell
python test.py --cuda -d ucf24 -v yowof-r18 --weight path/to/weight --inf_mode clip --show
```

* run yowof with *stream* inference mode

```Shell
python test.py --cuda -d ucf24 -v yowof-r18 --weight path/to/weight --inf_mode stream --show
```

### Test yowof-r50 on AVA_v2.2

* run yowof with *clip* inference mode

```Shell
python test.py --cuda -d ava_v2.2 -v yowof-r50 --weight path/to/weight --inf_mode clip --show
```

* run yowof with *stream* inference mode

```Shell
python test.py --cuda -d ava_v2.2 -v yowof-r50 --weight path/to/weight --inf_mode stream --show
```

## Evaluate YOWOF
* on UCF24

```Shell
python eval.py --cuda -d ucf24 -v yowof-r18 --weight path/to/weight
```

Our SOTA results on UCF24:
```Shell
AP: 78.15% (1)
AP: 97.13% (10)
AP: 84.41% (11)
AP: 63.61% (12)
AP: 72.00% (13)
AP: 91.29% (14)
AP: 86.57% (15)
AP: 91.28% (16)
AP: 76.65% (17)
AP: 94.29% (18)
AP: 96.24% (19)
AP: 45.84% (2)
AP: 96.02% (20)
AP: 83.92% (21)
AP: 79.63% (22)
AP: 54.22% (23)
AP: 90.16% (24)
AP: 85.96% (3)
AP: 79.05% (4)
AP: 68.01% (5)
AP: 95.18% (6)
AP: 92.14% (7)
AP: 89.02% (8)
AP: 86.96% (9)
mAP: 82.41%
```

* on AVA_v2.2

```Shell
python eval.py --cuda -d ucf24 -v yowof-r50 --weight path/to/weight
```

Our SOTA results on AVA_v2.2:
```Shell
Coming soon ...
```
## Detect AVA video

```Shell
python test_video_ava.py --cuda -d ucf24 -v yowof-r50 --weight path/to/weight --video ava/video/name
```

## Demo
* detection action instances with UCF24 labels

We provide some test videos of UCF24 in ```dataset/demo/ucf24_demo/```.

```Shell
python demo.py --cuda -d ucf24 -v yowof-r18 --weight path/to/weight --video ./dataset/demo/ucf24_demo/v_Basketball_g01_c02.mp4
```

* detection action instances with AVA labels

```Shell
python demo.py --cuda -d ava_v2.2 -v yowof-r50 --weight path/to/weight --video path/to/video
```
