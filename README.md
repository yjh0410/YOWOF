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
For UCF24:

Link: https://drive.google.com/file/d/1Dwh90pRi7uGkH5qLRjQIFiEmMJrAog5J/view?usp=sharing

For JHMDB21: 

Link: https://drive.google.com/file/d/15nAIGrWPD4eH3y5OTWHiUbjwsr-9VFKT/view?usp=sharing

## BaiduYunDisk
For UCF24:

Link: https://pan.baidu.com/s/11GZvbV0oAzBhNDVKXsVGKg

Password: hmu6 

For JHMDB21: 

Link: https://pan.baidu.com/s/1HSDqKFWhx_vF_9x6-Hb8jA 

Password: tcjd 

# Visualization
Coming soon ...

## AVA
You can use instructions from [here](https://github.com/yjh0410/AVA_Dataset) to prepare **AVA** dataset.

# Experiment
* Frame-mAP@0.5 IoU on UCF24

|    Model    |   Clip  |    FPS    |  FLOPs  |  mAP   |  Cls Accu  |  Recall  |  Weight  |
|-------------|---------|-----------|---------|--------|------------|----------|----------|
|  YOWOF-R18  |    8    |     220   |         |        |            |          |    -     |
|  YOWOF-R18  |   16    |     220   |         |        |            |          | [github]() |
|  YOWOF-R18  |   32    |     220   |         |        |            |          |    -     |


* Frame-mAP@0.5 IoU on AVA_v2.2

|     Model     |   Clip  |    FPS    |  FLOPs  |  mAP   |  Weight  |
|---------------|---------|-----------|---------|--------|----------|
|   YOWOF-R18   |    8    |    220    |         |        |    -     |
|   YOWOF-R18   |   16    |    220    |         |        |    -     |
|   YOWOF-R18   |   32    |    220    |         |        |    -     |
|   YOWOF-R50   |    8    |    125    |         |        |    -     |
|   YOWOF-R50   |   16    |    125    |         |        | [github]() |
|   YOWOF-R50   |   32    |    125    |         |        |    -     |
| YOWOF-R50-DC5 |         |           |         |        |    -     |
|  YOWOF-RX101  |         |           |         |        |    -     |

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

* on AVA_v2.2

```Shell
python eval.py --cuda -d ucf24 -v yowof-r50 --weight path/to/weight
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
