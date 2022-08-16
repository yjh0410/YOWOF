# YOWOF
You Only Watch One-level Feature for Spatio-Temporal Action Detection

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


# Experiment
## UCF24
|    Model    |   Size   |    FPS    |    FLOPs    |   Params    |   mAP    |    Cls Accu    |    Recall    |    Weight    |
|-------------|----------|-----------|-------------|-------------|----------|----------------|--------------|--------------|
|  YOWOF-R18  |   320    |     220   |    5.4 B    |    28 M     |   80.5   |      95.0      |      95.1    |       -      |
