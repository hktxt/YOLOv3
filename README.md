### Description

This repository contains implementation of training and inference code of YOLOv3 in PyTorch. It is self-contained on mainstream platform and it supports custom data trianing as well. Credit to Joseph Redmon for [YOLO](https://pjreddie.com/darknet/yolo/) and the paper can be found [here](https://pjreddie.com/media/files/papers/YOLOv3.pdf). I am highly inspired by these two repositories: [yolov3](https://github.com/ultralytics/yolov3) and [yolo_v3](https://github.com/ydixon/yolo_v3). It is recommended to read their codes to see the specific implementation steps and logic. Also you can just go to the [Dev]() for more details of YOLOv3.


### Requirements

  - python >= 3.6
  - numpy
  - torch >= 1.0.0
  - opencv
  - json
  - matplotlib
  - CUDA(optional)


### Download Data
```sh
$ cd data/
$ bash get_coco_dataset.sh
```

### Pretrained Weights
The pretrained weights can be found on [Google Drive](https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI), download yolov3-tiny.conv.15 & darknet53.conv.74 and place under the weights folder. Actually, they can be downloaded automatically. However, if you want to train the model from scratch, you can skip this step.

### Training
*  **COCO dataset**Just open [train.ipynb]() and run all cells respectively. It will train yolov3 using COCO dataset. Using FROM_SCRATCH to control whether train from scratch.
* **Custom dataset** For custom data training, you should get your own data ready and make annotations format is the same as yolo's. Then change path in train.ipynb and run it.


### Inference

Open [test.ipynb]() and run all cells respectively. It will evaluate images in sample folder using the model your have trained.

<img src="sample/Adrian.jpg" width="400"> <img src="sample/dog.jpg" width="400">

License
----
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)


   
## Notice

Please note, this is a research project! It should not be used as a definitive guide on object detection. Many engineering features have not been implemented. The demo should be considered for research and entertainment value only.
