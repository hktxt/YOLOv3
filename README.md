### Description

This repository contains implementation of inference and training code of YOLOv3 in PyTorch. It is self-contained on mainstream platform and it supports custom data trianing as well. Credit to Joseph Redmon for [YOLO](https://pjreddie.com/darknet/yolo/) and the paper can be found [here](https://pjreddie.com/media/files/papers/YOLOv3.pdf). I am highly inspired by these two repositories: [yolov3](https://github.com/ultralytics/yolov3) and [yolo_v3](https://github.com/ydixon/yolo_v3). Just go to the [Dev]() for more details of YOLOv3.


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

### Training

Just open [train.ipynb]() and run all cells respectively. It will train yolov3 using COCO dataset. For custom data training, you should get your own data ready and make annotations format is the same as yolo's. Then change path in train.ipynb and run.


### Inference

Open [test.ipynb]() and run all cells respectively. It will evaluate images in images folder using the model your trained.



License
----
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)


   
## Notice

Please note, this is a research project! It should not be used as a definitive guide on object detection. The demo should be considered for research and entertainment value only.
