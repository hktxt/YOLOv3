### Description

This repository contains implementation of inference and training code of YOLOv3 in PyTorch. The code can work on Linux, MacOS and Windows and it supports custom data trianing as well. Credit to Joseph Redmon for [YOLO](https://pjreddie.com/darknet/yolo/) and the paper can be found [here](https://pjreddie.com/media/files/papers/YOLOv3.pdf). I am highly inspired by these two repositories: [yolov3](https://github.com/ultralytics/yolov3) and [yolo_v3](https://github.com/ydixon/yolo_v3). Just go to the [Dev]() for more details of YOLOv3.


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

Just open train.ipynb and run all cells respectively. It will train yolov3 using COCO dataset. For custom data training, you should get your data ready and make annotations format indentical with yolo. Then change path in train.ipynb and run.


### Inference



You can also:
  - Import and save files from GitHub, Dropbox, Google Drive and One Drive
  - Drag and drop markdown and HTML files into Dillinger
  - Export documents as Markdown, HTML and PDF

Markdown is a lightweight markup language based on the formatting conventions that people naturally use in email.  As [John Gruber] writes on the [Markdown site][df1]

> The overriding design goal for Markdown's
> formatting syntax is to make it as readable
> as possible. The idea is that a
> Markdown-formatted document should be
> publishable as-is, as plain text, without
> looking like it's been marked up with tags
> or formatting instructions.

This text you see here is *actually* written in Markdown! To get a feel for Markdown's syntax, type some text into the left window and watch the results in the right.


### Installation

Dillinger requires [Node.js](https://nodejs.org/) v4+ to run.

Install the dependencies and devDependencies and start the server.

```sh
$ cd dillinger
$ npm install -d
$ node app
```

For production environments...

```sh
$ npm install --production
$ NODE_ENV=production node app
```


### Development

Want to contribute? Great!

Dillinger uses Gulp + Webpack for fast developing.
Make a change in your file and instantanously see your updates!

Open your favorite Terminal and run these commands.

First Tab:
```sh
$ node app
```

Second Tab:
```sh
$ gulp watch
```

(optional) Third:
```sh
$ karma test
```
#### Building for source
For production release:
```sh
$ gulp build --prod
```
Generating pre-built zip archives for distribution:
```sh
$ gulp build dist --prod
```
### Docker
Dillinger is very easy to install and deploy in a Docker container.

By default, the Docker will expose port 8080, so change this within the Dockerfile if necessary. When ready, simply use the Dockerfile to build the image.

```sh
cd dillinger
docker build -t joemccann/dillinger:${package.json.version}
```
This will create the dillinger image and pull in the necessary dependencies. Be sure to swap out `${package.json.version}` with the actual version of Dillinger.

Once done, run the Docker image and map the port to whatever you wish on your host. In this example, we simply map port 8000 of the host to port 8080 of the Docker (or whatever port was exposed in the Dockerfile):

```sh
docker run -d -p 8000:8080 --restart="always" <youruser>/dillinger:${package.json.version}
```

Verify the deployment by navigating to your server address in your preferred browser.

```sh
127.0.0.1:8000
```



License
----
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)


   
# Notation

Please note, this is a research project! It should not be used as a definitive guide on object detection. The demo should be considered for research and entertainment value only.
