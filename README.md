# Faceblur Server

This repository provides a very simple Flask server that can blur faces in images.
[yolov8-face](https://github.com/derronqi/yolov8-face/tree/main) by [derronqi](https://github.com/derronqi) is used as the underlying model.

## Getting started

```shell
docker build -t faceblur . && docker run -d -p 5000:5000 -t faceblur
```

After starting the server the `/v1/blur/image` endpoint is available to process images.
The endpoint returns the original image with ideally all faces blurred.
