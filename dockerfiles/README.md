## Building onnxruntime with tensorrt
From project root
```
docker build -t onnxruntime-tensorrt -f dockerfiles/onnxruntime-tensorrt.dockerfile .
```
## Running onnxruntime tensorrt docker image
Run with gui and mount current directory
```
docker run -it --runtime=nvidia -v `pwd`/:/app/ --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --env="DISPLAY" --net=host onnxruntime-tensorrt --env QT_X11_NO_MITSHM=1 bash
```