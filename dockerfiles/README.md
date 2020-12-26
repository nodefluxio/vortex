## Building onnxruntime with tensorrt
From project root
```
docker build --target runtime -t onnxruntime-tensorrt:runtime -f dockerfiles/onnxruntime-tensorrt.dockerfile .
```
To build with development to enable validation and reporting:
```
docker build --target development -t onnxruntime-tensorrt:development -f dockerfiles/onnxruntime-tensorrt.dockerfile .
```
## Running onnxruntime tensorrt docker image
Run with gui and mount current directory
```
docker run -it --runtime=nvidia -v `pwd`/:/app/ --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --env="DISPLAY" --privileged --pid=host --net=host --env QT_X11_NO_MITSHM=1 onnxruntime-tensorrt bash
```