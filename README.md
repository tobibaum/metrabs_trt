# Run Metrabs in C++ w/ Tensorrt at >=30fps

This is a faster implementation of (Metrabs)[https://github.com/isarandi/metrabs] under the following conditions:

- implemented in C++
- assume a single person in view
- using external yolo

## Requirements and Install
(tf2onnx)[https://github.com/onnx/tensorflow-onnx.git]

#### yolo tensorrt
https://github.com/enazoe/yolo-tensorrt.git

## Convert model
the conversions happens in multiple steps inside and out of a docker container.

### 1. Split out backbone and metrab head 

## Run Inference
congrats if you made it this far!
