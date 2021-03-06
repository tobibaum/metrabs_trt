# Run Metrabs in C++ w/ Tensorrt at >30fps

This is a faster implementation of [Metrabs](https://github.com/isarandi/metrabs) under the following conditions:

- implemented in C++
- assume a single person in view
- using 3rd party yolo

The original `saved_model` approach is sped up by splitting the model into separate parts and running each in an optimized way:
1. detection (accelerated with TensorRT)
2. main backbone (e.g. efficientnet/resnet, accelerated with TensorRT)
3. metrabs head (original code)

## Requirements and Install
In order to run this whole ordeal in C++ a few libraries need to be installed. Skip to the next part, if you are only interested in the model conversion.

#### tf2onnx
```
git clone https://github.com/onnx/tensorflow-onnx.git
cd tensorflow-onnx
python setup.py install
```
needed to convert from `tf.saved_model` to `.onnx`

#### yolo tensorrt 
```
git clone https://github.com/enazoe/yolo-tensorrt.git
cd yolo-tensorrt/
mkdir build
cd build/
cmake ..
make
./yolo-trt
```

#### compile opencv with contrib packages 
`opencv_dnn` is needed in c++ implementation

```
# activate your python env
source ~/env/bin/activate
git clone https://github.com/opencv/opencv.git
...
git clone https://github.com/opencv/opencv_contrib.git
...
cd opencv
mkdir build
cd build
cmake \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D INSTALL_C_EXAMPLES=OFF \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D OPENCV_EXTRA_MODULES_PATH=<clone opencv_contrib and point to that> \
      -D PYTHON_EXECUTABLE=`which python` \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D WITH_CUBLAS=ON \
      -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -D OpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so \
      -D OpenCL_INCLUDE_DIR=/usr/local/cuda/include/ \
      ..
make -j8
sudo make install
```

#### onnx tensorrt
see https://github.com/onnx/onnx-tensorrt.git. this will provide necessary lib `libnvonnxparser.so`

#### tiny tensorrt
see https://github.com/zerollzeng/tiny-tensorrt.git. Convenient C++ wrapper to run TensorRT models.

------------------
## Convert model
the conversions happens in multiple steps inside and out of a docker container.

the base models can be downloaded from the [metrabs-modelzoo](https://github.com/isarandi/metrabs/blob/master/docs/MODELS.md)

### 1. Split out backbone and metrab head
run the notebook `convert_tensorrt.ipynb`. change the parameters in the first cell according to your setup and model you want to convert:
```
# input names
model_name = 'efficientnetv2-l'
model_folder = '/disk/apps/metrabs/models/metrabs_eff2l_y4/'

# output names
base_fold = "mods/effnet-l/"
```
run all cells. this should create the following files:

```
<base_fold>
????????? bbone
??????? ????????? assets
??????? ????????? saved_model.pb
??????? ????????? variables
???????     ????????? variables.data-00000-of-00001
???????     ????????? variables.index
????????? metrab_head
    ????????? assets
    ????????? saved_model.pb
    ????????? variables
        ????????? variables.data-00000-of-00001
        ????????? variables.index
```

### 2. convert backbone saved-model to onnx
```
python -m tf2onnx.convert --saved-model mods/effnet-l/bbone --output mods/effnet-l/bbone.onnx
```

### 3. acclerate backbone onnx with tensorrt plan
I only got this working from within a docker :/ installing TensorRT natively is very complicated.
```
./tools/run_docker
root@41ebf45bdca1:/workspace# jupyter lab
```
the notebook is hosted at localhost:888**9**

run `onnx_trt_plan.ipynb` to convert 

#### 4. symbolic link the configs from yolo-tensorrt installed above
```
cd mods/effnet-l
ln -s <path to yolo-tensorrt>/configs
```

#### 5. final structure
you should now have the following folders and files:
```
effnet-l/
????????? bbone
??????? ????????? assets
??????? ????????? saved_model.pb
??????? ????????? variables
???????     ????????? variables.data-00000-of-00001
???????     ????????? variables.index
????????? bbone.onnx
????????? bbone.plan
????????? configs -> ../effnet-s/configs
????????? metrab_head
    ????????? assets
    ????????? saved_model.pb
    ????????? variables
        ????????? variables.data-00000-of-00001
        ????????? variables.index
```

------------------------

## Run Inference
congrats if you made it this far! let me know what worked.

The attached c++ code runs inference on a video and prints out the milliseconds spent on each frame.
#### build the c++ code
you will have to change some of the hardcode from `build.sh` to your local paths of the above installed stuff.
```
cd cpp
./build.sh
```

#### run
```
./metrabs <path-to-model-structure> <video-path>
```

#### caveats
- For now, the demo will show the cropped and resized frame with the resulting keypoints of 32 joints. I leave the reconstruction of the keypoint coordinates into the original image frame to the reader.

- I have only implemented the 2D parts of the model. in order to extract the full 3D model, you will have to modify the `metrabs_head` part in `convert_tensorrt.ipynb`, according to the code in [`metrabs/metrabs.py`](https://github.com/isarandi/metrabs/blob/master/src/models/metrabs.py#L33)

- different model types have different bottleneck sizes before the metrabs header. this is currently clumsily implemented using a command line parameter: `./metrabs <some resnet model> <video> 2048`
