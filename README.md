# Run Metrabs in C++ w/ Tensorrt at >=30fps

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
├── bbone
│   ├── assets
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
└── metrab_head
    ├── assets
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index
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



## Run Inference
congrats if you made it this far!
