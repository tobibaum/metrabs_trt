lib_path=/disk/apps/tensorflow_c/lib
inc_path=/disk/apps/tensorflow_c/include
inc_cuda=/usr/local/cuda-11.4/targets/x86_64-linux/include/
yolo_trt=/home/tobi/apps/yolo-tensorrt

g++ -o metrabs main.cpp `pkg-config opencv --cflags --libs` \
        -I${inc_path} -I${inc_cuda} -I/usr/include/eigen3 \
        -I${yolo_trt}/modules -I${yolo_trt}/extra \
        -L${lib_path} -L${yolo_trt}/build \
        -ltensorflow -ltinytrt -lnvonnxparser -ldetector \
        -lopencv_dnn -lopencv_highgui
