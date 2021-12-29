lib_path=/disk/apps/tensorflow_c/lib
inc_path=/disk/apps/tensorflow_c/include
inc_cuda=/usr/local/cuda-11.4/targets/x86_64-linux/include/

g++ -o metrabs main.cpp `pkg-config opencv --cflags --libs` -I${inc_path} -I${inc_cuda} -I/usr/include/eigen3 -L${lib_path} -ltensorflow -ltinytrt -lnvonnxparser
