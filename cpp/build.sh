lib_path=/disk/apps/tensorflow_c/lib
inc_path=/disk/apps/tensorflow_c/include

g++ -o metrabs main.cpp `pkg-config opencv --cflags --libs` -I${inc_path} -I/usr/include/eigen3 -L${lib_path} -ltensorflow
