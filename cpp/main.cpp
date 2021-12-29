#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include "tensorflow/c/c_api.h"

#include "Trt.h"

//#include "tensorflow/core/public/session.h"
//#include "tensorflow/core/framework/tensor.h"
//#include "tensorflow/core/platform/env.h"

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

std::vector<float> convert_mat_to_fvec(cv::Mat mat){
    std::vector<float> array;
    if (mat.isContinuous()) {
        array.assign((float*)mat.data, (float*)mat.data + mat.total()*mat.channels());
    } else {
        for (int i = 0; i < mat.rows; ++i) {
          array.insert(array.end(), mat.ptr<float>(i), mat.ptr<float>(i)+mat.cols*mat.channels());
        }
    }
    return array;
}

void NoOpDeallocator(void* data, size_t a, void* b) {}
int main(int argc, char** argv)
{
    if(argc < 3){
        std::cout << "usage: ./metrabs <model-dir> <img-url>" << std::endl;
        return 0;
    }
    cv::Mat image = cv::imread(argv[2]);

    // batchsize!
    int NumInputs = 1;
    int NumOutputs = 1;

    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_time = start_time;
    auto now = std::chrono::high_resolution_clock::now();
    float durr = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

    Trt* onnx_net = new Trt();
    onnx_net->CreateEngine("../mods/effnet.onnx", "../mods/effnet.plan", 1, 0);
    int inputBindIndex = 0;
    int outputBindIndex = 1;

    cv::Mat img_f32;
    cv::cvtColor(image, image, CV_BGR2RGB);
    image.convertTo(img_f32, CV_32FC3);

    img_f32 = img_f32/256.f;

    //std::vector<float> input(786432/4);
    std::vector<float> output(327680/4);

    //input = img_f32.data;


    std::vector<float> fvec;
    for(int i=0; i<100;i++){
        //onnx_net->CopyFromHostToDevice(input, inputBindIndex);
        fvec = convert_mat_to_fvec(img_f32);
        onnx_net->CopyFromHostToDevice(fvec, inputBindIndex);
        onnx_net->Forward();
        onnx_net->CopyFromDeviceToHost(output, outputBindIndex);

        now = std::chrono::high_resolution_clock::now();
        durr = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
        last_time = now;
        std::cout << durr << std::endl;
    }

    for(int i=0;i<100;i++){
        std::cout << fvec[i] << " " ;
    }
    std::cout << std::endl;
    for(int i=0;i<100;i++){
        std::cout << output[i] << " " ;
    }
    std::cout << std::endl;

    //return 1;

    //********* Read model
    TF_Graph* Graph = TF_NewGraph();
    TF_Status* Status = TF_NewStatus();
    TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
    //SessionOpts->set_per_process_gpu_memory_fraction(0.333);
    TF_Buffer* RunOpts = NULL;

    const char* saved_model_dir = argv[1];
    const char* tags = "serve";

    int ntags = 1;
    TF_Session* Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);

    if(TF_GetCode(Status) == TF_OK)
        printf("TF_LoadSessionFromSavedModel OK\n");
    else
        printf("%s",TF_Message(Status));

    //****** Get input tensor
    TF_Output* Input = (TF_Output*)malloc(sizeof(TF_Output) * NumInputs );
    TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_feature"), 0};

    if(t0.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName serving_default_feature\n");
    else
        printf("TF_GraphOperationByName serving_default_feature is OK\n");

    Input[0] = t0;

    //********* Get Output tensor
    TF_Output* Output = (TF_Output*)malloc(sizeof(TF_Output) * NumOutputs);
    TF_Output t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};

    if(t2.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
    else
        printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");

    Output[0] = t2;

    //********* Allocate data for inputs & outputs
    TF_Tensor** InputValues  = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumInputs);
    TF_Tensor** OutputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumOutputs);

    cv::resize(image, image, cv::Size(256, 256));
    int wd = image.rows;
    int ht = image.cols;

    for(int cnt=0; cnt<100; cnt++){
        int ndims = 4;
        //int64_t dims[] = {wd, ht, 3};
        //unsigned char* data = image.data;
        int64_t dims[] = {1, 8, 8, 1280};
        //float* data = &output[0];
        int ndata = 8*8*1280*4;

        float data[ndata];
        for(int i=0;i<ndata;i++){
            //data[i] = .2f;
            data[i] = output[i];
        }

        //int ndata = wd*ht*3*4;
        //TF_Tensor* int_tensor = TF_NewTensor(TF_UINT8, dims, ndims, data, ndata, &NoOpDeallocator, 0);
        TF_Tensor* int_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, data, ndata, &NoOpDeallocator, 0);

        if (int_tensor != NULL)
            printf("TF_NewTensor is OK\n");
        else
            printf("ERROR: Failed TF_NewTensor\n");

        InputValues[0] = int_tensor;

        // Run the Session
        printf("run sesh...\n");
        TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0, NULL , Status);
        printf("done\n");

        if(TF_GetCode(Status) == TF_OK)
            printf("Session is OK\n");
        else
            printf("%s",TF_Message(Status));

        now = std::chrono::high_resolution_clock::now();
        durr = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
        last_time = now;
        std::cout << durr << std::endl;
    }

    // Free memory
    TF_DeleteGraph(Graph);
    TF_DeleteSession(Session, Status);
    TF_DeleteSessionOptions(SessionOpts);
    TF_DeleteStatus(Status);

    void* buff = TF_TensorData(OutputValues[0]);
    float* offsets = (float*)buff;

    Eigen::Map<Eigen::MatrixXf> res_mat(offsets, 2, 32);
    printf("Result Tensor :\n");
    std::cout << res_mat.transpose() << std::endl;
    //for(int i=0; i<100; i++){
    //    printf("%f\n", offsets[i]);
    //}
    return 0;
}
