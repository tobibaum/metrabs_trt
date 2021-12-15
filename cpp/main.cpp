#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include "tensorflow/c/c_api.h"

//#include "tensorflow/core/public/session.h"
//#include "tensorflow/core/framework/tensor.h"
//#include "tensorflow/core/platform/env.h"

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

void NoOpDeallocator(void* data, size_t a, void* b) {}
int main(int argc, char** argv)
{
    if(argc < 2){
        std::cout << "usage: ./metrabs <model-dir>" << std::endl;
        return 0;
    }
    // batchsize!
    int NumInputs = 1;
    int NumOutputs = 1;

    //tensorflow::SessionOptions session_options_;
    //tensorflow::RunOptions run_options_;
    //tensorflow::SavedModelBundle model_;

    //auto status = tensorflow::LoadSavedModel(session_options_,
    //                                         run_options_,
    //                                         path_to_model_,
    //                                         {tensorflow::kSavedModelTagServe},
    //                                         &model_);
    //if (!status.ok()) {
    //    std::cerr << "Failed to load model: " << status;
    //return;
    //}

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
    TF_Output* Input = (TF_Output*)malloc(sizeof(TF_Output) * NumInputs);
    TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_image"), 0};

    if(t0.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName serving_default_image\n");
    else
        printf("TF_GraphOperationByName serving_default_image is OK\n");

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

    cv::Mat image = cv::imread("image.png");
    int wd = image.rows;
    int ht = image.cols;

    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_time = start_time;
    auto now = std::chrono::high_resolution_clock::now();
    float durr = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

    for(int cnt=0; cnt<100; cnt++){
        int ndims = 3;
        int64_t dims[] = {wd, ht, 3};
        unsigned char* data = image.data;

        int ndata = wd*ht*3;
        TF_Tensor* int_tensor = TF_NewTensor(TF_UINT8, dims, ndims, data, ndata, &NoOpDeallocator, 0);

        if (int_tensor != NULL)
            printf("TF_NewTensor is OK\n");
        else
            printf("ERROR: Failed TF_NewTensor\n");

        InputValues[0] = int_tensor;

        // Run the Session
        TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0, NULL , Status);

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

    Eigen::Map<Eigen::MatrixXf> res_mat(offsets, 3, 122);
    printf("Result Tensor :\n");
    std::cout << res_mat.transpose() << std::endl;
    //for(int i=0; i<100; i++){
    //    printf("%f\n", offsets[i]);
    //}
    return 0;
}
