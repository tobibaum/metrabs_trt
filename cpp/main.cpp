#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include "tensorflow/c/c_api.h"

#include "Trt.h"

#include <Eigen/Geometry>
//#include <opencv2/opencv.hpp>

#include "class_detector.h"

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

class TFModelLoader {
private:
    // batchsize!
    int NumInputs = 1;
    int NumOutputs = 1;
    TF_Graph* Graph;
    TF_Status* Status;
    TF_SessionOptions* SessionOpts;
    TF_Output* Output;
    TF_Output* Input;
    TF_Tensor** InputValues;
    TF_Tensor** OutputValues;
    TF_Session* Session;

public:
    ~TFModelLoader(){
        // Free memory
        TF_DeleteGraph(Graph);
        TF_DeleteSession(Session, Status);
        TF_DeleteSessionOptions(SessionOpts);
        TF_DeleteStatus(Status);
    }

    TFModelLoader(const char* saved_model_dir, char const *feat_name){
        //********* Read model
        Graph = TF_NewGraph();
        Status = TF_NewStatus();
        SessionOpts = TF_NewSessionOptions();
        TF_Buffer* RunOpts = NULL;

        //const char* saved_model_dir = argv[1];
        const char* tags = "serve";

        int ntags = 1;
        Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);

        if(TF_GetCode(Status) == TF_OK)
            printf("TF_LoadSessionFromSavedModel OK\n");
        else
            printf("%s",TF_Message(Status));

        //****** Get input tensor
        Input = (TF_Output*)malloc(sizeof(TF_Output) * NumInputs );
        TF_Output t0 = {TF_GraphOperationByName(Graph, feat_name), 0};

        if(t0.oper == NULL)
            printf("ERROR: Failed TF_GraphOperationByName serving_default_\n");
        else
            printf("TF_GraphOperationByName serving_default_ is OK\n");

        Input[0] = t0;

        //********* Get Output tensor
        Output = (TF_Output*)malloc(sizeof(TF_Output) * NumOutputs);
        TF_Output t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};

        if(t2.oper == NULL)
            printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
        else
            printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");

        Output[0] = t2;

        //********* Allocate data for inputs & outputs
        InputValues  = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumInputs);
        OutputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumOutputs);

    }

    template<class T>
    float* run(T* data, int ndata, int ndims, int64_t* dims){
        TF_Tensor* int_tensor;
        if(std::is_same<T, float>::value){
            int_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, data, ndata, &NoOpDeallocator, 0);
        } else {
            int_tensor = TF_NewTensor(TF_UINT8, dims, ndims, data, ndata, &NoOpDeallocator, 0);
        }

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

        void* buff = TF_TensorData(OutputValues[0]);
        float* offsets = (float*)buff;
        return offsets;
    }
};

int main(int argc, char** argv)
{
    if(argc < 3){
        std::cout << "usage: ./metrabs <model-dir> <img-url>" << std::endl;
        return 0;
    }
    cv::Mat image = cv::imread(argv[2]);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_time = start_time;
    auto now = std::chrono::high_resolution_clock::now();
    float durr = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

    int inputBindIndex = 0;
    int outputBindIndex = 1;
    std::vector<float> fvec;
    
    cv::Mat image_full = cv::imread("image.png");

    // ===
    // yolo detection pre-built
    // ===
	Config config_v4;
	config_v4.net_type = YOLOV4;
	config_v4.file_model_cfg = "/disk/apps/yolo-tensorrt/configs/yolov4.cfg";
	config_v4.file_model_weights = "/disk/apps/yolo-tensorrt/configs/yolov4.weights";
	config_v4.calibration_image_list_file_txt = "/disk/apps/yolo-tensorrt/configs/calibration_images.txt";
	config_v4.inference_precison = FP16;
	config_v4.detect_thresh = 0.5;

	std::unique_ptr<Detector> detector(new Detector());
	detector->init(config_v4);
	std::vector<BatchResult> batch_res;

    cv::Mat crop;

    for(int j=0; j<100; j++){
        std::vector<cv::Mat> batch_img;
        batch_img.push_back(image_full);
        //batch_img.push_back(image_full);

        detector->detect(batch_img, batch_res);
        now = std::chrono::high_resolution_clock::now();
        durr = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
        last_time = now;
        std::cout << durr << std::endl;
        for (const auto &r : batch_res[0]){
            if(r.id != 0)
                continue;
            std::cout << r.rect << std::endl;
            crop = image_full(r.rect);
        }
    }
    cv::resize(crop, crop, cv::Size(256, 256));

    // ===
    // detection tf
    // ===
    if(false){
        TFModelLoader tf_loader_det = TFModelLoader("../mods/det", "serving_default_image");

        float* res;
        for(int j=0; j<100; j++){
            int wd = image_full.cols;
            int ht = image_full.rows;
            int64_t dims_det[] = {ht, wd, 3};

            res = tf_loader_det.run(image_full.data, wd*ht*3, 3, dims_det);
            now = std::chrono::high_resolution_clock::now();
            durr = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
            last_time = now;
            std::cout << durr << std::endl;
        }

        for(int i=0; i < 5; i++){
            std::cout << res[i] << " ";
        }
        std::cout << std::endl;
    }

    // ===
    // effnet backbone
    // ===
    Trt* onnx_net = new Trt();
    onnx_net->CreateEngine("../mods/effnet.onnx", "../mods/effnet.plan", 1, 0);
    //int inputBindIndex = 0;
    //int outputBindIndex = 1;

    cv::Mat img_f32;
    crop.convertTo(img_f32, CV_32FC3);
    //cv::resize(image, image, cv::Size(256, 256));

    img_f32 = img_f32/256.f;
    std::vector<float> output(327680/4);

    //std::vector<float> fvec;
    for(int i=0; i<100;i++){
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

    // ===
    // metrabs heads
    // ===
    TFModelLoader tf_loader = TFModelLoader(argv[1], "serving_default_feature");

    float* offsets;
    for(int cnt=0; cnt<100; cnt++){

        int ndims = 4;
        int64_t dims[] = {1, 8, 8, 1280};
        //float* data = &output[0];
        int ndata = 8*8*1280*4;

        float data[ndata];
        for(int i=0;i<ndata;i++){
            //data[i] = .2f;
            data[i] = output[i];
        }

        // run inference
        offsets = tf_loader.run(data, ndata, ndims, dims);

        now = std::chrono::high_resolution_clock::now();
        durr = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
        last_time = now;
        std::cout << durr << std::endl;
    }

    Eigen::Map<Eigen::MatrixXf> res_mat(offsets, 2, 32);
    printf("Result Tensor :\n");
    std::cout << res_mat.transpose() << std::endl;
    return 0;
}
