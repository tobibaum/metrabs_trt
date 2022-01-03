#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include "tensorflow/c/c_api.h"
#include <Eigen/Geometry>

#include "Trt.h"
#include "class_detector.h"

void NoOpDeallocator(void* data, size_t a, void* b) {}

/*
 * https://github.com/AmirulOm/tensorflow_capi_sample/blob/master/main.c
 */
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
    int ndata;
    int ndims;
    int64_t* dims;

public:
    ~TFModelLoader(){
        // Free memory
        TF_DeleteGraph(Graph);
        TF_DeleteSession(Session, Status);
        TF_DeleteSessionOptions(SessionOpts);
        TF_DeleteStatus(Status);
    }

    TFModelLoader(std::string base_dir, char const *feat_name, int _ndata, int _ndims, int64_t* _dims){
        dims = _dims;
        ndims = _ndims;
        ndata = _ndata;
        //********* Read model
        Graph = TF_NewGraph();
        Status = TF_NewStatus();
        SessionOpts = TF_NewSessionOptions();
        TF_Buffer* RunOpts = NULL;

        auto temp = base_dir + "/metrab_head";
        const char* saved_model_dir = temp.c_str();
        std::cout << "===metrab head loaded:=== " << saved_model_dir << std::endl;
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

    template<class T> float* run(T* data){
        TF_Tensor* int_tensor;
        if(std::is_same<T, float>::value){
            int_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, data, ndata, &NoOpDeallocator, 0);
        } else {
            int_tensor = TF_NewTensor(TF_UINT8, dims, ndims, data, ndata, &NoOpDeallocator, 0);
        }

        if (int_tensor == NULL)
            printf("ERROR: Failed TF_NewTensor\n");

        InputValues[0] = int_tensor;

        // Run the Session
        TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0, NULL , Status);

        if(TF_GetCode(Status) != TF_OK)
            printf("%s",TF_Message(Status));

        void* buff = TF_TensorData(OutputValues[0]);
        float* offsets = (float*)buff;
        return offsets;
    }
};

/*
 * https://github.com/enazoe/yolo-tensorrt.git
 */
class YoloDetector {
private:
    std::unique_ptr<Detector> detector;

public:
    YoloDetector(std::string base_dir){
        Config config_v4;
        config_v4.net_type = YOLOV4;
        config_v4.file_model_cfg = base_dir + "/configs/yolov4.cfg";
        config_v4.file_model_weights = base_dir + "/configs/yolov4.weights";
        config_v4.inference_precison = FP16;
        config_v4.detect_thresh = 0.5;

        detector = std::unique_ptr<Detector>(new Detector());
        detector->init(config_v4);
    }

    cv::Mat detect(cv::Mat image, bool &result){
        std::vector<BatchResult> batch_res;
        std::vector<cv::Mat> batch_img;
        batch_img.push_back(image);

        // inference (muted!)
        std::cout.setstate(std::ios_base::failbit);
        detector->detect(batch_img, batch_res);
        std::cout.clear();

        cv::Mat crop;
        result = false;

        for (const auto &r : batch_res[0]){
            if(r.id != 0)
                continue;
            // TODO: handle multiple detections!
            crop = image(r.rect);
            result = true;
        }
        return crop;
    }
};

/*
 * https://github.com/zerollzeng/tiny-tensorrt.git
 */
class EffnetBBone {
private:
    Trt* onnx_net;
    int inputBindIndex = 0;
    int outputBindIndex = 1;

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

public:
    EffnetBBone(std::string base_dir){
        onnx_net = new Trt();
        onnx_net->CreateEngine(base_dir + "/bbone.onnx", base_dir + "/bbone.plan", 1, 0);
        onnx_net->SetLogLevel((int)Severity::kINTERNAL_ERROR);
        //onnx_net->SetWorkpaceSize(2*1024*1024);
    }

    std::vector<float> run(cv::Mat crop){
        cv::Mat img_f32;
        crop.convertTo(img_f32, CV_32FC3);
        img_f32 = img_f32/256.f;
        std::vector<float> fvec = convert_mat_to_fvec(img_f32);
        std::vector<float> output(327680/4);

        // inference
        onnx_net->CopyFromHostToDevice(fvec, inputBindIndex);
        onnx_net->Forward();
        onnx_net->CopyFromDeviceToHost(output, outputBindIndex);
        return output;
    }
};


int main(int argc, char** argv)
{
    if(argc < 3){
        std::cout << "usage: ./metrabs <model-base-dir> <video-file>" << std::endl;
        return 0;
    }
    std::string base_dir = argv[1];

    // timers
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_time = start_time;
    auto now = std::chrono::high_resolution_clock::now();
    float durr = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

    // === INIT ===
    YoloDetector yolo_det = YoloDetector(base_dir);

    EffnetBBone effnet = EffnetBBone(base_dir);

    int64_t dims[] = {1, 8, 8, 1280};
    //int64_t dims[] = {1, 8, 8, 2048};
    TFModelLoader tf_loader = TFModelLoader(base_dir, "serving_default_feature",
            8*8*1280*4, 4, dims);
            //8*8*2048*4, 4, dims);

    Eigen::MatrixXf res_mat;

    cv::VideoCapture cap(argv[2]);
    cv::Mat frame;
    for(;;){
        cap >> frame;
        if (frame.empty())
            break;

        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        // === yolo detection pre-built ===
        bool result;
        cv::Mat crop = yolo_det.detect(frame, result);
        if(!result)
            continue;
        cv::resize(crop, crop, cv::Size(256, 256));

        // === effnet backbone ===
        std::vector<float> output = effnet.run(crop);

        // === metrabs heads ===
        float* data = &output[0];
        float* offsets = tf_loader.run(data);
        res_mat = Eigen::Map<Eigen::MatrixXf>(offsets, 2, 32);

        // === timing ===
        now = std::chrono::high_resolution_clock::now();
        durr = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
        last_time = now;
        std::cout << durr << std::endl;

        // === viz ===
        cv::namedWindow("image", cv::WINDOW_NORMAL);
        cv::resizeWindow("image", 900, 500);

        cv::Mat draw_mat;
        cv::cvtColor(crop, draw_mat, cv::COLOR_RGB2BGR);
        // draw dots!
        for(int k=0; k<res_mat.cols(); k++){
            cv::Point pt(int(res_mat(0, k)), int(res_mat(1, k)));
            cv::circle(draw_mat, pt, 2, {255, 125, 0}, 2);
        }
        cv::imshow("image", draw_mat);

		int32_t key = cv::waitKey(1);
        if(key == 'q' || key == 27){
            break;
        }
    }

    std::cout << res_mat.transpose() << std::endl;
    return 0;
}
