#ifndef __MODEL_H__
#define __MODEL_H__

#include <map>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include "cuda_utils.h"
#include <cuda_runtime_api.h>

#include "nvidia_helper/buffers.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>

using namespace nvinfer1;

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <xtensor/xarray.hpp>

extern int W;
extern int H;

namespace layer_names {
    constexpr const char* INPUT = "images";
    constexpr const char* HIDDEN_IN = "h_in";
    constexpr const char* CELL_IN = "c_in";

    constexpr const char* HIDDEN_OUT = "h_out";
    constexpr const char* CELL_OUT = "c_out";

    constexpr const char* P0_REG = "p0_reg";
    constexpr const char* P0_OBJ = "p0_obj";
    constexpr const char* P0_CLS = "p0_cls";
    
    constexpr const char* P1_REG = "p1_reg";
    constexpr const char* P1_OBJ = "p1_obj";
    constexpr const char* P1_CLS = "p1_cls";
    
    constexpr const char* P2_REG = "p2_reg";
    constexpr const char* P2_OBJ = "p2_obj";
    constexpr const char* P2_CLS = "p2_cls";

    constexpr const int REG = 0, OBJ = 1, CLS = 2;
    const std::vector<std::vector<const char *>> name_iter = {
        { P0_REG, P0_OBJ, P0_CLS },
        { P1_REG, P1_OBJ, P1_CLS },
        { P2_REG, P2_OBJ, P2_CLS }
    };
};

struct DetectionInfo {
    std::vector<int> scores;
    std::vector<xt::xarray<float>> poly4s;
    std::vector<xt::xarray<float>> tri_ptss;
};

class Model {
public:
    Model(const std::string& pkg_path, const int batch_size);
    ~Model();

    int preprocess(const cv::Mat& image);
    void inference();
    void postprocess();

    const std::vector<DetectionInfo>& getDetections() {
        return detections_info_;
    }

private:
    void __copyLSTMOutputsToInputs();
    xt::xarray<float> __toXTensor(const char* tensor_name);
    void __decodePredictions(float conf_th=0.15, float nms_iou=0.2, int topk=50);
    void __tinyFilterOnDets();

private:
    // TensorRT 객체
    std::shared_ptr<IRuntime> runtime_;
    std::shared_ptr<ICudaEngine> engine_;
    std::unique_ptr<IExecutionContext, samplesCommon::InferDeleter> context_;
    cudaStream_t stream_;

    // model 관련 정보
    std::unique_ptr<samplesCommon::BufferManager> buffers_;

    std::vector<DetectionInfo> detections_info_;

    const std::vector<int> strides_ = {8, 16, 32};

    int input_width_, input_height_;
    int batch_size_;

    bool first_inference_;
};

#endif
