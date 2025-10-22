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
#include <xtensor/xadapt.hpp>

constexpr int W = 1536;
constexpr int H = 864;

namespace layer_names {
    constexpr char* INPUT = "images";
    constexpr char* HIDDEN_IN = "h_in";
    constexpr char* CELL_IN = "c_in";

    constexpr char* HIDDEN_OUT = "h_out";
    constexpr char* CELL_OUT = "c_out";

    constexpr char* P0_REG = "p0_reg";
    constexpr char* P0_OBJ = "p0_obj";
    constexpr char* P0_CLS = "p0_cls";
    
    constexpr char* P1_REG = "p1_reg";
    constexpr char* P1_OBJ = "p1_obj";
    constexpr char* P1_CLS = "p1_cls";
    
    constexpr char* P2_REG = "p2_reg";
    constexpr char* P2_OBJ = "p2_obj";
    constexpr char* P2_CLS = "p2_cls";
};

struct TensorInfo {
    int index;
    Dims shape;
    size_t size;
    void* ptr;
};

class Model {
public:
    Model(const std::string& pkg_path);
    ~Model();

    void preprocess(const cv::Mat& image);
    void inference();
    void postprocess();

    void publishBEVInfo(); // 판단에 필요한 데이터 발행
    void publishVizResult(); // 결과 시각화용 데이터 발행

private:
    void __decode_predictions();
    void __tiny_filter_on_dets();

private:
    // TensorRT 객체
    std::shared_ptr<IRuntime> runtime_;
    std::shared_ptr<ICudaEngine> engine_;
    std::unique_ptr<IExecutionContext, samplesCommon::InferDeleter> context_;
    cudaStream_t stream_;

    // model 관련 정보
    std::unique_ptr<samplesCommon::BufferManager> buffers_;

    const int strides_[3] = {8, 16, 32};

    int input_width_, input_height_;
};

#endif
