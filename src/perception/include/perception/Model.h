#ifndef __MODEL_H__
#define __MODEL_H__

#include <vector>

#include <cuda_fp16.h>
#include "cuda_utils.h"
#include <cuda_runtime_api.h>

#include <NvInfer.h>
#include <NvInferRuntime.h>

using namespace nvinfer1;

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>

constexpr int W = 1536;
constexpr int H = 864;

constexpr char* NAME_INPUT = "";
constexpr char* NAME_REG = "";
constexpr char* NAME_OBJ = "";
constexpr char* NAME_CLS = "";

class Model {
public:
    Model();
    ~Model();

    cv::Mat preprocess(const cv::Mat& image);
    void inference(const cv::Mat& model_input);
    void postprocess();

    void publishBEVInfo(); // 판단에 필요한 데이터 발행
    void publishVizResult(); // 결과 시각화용 데이터 발행

private:
    void __decode_predictions();
    void __tiny_filter_on_dets();

private:
    // TensorRT 객체
    IRuntime* runtime_;
    ICudaEngine* engine_;
    IExecutionContext* context_;
    cudaStream_t stream_;

    // cuda 데이터
    std::vector<void*> input_buffers_, output_buffers_;
    std::vector<std::string> input_names_, output_names_;
    std::vector<int> input_sizes_, output_sizes_;
    std::vector<Dims> input_shapes_, output_shapes_;

    int input_width_, input_height_;
};

#endif
