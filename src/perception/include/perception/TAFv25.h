#ifndef __TAFv25_H__
#define __TAFv25_H__

#include <vector>
#include <iostream>

#include <xtensor/containers/xarray.hpp>

#include <cuda_fp16.h>
#include "cuda_utils.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>

using namespace nvinfer1;

constexpr char* INPUT_NAME = "x";
constexpr char* OUTPUT_NAME = "y";

constexpr int INPUT_INDEX = 0;
constexpr int OUTPUT_INDEX = 1;

class TAFv25 {
public:
    TAFv25(int original_width, int original_height);
    ~TAFv25();

    xt::xarray<half> preprocess(cv::Mat img);
    xt::xarray<float> inference(xt::xarray<half>& model_input);
    xt::xarray<float> postprocess(xt::xarray<float> model_output);
    xt::xarray<float> toBEV(xt::array<float>& model_output);
    void TAFv25::visualizeDetections(cv::Mat& image, const xt::xarray<float>& detections);

private:
    const int ORIGINAL_WIDTH;
    const int ORIGINAL_HEIGHT;

    // TensorRT 객체
    IRuntime* runtime_;
    ICudaEngine* engine_
    IExecutionContext* context_;
    cudaStream_t stream_;

    int io_size_[] = { sizeof(half), sizeof(half) };
    
    void* buffers_[2];

    // ouput layer shape
    Dims output_shape_;

private:
    xt::array<float> completeParallelogramse(xt::array<float>& corners1, xt::array<float>& corners2, xt::array<float>& centers, bool include_centers);

    template <typename E>
    xt::array<float> pixelToWorldPlane(const xt::xexpression<E>&, const xt::array<float>& H);
};

#endif
