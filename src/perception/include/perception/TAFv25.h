#ifndef __TAFv25_H__
#define __TAFv25_H__

#include <vector>
#include <iostream>

#include <xtensor/xarray.hpp>

#include <cuda_fp16.h>
#include "cuda_utils.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <opencv2/opencv.hpp>

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
    xt::xarray<float> toBEV(xt::xarray<float>& model_output);
    void visualizeDetections(cv::Mat& image, const xt::xarray<float>& detections);

private:
    const int ORIGINAL_WIDTH;
    const int ORIGINAL_HEIGHT;

    // TensorRT 객체
    IRuntime* runtime_;
    ICudaEngine* engine_;
    IExecutionContext* context_;
    cudaStream_t stream_;

    int io_size_[2] = { sizeof(half), sizeof(half) };
    
    void* buffers_[2];

    // ouput layer shape
    Dims output_shape_;

private:
    xt::xarray<float> completeParallelograms(xt::xarray<float>& corners1, xt::xarray<float>& corners2, xt::xarray<float>& centers, bool include_centers);

    template <typename E>
    xt::xarray<float> pixelToWorldPlane(const xt::xexpression<E>&, const xt::xarray<float>& H);
};

#endif
