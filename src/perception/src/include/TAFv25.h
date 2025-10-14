#include <vector>
#include <iostream>

#include <xtensor/xarray.hpp>

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

    xt::xarray<half> preprocess();
    xt::xarray<float> inference();
    xt::xarray<float> postprocess();

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
};