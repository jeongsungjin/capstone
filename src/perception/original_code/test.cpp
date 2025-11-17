#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_fp16.h>

#include <cassert>

#include <fstream>
#include <iostream>
#include <vector>

#include <string>

#include "utils.hpp"

#include "cuda_utils.h"

#include <opencv2/opencv.hpp>

using namespace nvinfer1;

int main(){
    std::vector<char> engineData = readPlanFile("/home/ctrl/capstone/src/perception/engine/model_cm89.plan");

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine) { std::cerr << "Failed to deserialize engine\n"; return -1; }
    
    IExecutionContext* context = engine->createExecutionContext();
    if (!context) { std::cerr << "Failed to create context\n"; return -1; }

    int32_t input_idx = 0;
    if (!engine->bindingIsInput(input_idx)) { 
        std::cerr << "input idx is not correct\n"; 
        return -1; 
    }

    int32_t output_idx = 1;
    if (engine->bindingIsInput(output_idx)) { 
        std::cerr << "output idx is not correct\n"; 
        return -1; 
    }
    
    std::string path = "/home/ctrl/capstone/src/perception/samples/k729_cam1_1730382931-498000000.jpg";
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    int ori_w = img.cols, ori_h = img.rows;

    cv::Mat letter_box = LetterBox()(img);
    int channels = letter_box.channels(), rows = letter_box.rows, cols = letter_box.cols;
    cv::cvtColor(letter_box, letter_box, cv::COLOR_BGR2RGB);

    auto hwc = cvMatToXTensor(letter_box, 1.0 / 255.0);
    auto nhwc = xt::reshape_view(hwc, {1, rows, cols, channels});
    
    xt::xarray<half> nchw = xt::transpose(nhwc, {0, 3, 1, 2});

    // // fp16 is 2byte
    int IO_SIZE[] = { sizeof(half), sizeof(half) };
    for(int i = 0; i < 2; i++){
        auto bshape = engine->getBindingDimensions(i);
        for(int j = 0; j < bshape.nbDims; j++){
            IO_SIZE[i] *= bshape.d[j];
        }
    }

    void* buffers[2];

    CUDA_CHECK(cudaMalloc(&buffers[input_idx], IO_SIZE[input_idx]));
    CUDA_CHECK(cudaMalloc(&buffers[output_idx], IO_SIZE[output_idx]));
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers[input_idx], nchw.data(), IO_SIZE[input_idx], cudaMemcpyHostToDevice, stream));
    
    context->enqueueV2(buffers, stream, nullptr);
    
    half prob[IO_SIZE[output_idx] / sizeof(half)];
    CUDA_CHECK(cudaMemcpyAsync(prob, buffers[1], IO_SIZE[output_idx], cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    
    auto shape = context->getTensorShape("y");
    assert(shape.nbDims == 3);

    // 아마 batch 단위 inference로 바꾸면 에러가 날 것임!!
    assert(shape.d[0] == 1);
    assert(shape.d[1] == 7);
    assert(shape.d[2] == 24570);

    std::vector<float> _output(1 * 7 * 24570);
    for(int i = 0; i < _output.size(); i++){
        _output[i] =__half2float(prob[i]);
    }

    xt::xarray<float> output = xt::adapt(std::move(_output), {1, 7, 24570});
    auto nms = non_max_suppression(output)[0];
    auto tri = scale_triangles(xt::view(nms, xt::all(), xt::range(0, 6)), {832, 1440}, {ori_h, ori_w});
    visualize_detections(
        tri,
        img,
        "/home/ctrl/capstone/src/perception/output/result.png"
    );

    cudaStreamDestroy(stream);

    CUDA_CHECK(cudaFree(buffers[input_idx]));
    CUDA_CHECK(cudaFree(buffers[output_idx]));

    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
