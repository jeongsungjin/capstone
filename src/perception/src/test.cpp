#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_fp16.h>

#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xio.hpp>

#include <cassert>

#include <fstream>
#include <iostream>
#include <vector>

#include <string>

#include "utils.hpp"

#include "cuda_utils.h"

constexpr char* INPUT_NAME = "x";
constexpr char* OUTPUT_NAME = "y";

#include <opencv2/opencv.hpp>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;

std::vector<char> readPlanFile(const std::string& filename){
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open plan file");

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

xt::xarray<float> cvMatToXTensor(const cv::Mat& img, const float scale_factor=1.0){
    CV_Assert(img.depth() == CV_8U);
    int rows = img.rows;
    int cols = img.cols;
    int channels = img.channels();

    std::vector<size_t> shape = {static_cast<size_t>(rows),
                                 static_cast<size_t>(cols),
                                 static_cast<size_t>(channels)};

    std::vector<size_t> strides = {static_cast<size_t>(cols * channels),
                                   static_cast<size_t>(channels),
                                   static_cast<size_t>(1)};

    CV_Assert(img.isContinuous());

    std::vector<float> data(rows * cols * channels);
    const unsigned char* src = img.ptr<unsigned char>();
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(src[i]) * scale_factor;
    }

    return xt::adapt(std::move(data), shape);
}

int main(){
    std::vector<char> engineData = readPlanFile("/home/guest5/capstone/src/perception/engine/model_v2_half.plan");

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
    
    std::string path = "/home/guest5/capstone/src/perception/samples/k729_cam1_1730382931-498000000.jpg";
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    int ori_w = img.cols, ori_h = img.rows;

    cv::Mat letter_box = LetterBox()(img);
    int channels = letter_box.channels(), rows = letter_box.rows, cols = letter_box.cols;
    cv::cvtColor(letter_box, letter_box, cv::COLOR_BGR2RGB);

    xt::xarray<float> tensor = cvMatToXTensor(letter_box, 1.0 / 255.0);
    xt::xarray<__half> nchw = xt::cast<__half>(
            xt::reshape_view(tensor, {1, channels, rows, cols})
        );

    uint8_t* ptr = reinterpret_cast<uint8_t*>(nchw.data());

    // fp16 is 2byte
    int IO_SIZE[] = { 2, 2 };
    for(int i = 0; i < 2; i++){
        auto bshape = engine->getBindingDimensions(i);
        for(int j = 0; j < bshape.nbDims; j++){
            IO_SIZE[i] *= bshape.d[j];
        }
    }

    std::cout << "IO_SIZE[0] : " << IO_SIZE[0] << '\n';

    void* buffers[2];

    CUDA_CHECK(cudaMalloc(&buffers[input_idx], IO_SIZE[input_idx]));
    CUDA_CHECK(cudaMalloc(&buffers[output_idx], IO_SIZE[output_idx]));
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers[input_idx], ptr, IO_SIZE[input_idx], cudaMemcpyHostToDevice, stream));
    
    context->enqueue(1, buffers, stream, nullptr);
    
    // float prob[IO_SIZE[output_idx]];
    // CUDA_CHECK(cudaMemcpyAsync(prob, buffers[1], IO_SIZE[output_idx], cudaMemcpyDeviceToHost, stream));
    
    // cudaStreamSynchronize(stream);

    // cudaStreamDestroy(stream);

    // CUDA_CHECK(cudaFree(buffers[input_idx]));
    // CUDA_CHECK(cudaFree(buffers[output_idx]));

    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
