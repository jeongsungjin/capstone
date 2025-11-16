#pragma once
#include <cuda_runtime_api.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>
#include <NvInfer.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK

class Logger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        if (severity <= nvinfer1::ILogger::Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

extern Logger gLogger;

