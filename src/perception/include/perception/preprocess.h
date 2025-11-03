#pragma once

#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

extern "C"
void launchPreprocessKernel(
    const cv::cuda::GpuMat& input, 
    float* output, 
    int width, 
    int height,
    float scale, 
    cudaStream_t stream
);

// Batched version: input is a single stacked GpuMat of size (height*batch) x width (CV_8UC3).
// For each batch b, rows [b*height : (b+1)*height) contain image b.
// Output layout is CHW per image, contiguous per batch: [b][3][H][W].
extern "C"
void launchPreprocessKernelBatched(
    const cv::cuda::GpuMat& stackedInput,
    float* output,
    int width,
    int height,
    int batch,
    float scale,
    cudaStream_t stream
);