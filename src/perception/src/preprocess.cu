#include "perception/preprocess.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void HWC_BGR_to_CHW_RGB_Normalize(
    const uchar3* input,
    float* output,
    int height,
    int width,
    float scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int hwc_idx = y * width + x;
    uchar3 bgr_pixel = input[hwc_idx]; // BGR 픽셀 로드

    int channel_size = height * width;
    int chw_idx_r = 0 * channel_size + y * width + x;
    int chw_idx_g = 1 * channel_size + y * width + x;
    int chw_idx_b = 2 * channel_size + y * width + x;

    output[chw_idx_r] = (static_cast<float>(bgr_pixel.z) * scale); // R
    output[chw_idx_g] = (static_cast<float>(bgr_pixel.y) * scale); // G
    output[chw_idx_b] = (static_cast<float>(bgr_pixel.x) * scale); // B
}

extern "C" 
void launchPreprocessKernel(
    const cv::cuda::GpuMat& input, 
    float* output, 
    int width, 
    int height,
    float scale, 
    cudaStream_t stream)
{
    if (input.type() != CV_8UC3 || !input.isContinuous()) {
        throw std::runtime_error("Input GpuMat must be of type CV_8UC3 and continuous."); 
    }

    // CUDA 커널 실행을 위한 그리드/블록 크기 설정
    dim3 block(16, 16); // 32x32 = 1024 스레드
    dim3 grid;
    grid.x = (width + block.x - 1) / block.x;
    grid.y = (height + block.y - 1) / block.y;

    // GpuMat에서 raw 포인터 가져오기
    const uchar3* d_input_ptr = reinterpret_cast<const uchar3*>(input.ptr<unsigned char>());

    // 커널 실행
    HWC_BGR_to_CHW_RGB_Normalize<<<grid, block, 0, stream>>>(
        d_input_ptr, 
        output, 
        height, 
        width,
        scale
    );
}

// Batched kernel: input is stacked as (batch * height) x width (CV_8UC3)
__global__ void HWC_BGR_to_CHW_RGB_Normalize_Batched(
    const uchar3* input,
    float* output,
    int height,
    int width,
    int batch,
    float scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (x >= width || y >= height || b >= batch) {
        return;
    }

    size_t in_h = static_cast<size_t>(height);
    size_t in_w = static_cast<size_t>(width);
    size_t hw = in_h * in_w;

    size_t in_idx = (static_cast<size_t>(b) * in_h + static_cast<size_t>(y)) * in_w + static_cast<size_t>(x);
    uchar3 bgr_pixel = input[in_idx];

    // Output is per-batch CHW contiguous
    size_t out_per_image = 3 * hw;
    size_t base = static_cast<size_t>(b) * out_per_image;
    size_t pos = static_cast<size_t>(y) * in_w + static_cast<size_t>(x);

    output[base + 0 * hw + pos] = static_cast<float>(bgr_pixel.z) * scale; // R
    output[base + 1 * hw + pos] = static_cast<float>(bgr_pixel.y) * scale; // G
    output[base + 2 * hw + pos] = static_cast<float>(bgr_pixel.x) * scale; // B
}

extern "C"
void launchPreprocessKernelBatched(
    const cv::cuda::GpuMat& stackedInput,
    float* output,
    int width,
    int height,
    int batch,
    float scale,
    cudaStream_t stream)
{
    if (stackedInput.type() != CV_8UC3 || !stackedInput.isContinuous()) {
        throw std::runtime_error("Stacked GpuMat must be CV_8UC3 and continuous.");
    }
    if (stackedInput.rows != height * batch || stackedInput.cols != width) {
        throw std::runtime_error("Stacked GpuMat shape mismatch: expected (height*batch, width, 3).");
    }

    dim3 block(16, 16, 1);
    dim3 grid;
    grid.x = (width + block.x - 1) / block.x;
    grid.y = (height + block.y - 1) / block.y;
    grid.z = batch;

    const uchar3* d_input_ptr = reinterpret_cast<const uchar3*>(stackedInput.ptr<unsigned char>());

    HWC_BGR_to_CHW_RGB_Normalize_Batched<<<grid, block, 0, stream>>>(
        d_input_ptr,
        output,
        height,
        width,
        batch,
        scale
    );
}