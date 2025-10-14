#include "TAFv25.h"

#include "utils.hpp"

TAFv25::TAFv25(int original_width, int original_height):
    ORIGINAL_WIDTH(original_width), ORIGINAL_HEIGHT(original_height)
{
    std::vector<char> engineData = readPlanFile("/home/ivsp/capstone/src/perception/engine/model_cm86.plan");

    runtime_ = createInferRuntime(gLogger);
    engine_ = runtime_->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine_) { 
        throw std::runtime_error("Failed to deserialize engine");
    }
    
    context_ = engine_->createExecutionContext();
    if (!context_) { 
        throw std::runtime_error("Failed to create context");
    }
    
    output_shape_ = context_->getTensorShape("y");
    if(output_shape_.nbDims != 3){
        throw std::runtime_error("output shape is not correct");
    }

    if(output_shape.d[0] == 1 
        && output_shape.d[1] == 7
        && output_shape.d[2] == 24570)
    {    
        throw std::runtime_error("output shape is not correct");
    }
    
    if (!engine_->bindingIsInput(INPUT_INDEX)) {
        throw std::runtime_error("input idx is not correct");
    }
    
    if (engine_->bindingIsInput(OUTPUT_INDEX)) {
        throw std::runtime_error("output idx is not correct");
    }
    
    for(int i = 0; i < 2; i++){
        auto bshape = engine_->getBindingDimensions(i);
        for(int j = 0; j < bshape.nbDims; j++){
            IO_SIZE_[i] *= bshape.d[j];
        }
    }

    CUDA_CHECK(cudaMalloc(&buffers_[INPUT_INDEX], IO_SIZE_[INPUT_INDEX]));
    CUDA_CHECK(cudaMalloc(&buffers_[OUTPUT_INDEX], IO_SIZE_[OUTPUT_INDEX]));

    CUDA_CHECK(cudaStreamCreate(&stream_));
}

TAFv25::~TAFv25(){
    cudaStreamDestroy(stream_);

    CUDA_CHECK(cudaFree(buffers[INPUT_INDEX]));
    CUDA_CHECK(cudaFree(buffers[OUTPUT_INDEX]));

    context_->destroy();
    engine_->destroy();
    runtime_->destroy();
}

xt::xarray<half> TAFv25::preprocess(cv::Mat img){
    cv::Mat letter_box = LetterBox()(img);
    int channels = letter_box.channels(), rows = letter_box.rows, cols = letter_box.cols;
    cv::cvtColor(letter_box, letter_box, cv::COLOR_BGR2RGB);

    auto hwc = cvMatToXTensor(letter_box, 1.0 / 255.0);
    auto nhwc = xt::reshape_view(hwc, {1, rows, cols, channels});

    xt::xarray<half> nchw = xt::transpose(nhwc, {0, 3, 1, 2});
}

xt::xarray<float> TAFv25::inference(){
    CUDA_CHECK(cudaMemcpyAsync(
        buffers_[INPUT_INDEX], 
        nchw.data(), 
        IO_SIZE[INPUT_INDEX], 
        cudaMemcpyHostToDevice, 
        stream
    ));

    context->enqueueV2(buffers, stream, nullptr);
    
    half prob[IO_SIZE[output_idx] / sizeof(half)];
    CUDA_CHECK(cudaMemcpyAsync(prob, buffers[1], IO_SIZE[output_idx], cudaMemcpyDeviceToHost, stream));
    
    cudaStreamSynchronize(stream); // GPU 비동기 동작을 대기하는 blocking 함수

    std::vector<float> _output(output_shape_.d[0] * output_shape_.d[1] * output_shape_.d[2]);
    for(int i = 0; i < _output.size(); i++){
        _output[i] =__half2float(prob[i]);
    }

    xt::xarray<float> output = xt::adapt(
        std::move(_output),
        {
            output_shape_.d[0], 
            output_shape_.d[1], 
            output_shape_.d[2]
        }
    );

    return output;
}

void pixel_to_world_plane();
void complete_parallelograms();

void TAFv25::postprocess(){
    auto nms = non_max_suppression(output)[0];
    auto results = scale_triangles(xt::view(nms, xt::all(), xt::range(0, 6)), {832, 1440}, {ori_h, ori_w});
    visualize_detections(
        results,
        img,
        "/home/ivsp/capstone/src/perception/output/result.png"
    );

    auto v_centers_p = xt::view(results, xt::all(), xt::range(0, 1));
    auto v_front_1_p = xt::view(results, xt::all(), xt::range(2, 3));
    auto v_front_2_p = xt::view(results, xt::all(), xt::range(4, 5));

    auto r_centers_w = pixel_to_world_plane();
    auto r_front_1_w = pixel_to_world_plane();
    auto r_front_2_w = pixel_to_world_plane();

    auto oriented_bbox = complete_parallelograms(r_front_1_w, r_front_2_w, r_centers_w);

    return oriented_bbox;
}


/*
def norm_homogeneous2(x: np.ndarray) -> np.ndarray:
    return x / x[:, 2:3] 

# (N, 2) -> (N, 3)
def homogeneous(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def pixel_to_world_plane(x: np.ndarray, H: np.ndarray, rescale: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    x = x.reshape(1, 2) if x.ndim == 1 else x
    x = rescale(x) if rescale is not None else x
    hom = homogeneous(x)
    res = norm_homogeneous2(hom @ H.T)
    return res[:, :2]
*/


/*
def mirror_points(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    assert points.shape == centers.shape
    return centers - (points - centers)

def get_heading(p1: np.ndarray, p2: np.ndarray, degrees: bool=False) -> np.ndarray:
    p1 = p1.reshape((1, 2)) if p1.ndim == 1 else p1
    p2 = p2.reshape((1, 2)) if p2.ndim == 1 else p2
    dir = p1 - p2
    radians = np.arctan2(dir[:,1], dir[:,0])
    return np.degrees(radians) if degrees else radianss

def complete_parallelograms(corners1: np.ndarray, corners2: np.ndarray, centers: np.ndarray, include_centers: bool = True) -> np.ndarray:
    corners3 = mirror_points(corners1, centers)
    corners4 = mirror_points(corners2, centers)
    corners = np.array([corners1, corners2, corners4, corners3])  # e.g. fl, fr, bl, br
    corners = np.vstack([corners, centers.reshape(1, -1, 2)]) if include_centers else corners
    return np.moveaxis(corners, 0, 1)  # (N, 4, 2)  or (N, 5, 2)
*/
