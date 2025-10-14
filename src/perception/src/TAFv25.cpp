#include "perception/TAFv25.h"

#include "perception/utils.hpp"

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
    return nchw;
}

xt::xarray<float> TAFv25::inference(xt::xarray<half>& model_input){
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
    
    cudaStreamSynchronize(stream);

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

xt::array<float> TAFv25::postprocess(xt::xarray<float> output){
    auto nms = non_max_suppression(output)[0];
    xt::array<float> ret = scale_triangles(xt::view(nms, xt::all(), xt::range(0, 6)), {832, 1440}, {ori_h, ori_w});

    return ret;
}

template <typename E>
xt::array<float> TAFv25::pixelToWorldPlane(const xt::xexpression<E>&, const xt::array<float>& H){
    auto hom = xt::concatenate(xt::xtuple(x, xt::ones({x.shape()[0], 1})), 1);
    auto res = xt::linalg::dot(hom, xt::transpose(H));
    res /= xt::view(res, xt::all(), xt::keep(2));
    return xt::eval(xt::view(res, xt::all(), xt::range(0, 1)));
}

xt::array<float> TAFv25::completeParallelogramse(xt::array<float>& corners1, xt::array<float>& corners2, xt::array<float>& centers, bool include_centers){
    auto corners3 = centers * 2 - corners1;
    auto corners4 = centers * 2 - corners2;

    auto dir = corners1 - corners3;
    auto rads = xt::atan2(
        xt::view(dir, xt::all(), xt::keep(1)),
        xt::view(dir, xt::all(), xt::keep(0))
    );

    auto widths = xt::linalg::norm(corners2 - corners1, 2, 1);
    auto heights = xt::linalg::norm(corners3 - corners1, 2, 1);

    xt::array<float> ret = xt::concatenate(
        xt::xtuple(
            centers,
            widths,
            heights,
            rads
        ), 1
    );

    return xt::eval(ret);
}

void TAFv25::visualizeDetections(cv::Mat& image, const xt::xarray<float>& detections){
    for (std::size_t i = 0; i < detections.shape()[0]; ++i){
        auto det = xt::view(detections, i, xt::range(0, 6));

        float x1 = det(0);
        float y1 = det(1);
        float x2 = det(2);
        float y2 = det(3);
        float x3 = det(4);
        float y3 = det(5);

        float x2_mirror = 2.0f * x1 - x2;
        float y2_mirror = 2.0f * y1 - y2;
        float x3_mirror = 2.0f * x1 - x3;
        float y3_mirror = 2.0f * y1 - y3;

        std::vector<cv::Point> triangle_points = {
            cv::Point(static_cast<int>(x2), static_cast<int>(y2)),
            cv::Point(static_cast<int>(x3), static_cast<int>(y3)),
            cv::Point(static_cast<int>(x2_mirror), static_cast<int>(y2_mirror)),
            cv::Point(static_cast<int>(x3_mirror), static_cast<int>(y3_mirror))
        };

        const cv::Point* pts[1] = { triangle_points.data() };
        int npts[] = { static_cast<int>(triangle_points.size()) };

        cv::polylines(image, pts, npts, 1, true, cv::Scalar(0, 255, 0), 2);
    }
}

xt::array<float> TAFv25::toBEV(xt::array<float>& model_output){
    auto v_centers_p = xt::view(model_output, xt::all(), xt::range(0, 1));
    auto v_front_1_p = xt::view(model_output, xt::all(), xt::range(2, 3));
    auto v_front_2_p = xt::view(model_output, xt::all(), xt::range(4, 5));

    auto r_centers_w = pixel_to_world_plane(v_centers_p);
    auto r_front_1_w = pixel_to_world_plane(v_front_1_p);
    auto r_front_2_w = pixel_to_world_plane(v_front_2_p);

    xt::array<float> oriented_bbox = complete_parallelograms(r_front_1_w, r_front_2_w, r_centers_w, true);

    return oriented_bbox;
}
