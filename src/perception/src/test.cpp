#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_fp16.h>

#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xcontainer.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xindex_view.hpp>

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

xt::xarray<half> cvMatToXTensor(const cv::Mat& img, const float scale_factor=1.0){
    CV_Assert(img.depth() == CV_8U);
    int rows = img.rows;
    int cols = img.cols;
    int channels = img.channels();

    std::vector<size_t> shape = {static_cast<size_t>(rows),
                                 static_cast<size_t>(cols),
                                 static_cast<size_t>(channels)};

    CV_Assert(img.isContinuous());

    std::vector<half> data(rows * cols * channels);
    const unsigned char* src = img.ptr<unsigned char>();
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = __float2half(static_cast<float>(src[i]) * scale_factor);
    }

    return xt::adapt(std::move(data), shape);
}

xt::xarray<float> compute_bounding_boxes(const xt::xarray<float>& triangles){
    auto x1 = xt::col(triangles, 0);
    auto y1 = xt::col(triangles, 1);
    auto x2 = xt::col(triangles, 2);
    auto y2 = xt::col(triangles, 3);
    auto x3 = xt::col(triangles, 4);
    auto y3 = xt::col(triangles, 5);

    auto x2_mirror = 2.0f * x1 - x2;
    auto y2_mirror = 2.0f * y1 - y2;
    auto x3_mirror = 2.0f * x1 - x3;
    auto y3_mirror = 2.0f * y1 - y3;

    auto x_coords = xt::stack(xt::xtuple(x1, x2, x3, x2_mirror, x3_mirror), 1);  // (N, 5)
    auto y_coords = xt::stack(xt::xtuple(y1, y2, y3, y2_mirror, y3_mirror), 1);  // (N, 5)

    auto min_x = xt::amin(x_coords, {1});
    auto max_x = xt::amax(x_coords, {1});
    auto min_y = xt::amin(y_coords, {1});
    auto max_y = xt::amax(y_coords, {1});

    auto bbox = xt::stack(xt::xtuple(min_x, min_y, max_x, max_y), 1);

    return bbox;
}

std::vector<int> nms_bboxes(const xt::xarray<float>& triangles,
                            const xt::xarray<float>& confidences,
                            float iou_threshold)
{
    // 1️⃣ Bounding boxes
    auto bboxes = compute_bounding_boxes(triangles);
    auto x1 = xt::col(bboxes, 0);
    auto y1 = xt::col(bboxes, 1);
    auto x2 = xt::col(bboxes, 2);
    auto y2 = xt::col(bboxes, 3);

    // 2️⃣ Areas
    auto areas = (x2 - x1 + 1.0f) * (y2 - y1 + 1.0f);

    // 3️⃣ Sort descending by confidence
    auto order = xt::argsort(confidences);
    std::reverse(order.begin(), order.end());

    std::vector<int> keep;
    while (order.size() > 0)
    {
        int i = order(0);
        keep.push_back(i);

        if (order.size() == 1)
            break;

        auto rest = xt::view(order, xt::range(1, xt::placeholders::_));

        // broadcast scalar → vector shape
        auto xx1 = xt::maximum(x1(i), xt::index_view(x1, rest));
        auto yy1 = xt::maximum(y1(i), xt::index_view(y1, rest));
        auto xx2 = xt::minimum(x2(i), xt::index_view(x2, rest));
        auto yy2 = xt::minimum(y2(i), xt::index_view(y2, rest));

        auto w = xt::maximum(0.0f, xx2 - xx1 + 1.0f);
        auto h = xt::maximum(0.0f, yy2 - yy1 + 1.0f);
        auto inter = w * h;

        auto iou = inter / (areas(i) + xt::index_view(areas, rest) - inter);

        // filter where iou <= threshold
        auto mask = xt::flatten_indices(xt::where(iou <= iou_threshold));

        std::vector<int> new_order;
        new_order.reserve(mask.size());
        for (std::size_t j = 0; j < mask.size(); ++j)
        {
            new_order.push_back(rest(mask[j]));
        }
        order = xt::adapt(new_order);
    }

    return keep;
}

std::vector<xt::xarray<float>> non_max_suppression(
    const xt::xarray<float>& prediction,
    float conf_thres = 0.1f,
    float iou_thres = 0.5f,
    bool agnostic = false,
    bool multi_label = false,
    int max_det = 300,
    int nc = 0,
    int max_nms = 30000,
    int max_wh = 7680)
{
    // Validate thresholds
    if (conf_thres < 0.0f || conf_thres > 1.0f)
        throw std::runtime_error("Invalid conf_thres");
    if (iou_thres < 0.0f || iou_thres > 1.0f)
        throw std::runtime_error("Invalid iou_thres");

    // prediction: shape (1, 84, 6300)
    auto shape = prediction.shape();
    int bs = shape[0];  // batch size
    int ch = shape[1];
    int nx = shape[2];

    nc = (nc > 0) ? nc : (ch - 6);
    int nm = ch - nc - 6;
    int mi = 6 + nc;

    // candidates
    auto cls_part = xt::view(prediction, xt::all(), xt::range(6, mi), xt::all());
    auto max_cls_conf = xt::amax(cls_part, {1});
    auto xc = xt::greater(max_cls_conf, conf_thres); // shape (bs, nx)

    // transpose to (bs, nx, ch)
    auto pred_t = xt::transpose(prediction, {0, 2, 1});
    std::vector<xt::xarray<float>> output(bs);

    for (int xi = 0; xi < bs; ++xi)
    {
        auto x = xt::view(pred_t, xi, xt::all(), xt::all());  // (nx, ch)
        auto mask = xt::view(xc, xi, xt::all());              // (nx)
        // filter by confidence
        auto indices = xt::flatten_indices(xt::where(mask));
        x = xt::view(x, indices, xt::all());

        if (x.shape()[0] == 0)
        {
            output[xi] = xt::zeros<float>({0, 8 + nm});
            continue;
        }

        // Split (box, cls, mask)
        auto box = xt::view(x, xt::all(), xt::range(0, 6));
        auto cls = xt::view(x, xt::all(), xt::range(6, 6 + nc));
        auto mask_part = xt::view(x, xt::all(), xt::range(6 + nc, 6 + nc + nm));

        if (multi_label && nc > 1)
        {
            // multi-label version
            auto ij = xt::where(cls > conf_thres);

            std::vector<std::size_t> i_idx;
            std::vector<std::size_t> j_idx;
            i_idx.reserve(ij.size());
            j_idx.reserve(ij.size());

            // extract row/col indices from where result
            for (auto& pair : ij)
            {
                i_idx.push_back(pair[0]);
                j_idx.push_back(pair[1]);
            }

            if (i_idx.empty())
            {
                output[xi] = xt::zeros<float>({0, 8 + nm});
                continue;
            }

            auto selected_box  = xt::index_view(box, i_idx);
            xt::xarray<float> selected_conf = xt::zeros<float>({i_idx.size()});

            for (std::size_t k = 0; k < i_idx.size(); ++k)
                selected_conf(k) = cls(i_idx[k], j_idx[k]);

            auto selected_cls  = xt::cast<float>(xt::adapt(j_idx));
            auto selected_mask = xt::index_view(mask_part, i_idx);

            // reshape and concatenate all parts
            std::array<std::size_t, 2> conf_shape = {selected_conf.shape()[0], 1};
            std::array<std::size_t, 2> cls_shape  = {selected_cls.shape()[0], 1};

            x = xt::concatenate(
                    xt::xtuple(
                        selected_box,
                        xt::reshape_view(selected_conf, conf_shape),
                        xt::reshape_view(selected_cls,  cls_shape),
                        selected_mask),
                    1);
        }

        else
        {
            // best class only
            auto conf = xt::amax(cls, {1}, true);
            auto j = xt::argmax(cls, 1);
            auto j_f = xt::cast<float>(xt::expand_dims(j, 1));
            auto mask_filter = xt::flatten_indices(xt::where(xt::squeeze(conf) > conf_thres));
            if (mask_filter.size() == 0)
            {
                output[xi] = xt::zeros<float>({0, 8 + nm});
                continue;
            }

            x = xt::concatenate(
                xt::xtuple(
                    box, conf, j_f, mask_part),
                1);
            x = xt::index_view(x, mask_filter);
        }

        int n = x.shape()[0];
        if (n == 0)
        {
            output[xi] = xt::zeros<float>({0, 8 + nm});
            continue;
        }

        if (n > max_nms)
        {
            auto conf_col = xt::col(x, 6);
            auto order = xt::argsort(conf_col);
            std::reverse(order.begin(), order.end());
            order = xt::view(order, xt::range(0, max_nms));
            x = xt::index_view(x, order);
        }

        // class offset
        auto c = xt::col(x, 7);
        if (!agnostic)
            c = c * max_wh;

        auto boxes = xt::view(x, xt::all(), xt::range(0, 6)) + xt::expand_dims(c, 1);
        auto scores = xt::col(x, 6);

        auto keep_idx = nms_bboxes(boxes, scores, iou_thres);
        if ((int)keep_idx.size() > max_det)
            keep_idx.resize(max_det);

        xt::xarray<float> selected = xt::zeros<float>({(int)keep_idx.size(), (int)x.shape()[1]});
        for (std::size_t k = 0; k < keep_idx.size(); ++k)
        {
            xt::view(selected, k, xt::all()) = xt::view(x, keep_idx[k], xt::all());
        }

        output[xi] = selected;
    }

    return output;
}

xt::xarray<float> scale_triangles(xt::xarray<float> triangles,
                                  const std::array<int, 2>& src_shape,
                                  const std::array<int, 2>& target_shape,
                                  bool padding = true)
{
    // target_shape: (height, width)
    float gain = std::min(
        static_cast<float>(src_shape[0]) / static_cast<float>(target_shape[0]),
        static_cast<float>(src_shape[1]) / static_cast<float>(target_shape[1])
    );

    // wh padding
    float pad_w = std::round((src_shape[1] - target_shape[1] * gain) / 2.0f - 0.1f);
    float pad_h = std::round((src_shape[0] - target_shape[0] * gain) / 2.0f - 0.1f);

    if (padding)
    {
        // subtract padding (same logic as Python's triangles[..., i] -= pad[j])
        xt::view(triangles, xt::all(), 0) -= pad_w; // center_x
        xt::view(triangles, xt::all(), 1) -= pad_h; // center_y
        xt::view(triangles, xt::all(), 2) -= pad_w; // v1_x
        xt::view(triangles, xt::all(), 3) -= pad_h; // v1_y
        xt::view(triangles, xt::all(), 4) -= pad_w; // v2_x
        xt::view(triangles, xt::all(), 5) -= pad_h; // v2_y
    }

    // scale
    xt::view(triangles, xt::all(), xt::range(0, 6)) /= gain;

    return triangles;
}

void visualize_detections(const xt::xarray<float>& detections,
                          cv::Mat& image,
                          const std::string& save_path = "")
{
    for (std::size_t i = 0; i < detections.shape()[0]; ++i)
    {
        auto det = xt::view(detections, i, xt::range(0, 6));

        float x1 = det(0);
        float y1 = det(1);
        float x2 = det(2);
        float y2 = det(3);
        float x3 = det(4);
        float y3 = det(5);

        // mirrored points
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

    if (!save_path.empty())
        cv::imwrite(save_path, image);
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

    xt::xarray<half> nchw = xt::reshape_view(
        cvMatToXTensor(letter_box, 1.0 / 255.0),
        {1, channels, rows, cols}
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

    void* buffers[2];

    CUDA_CHECK(cudaMalloc(&buffers[input_idx], IO_SIZE[input_idx]));
    CUDA_CHECK(cudaMalloc(&buffers[output_idx], IO_SIZE[output_idx]));
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers[input_idx], ptr, IO_SIZE[input_idx], cudaMemcpyHostToDevice, stream));
    
    context->enqueueV2(buffers, stream, nullptr);
    
    uint8_t prob[IO_SIZE[output_idx]];
    CUDA_CHECK(cudaMemcpyAsync(prob, buffers[1], IO_SIZE[output_idx], cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    half* _output = reinterpret_cast<half *>(prob);

    auto shape = context->getTensorShape("y");
    xt::xarray<half> output = xt::adapt(std::move(_output), shape.d);
    xt::xarray<float> dst = xt::empty<float>(output.shape());
    for (std::size_t i = 0; i < output.size(); i++)
        dst.flat(i) = __half2float(output.flat(i));

    auto nms = non_max_suppression(dst)[0];
    auto tri = scale_triangles(xt::view(nms, xt::all(), xt::range(0, 6)), {832, 1440}, {ori_h, ori_w});
    visualize_detections(
        tri,
        img,
        "/home/guest5/capstone/src/perception/output/result.png"
    );

    cudaStreamDestroy(stream);

    CUDA_CHECK(cudaFree(buffers[input_idx]));
    CUDA_CHECK(cudaFree(buffers[output_idx]));

    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
