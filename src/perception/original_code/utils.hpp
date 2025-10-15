#ifndef UTILS__HPP
#define UTILS__HPP

#include <iostream>
#include <opencv2/opencv.hpp>

#include <typeinfo>

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

constexpr int LETTERBOX_SIZE[2] = {832, 1440};

class LetterBox {
public:
    LetterBox(bool auto_=false, bool scale_fill=false, bool scale_up=true, bool center=true, int stride=32) {        
        auto_       = auto_;
        scale_fill_ = scale_fill;
        scale_up_   = scale_up;
        center_     = center; 
        stride_     = stride;
    }

    cv::Mat operator()(const cv::Mat& img){
        double img_w = 1. * img.cols, img_h = 1. * img.rows;

        double ratio = std::min(
            LETTERBOX_SIZE[0] / img_h,
            LETTERBOX_SIZE[1] / img_w
        );

        int new_unpad_h = static_cast<int>(std::round(img_h * ratio));
        int new_unpad_w = static_cast<int>(std::round(img_w * ratio));

        int dh = (LETTERBOX_SIZE[0] - new_unpad_h) / 2;
        int dw = (LETTERBOX_SIZE[1] - new_unpad_w) / 2;

        if(img_h != new_unpad_h && img_w != new_unpad_w){
            cv::resize(img, img, cv::Size(new_unpad_w, new_unpad_h), cv::INTER_LINEAR);
        }

        int top = static_cast<int>(std::round(dh - 0.1));
        int bottom = static_cast<int>(std::round(dh + 0.1));
        int left = static_cast<int>(std::round(dw - 0.1));
        int right = static_cast<int>(std::round(dw - 0.1));
        
        cv::Mat ret;
        cv::copyMakeBorder(img, ret, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        
        return ret;
    }

private:
    bool auto_, scale_fill_, scale_up_, center_, stride_;
};

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
    xt::xarray<float>& prediction,
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

    auto pred_tp = xt::transpose(prediction, {0, 2, 1});

    int bs = prediction.shape(0);   // batch size
    int ch = prediction.shape(1);   // channel size
    nc = (nc > 0) ? nc : (ch - 6);  // number of classes
    int nm = ch - nc - 6;           // number of masks
    int mi = 6 + nc;                // mask start index
        
    auto cls_part = xt::view(prediction, xt::all(), xt::range(6, mi), xt::all());    
    auto max_cls_conf = xt::amax(cls_part, {1});
    auto xc = xt::greater(max_cls_conf, conf_thres); // shape (bs, nx)

    // init shape == (1, 0, 8) on based python
    std::vector<xt::xarray<float>> output(bs);
    for (int xi = 0; xi < bs; xi++) {
        auto mask = xt::view(xc, xi, xt::all());
        auto flatten_indices = xt::flatten_indices(xt::where(mask));
        auto x = xt::eval(xt::view(pred_tp, xi, xt::keep(flatten_indices), xt::all()));

        if (x.shape()[0] == 0){
            std::cout << "x.shape()[0] == 0\n";
            output[xi] = xt::zeros<float>({0, 8 + nm});
            continue;
        }

        size_t box_size = 6;
        auto box = xt::view(x, xt::all(), xt::range(0, box_size));
        auto cls = xt::view(x, xt::all(), xt::range(box_size, box_size + nc));
        auto mask_part = xt::view(x, xt::all(), xt::range(box_size + nc, box_size + nc + nm));
    
        auto conf = xt::expand_dims(xt::amax(cls, {1}), 1);
        auto j_f = xt::cast<float>(xt::expand_dims(xt::argmax(cls, 1), 1));
        auto mask_filter = xt::flatten_indices(xt::where(xt::squeeze(conf) > conf_thres));

        if (mask_filter.size() == 0){
            std::cout << "mask_filter.size() == 0\n";
            output[xi] = xt::zeros<float>({0, 8 + nm});
            continue;
        }

        x = xt::concatenate(
            xt::xtuple(
                xt::eval(box), 
                xt::eval(conf), 
                xt::eval(j_f), 
                xt::eval(mask_part)
            ), 1
        );
        x = xt::view(x, xt::keep(mask_filter), xt::all());

        int n = x.shape()[0];
        if (n == 0) {
            std::cout << "n == 0\n";
            output[xi] = xt::zeros<float>({0, 8 + nm});
            continue;
        }

        if (n > max_nms){
            auto conf_col = xt::col(x, 6);
            auto order = xt::argsort(conf_col);
            std::reverse(order.begin(), order.end());
            order = xt::view(order, xt::range(0, max_nms));
            x = xt::index_view(x, order);
        }

        // class offset
        auto c = xt::col(x, 6);
        if (!agnostic)
            c = c * max_wh;

        auto boxes = xt::view(x, xt::all(), xt::range(0, 6)) + xt::expand_dims(c, 1);
        auto scores = xt::col(x, 6);

        auto keep_idx = nms_bboxes(boxes, scores, iou_thres);
        if ((int)keep_idx.size() > max_det)
            keep_idx.resize(max_det);

        xt::xarray<float> selected = xt::zeros<float>({(int)keep_idx.size(), (int)x.shape()[1]});
        for (std::size_t k = 0; k < keep_idx.size(); k++){
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

#endif
