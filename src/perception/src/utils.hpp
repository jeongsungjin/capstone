#include <iostream>
#include <opencv2/opencv.hpp>

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
