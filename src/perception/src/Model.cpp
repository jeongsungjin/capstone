#include "perception/Model.h"

#include "timer.h"

#include <ros/ros.h>

#include "perception/utils.hpp"

#include <numeric>

#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xadapt.hpp>

int W = 1536;
int H = 864;

using namespace layer_names;

Model::Model(const std::string& pkg_path, const int batch_size): 
    first_inference_(true), batch_size_(batch_size), detections_info_(batch_size)
{
    std::vector<char> engineData = readPlanFile(
        pkg_path + "/engine/static_model_batch_" + std::to_string(batch_size) + ".plan"
    );

    runtime_ = std::shared_ptr<IRuntime>(
        createInferRuntime(gLogger),
        samplesCommon::InferDeleter()
    );
    if (!runtime_) throw std::runtime_error("Failed to create runtime");

    engine_ = std::shared_ptr<ICudaEngine>(
        runtime_->deserializeCudaEngine(engineData.data(), engineData.size()),
        samplesCommon::InferDeleter()
    );
    if (!engine_) throw std::runtime_error("Failed to deserialize engine");
    
    context_ = std::unique_ptr<IExecutionContext, samplesCommon::InferDeleter>(
        engine_->createExecutionContext()
    );
    if (!context_) throw std::runtime_error("Failed to create context");
    
    buffers_ = std::make_unique<samplesCommon::BufferManager>(engine_, /*batchSize=*/0, context_.get());
    CHECK(cudaStreamCreate(&stream_));
}

Model::~Model(){
    cudaStreamDestroy(stream_);
}

int Model::preprocess(const cv::Mat& img){
    Timer timer("Model::preprocess");

    input_width_ = img.cols;
    input_height_ = img.rows;

    std::vector<cv::Mat> imgs = {
        img
    };

    cv::Mat model_input = cv::dnn::blobFromImages(
        imgs,               // src img
        1. / 255.,          // scale factor
        cv::Size(W, H),     // resize size
        cv::Scalar(),       // mean
        true,               // swapRB
        false,              // crop
        CV_32F              // output data type
    );

    size_t bytes = model_input.total() * model_input.elemSize();
    size_t expected = buffers_->size(INPUT);
    if (expected != samplesCommon::BufferManager::kINVALID_SIZE_VALUE && expected != bytes) {
        std::cerr << "Warning: input bytes (" << bytes << ") != buffer size (" << expected << ")\n";
        return -1;
    }

    void* hostPtr = buffers_->getHostBuffer(INPUT);
    std::memcpy(hostPtr, model_input.data, bytes);

    __copyLSTMOutputsToInputs();

    first_inference_ = false;

    return 0;
}

void Model::inference(){
    Timer timer("Model::inference");

    buffers_->copyInputToDeviceAsync(stream_);
    context_->enqueueV2(buffers_->getDeviceBindings().data(), stream_, nullptr);
    buffers_->copyOutputToHostAsync(stream_);
    CHECK(cudaStreamSynchronize(stream_));
}

void Model::postprocess(){
    Timer timer("Model::postprocess");

    __decodePredictions();
}

void Model::__copyLSTMOutputsToInputs(){
    void* hiddenIn = buffers_->getHostBuffer(HIDDEN_IN);
    auto hiddenTensorSize = buffers_->size(HIDDEN_IN);
    
    void* cellIn = buffers_->getHostBuffer(CELL_IN);
    auto cellTensorSize = buffers_->size(CELL_IN);
    
    if(!first_inference_){
        void* hiddenOut = buffers_->getHostBuffer(HIDDEN_OUT);
        std::memcpy(hiddenIn, hiddenOut, hiddenTensorSize);
        
        void* cellOut = buffers_->getHostBuffer(CELL_OUT);
        std::memcpy(cellIn, cellOut, cellTensorSize);
    }

    else {
        std::memset(hiddenIn, 0, hiddenTensorSize);        
        std::memset(cellIn, 0, cellTensorSize);
    }
}

xt::xarray<float> Model::__toXTensor(const char* tensor_name) {
    if(engine_->getTensorShape(tensor_name).nbDims != 4){
        throw std::runtime_error("Unexpected tensor shape dimension");
    }

    float* h_ptr = reinterpret_cast<float *>(buffers_->getHostBuffer(tensor_name));
    
    auto shape = engine_->getTensorShape(tensor_name);
    std::vector<size_t> v_shape;
    for(int i = 0; i < shape.nbDims; i++){
        v_shape.emplace_back(shape.d[i]);
    }

    xt::xarray<float> ret = xt::adapt(
        std::move(h_ptr),
        buffers_->size(tensor_name) / sizeof(float),
        xt::no_ownership(),
        v_shape
    );
    
    return ret;
}

void Model::__decodePredictions(float conf_th, float nms_iou, int topk){
    // nms_iou = 0.2
    // topk = 50

    std::vector<DetectionInfo>(batch_size_).swap(detections_info_);

    std::vector<DetectionInfo> batch_results(batch_size_);
    std::vector<std::vector<cv::Rect2d>> bboxes_for_nms(batch_size_);
    std::vector<std::vector<float>> scores_for_nms(batch_size_);

    for(int l = 0; l < strides_.size(); l++){
        auto reg = __toXTensor(name_iter[l][REG]);
        auto obj = __toXTensor(name_iter[l][OBJ]);
        auto cls = __toXTensor(name_iter[l][CLS]);
        
        for(int b = 0; b < batch_size_; b++){
            int stride = strides_[l];

            auto obj_view = xt::view(obj, b, 0, xt::all(), xt::all());
            
            // @TODO 추후 multi class 추가 시 업데이트 해야 함.
            if(cls.shape(1) > 1)
                throw std::runtime_error("mutli class is not support");
            
            auto cls_view = xt::view(cls, b, 0, xt::all(), xt::all());
            
            auto obj_map = 1.0f / (1.0f + xt::exp(-obj_view));
            auto cls_map = 1.0f / (1.0f + xt::exp(-cls_view));
            auto score_map = obj_map * cls_map;
            
            auto cond = score_map > conf_th;
            if(!xt::any(cond)) continue;
            
            auto scores = xt::filter(score_map, cond);

            auto reg_view = xt::view(reg, b, xt::all(), xt::all(), xt::all());
            auto reg_transpose_view = xt::transpose(reg_view, {1, 2, 0});
            auto reg_map = xt::reshape_view(
                reg_transpose_view,
                {
                    static_cast<int>(cond.shape(0)) * static_cast<int>(cond.shape(1)), 
                    3,
                    2
                }
            );

            auto reg_all_indicies = xt::arange<size_t>(reg_map.shape(0));
            auto reg_threshold_indicies = xt::filter(
                reg_all_indicies, 
                xt::reshape_view(cond, { 
                    static_cast<int>(cond.shape(0)) * static_cast<int>(cond.shape(1)) 
                }
            ));
            auto pred_off = xt::eval(xt::view(reg_map, xt::keep(reg_threshold_indicies), xt::all(), xt::all()));
            
            auto arg_cond = xt::argwhere(cond);
            auto indicies = xt::from_indices(arg_cond);
            auto ys = xt::cast<float>(xt::view(indicies, xt::all(), 0));
            auto xs = xt::cast<float>(xt::view(indicies, xt::all(), 1));

            auto _centers = xt::stack(xt::xtuple((xs + 0.5) * stride, (ys + 0.5) * stride), 1);
            auto centers = xt::view(_centers, xt::all(), xt::newaxis(), xt::all());
            
            auto tri_np = (centers + pred_off * stride);
            int n = tri_np.shape(0);
            for(int i = 0; i < n; i++){
                auto p0 = xt::view(tri_np, i, 0, xt::all());
                auto p1 = xt::view(tri_np, i, 1, xt::all());
                auto p2 = xt::view(tri_np, i, 2, xt::all());
                
                auto p3 = 2 * p0 - p1;
                auto p4 = 2 * p0 - p2;

                auto poly = xt::eval(xt::stack(xt::xtuple(p1, p2, p3, p4), 0)); // shape (4,2)
                auto tri_eval = xt::eval(xt::view(tri_np, i, xt::all(), xt::all())); // shape (3,2)

                float sx = static_cast<float>(input_width_) / static_cast<float>(W);
                float sy = static_cast<float>(input_height_) / static_cast<float>(H);
                xt::xarray<float> scale = {sx, sy};
                auto poly_scaled = xt::eval(poly * scale);
                auto tri_scaled = xt::eval(tri_eval * scale);

                batch_results[b].scores.push_back(static_cast<float>(scores(i)));
                batch_results[b].poly4s.push_back(poly_scaled);
                batch_results[b].tri_ptss.push_back(tri_scaled);

                auto mins = xt::amin(poly_scaled, {0});
                auto maxs = xt::amax(poly_scaled, {0});
                float x0 = static_cast<float>(mins(0));
                float y0 = static_cast<float>(mins(1));
                float x1 = static_cast<float>(maxs(0));
                float y1 = static_cast<float>(maxs(1));

                bboxes_for_nms[b].push_back(cv::Rect2d(x0, y0, x1 - x0, y1 - y0));
                scores_for_nms[b].push_back(static_cast<float>(scores(i)));
            }
        }
    }

    for(int b = 0; b < batch_size_; b++){
        std::vector<int> batch_detections_indices_;

        cv::dnn::NMSBoxes(
            bboxes_for_nms[b],
            scores_for_nms[b],
            conf_th,
            nms_iou,
            batch_detections_indices_,
            1.0f,
            topk
        );

        for(auto idx: batch_detections_indices_){
            std::vector<cv::Point2f> pts = {
                cv::Point2f(batch_results[b].poly4s[idx](0, 0), batch_results[b].poly4s[idx](0, 1)),
                cv::Point2f(batch_results[b].poly4s[idx](1, 0), batch_results[b].poly4s[idx](1, 1)),
                cv::Point2f(batch_results[b].poly4s[idx](2, 0), batch_results[b].poly4s[idx](2, 1)),
                cv::Point2f(batch_results[b].poly4s[idx](3, 0), batch_results[b].poly4s[idx](3, 1))
            };

            cv::RotatedRect rect = cv::minAreaRect(pts);

            float width  = rect.size.width;
            float height = rect.size.height;
            float angle  = rect.angle;

            if(width < 0.3 || height < 0.3){
                continue;
            }

            if(width * height < 20.0){
                continue;
            }

            detections_info_[b].scores.emplace_back(batch_results[b].scores[idx]);
            detections_info_[b].poly4s.emplace_back(batch_results[b].poly4s[idx]);
            detections_info_[b].tri_ptss.emplace_back(batch_results[b].tri_ptss[idx]);
        }
    }
}
