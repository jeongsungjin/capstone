#include "perception/model.h"

#include "timer.h"

#include "perception/preprocess.h"

#include <numeric>

#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xmasked_view.hpp>

#include <algorithm>
#include <opencv2/cudawarping.hpp>

#include <iostream>

int W = 1536;
int H = 864;

Logger gLogger;

using namespace layer_names;

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
    
    buffers_ = std::make_unique<samplesCommon::BufferManager>(engine_, batch_size_, context_.get());
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* tname = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(tname);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            context_->setInputTensorAddress(tname, buffers_->getDeviceBuffer(tname));
        } else {
            context_->setOutputTensorAddress(tname, buffers_->getDeviceBuffer(tname));
        }
    }
    CHECK(cudaStreamCreate(&stream_));
    
    cv_stream_ = cv::cuda::StreamAccessor::wrapStream(stream_);
}

Model::~Model(){
    cudaStreamDestroy(stream_);
}

int Model::preprocess(const std::vector<std::shared_ptr<cv::Mat>>& images){
    Timer timer("preprocess");

    if (images.empty()) return -1;

    int effective_batch = std::min(static_cast<int>(images.size()), batch_size_);
    if (effective_batch != batch_size_) {
        std::cout << "preprocess: images.size() (" << images.size() << ") != model batch_size_ (" << batch_size_ << ") — using " << effective_batch << std::endl;
    }

    __copyLSTMOutputsToInputs();
    
    first_inference_ = false;
    
    // 입력 원본 크기는 첫 장 기준 기록 (시각화 스케일링 용)
    input_width_ = images[0]->cols;
    input_height_ = images[0]->rows;

    // 하나의 큰 GpuMat을 만들고, 각 이미지를 리사이즈 후 ROI에 복사
    cv::cuda::GpuMat d_stacked(H * effective_batch, W, CV_8UC3);

    for (int b = 0; b < effective_batch; ++b) {
        cv::cuda::GpuMat d_img, d_resized;
        d_img.upload(*images[b], cv_stream_);
        cv::cuda::resize(d_img, d_resized, cv::Size(W, H), 0, 0, cv::INTER_LINEAR, cv_stream_);

        cv::cuda::GpuMat roi = d_stacked.rowRange(b * H, (b + 1) * H).colRange(0, W);
        d_resized.copyTo(roi, cv_stream_);
    }

    float* d_out_base = reinterpret_cast<float*>(buffers_->getDeviceBuffer(INPUT));

    launchPreprocessKernelBatched(
        d_stacked,
        d_out_base,
        W,
        H,
        effective_batch,
        1.0f / 255.0f,
        stream_
    );

    return 0;
}

void Model::inference(){
    Timer timer("inference");

    context_->enqueueV3(stream_);
    buffers_->copyOutputToHostAsync(stream_);
    CHECK(cudaStreamSynchronize(stream_));
}

void Model::postprocess(std_msgs::msg::Float32MultiArray& detection_msg){
    Timer timer("postprocess");
    constexpr float kConf = 0.30f;
    constexpr float kNms = 0.20f;
    constexpr int   kTopk = 50;

    __decodePredictions(kConf, kNms, kTopk);

    const std::size_t num_cams = detections_info_.size();
    std::cout << num_cams << " cameras' detections to publish." << std::endl;

    std::size_t max_det = 0;
    for (const auto& info : detections_info_) {
        max_det = std::max(max_det, info.scores.size());
    }

    constexpr std::size_t feature_len = 16; // [valid, score, tri(6), poly(8)]

    detection_msg.layout.dim.clear();
    detection_msg.layout.data_offset = 0;

    if (num_cams == 0) {
        detection_msg.data.clear();
        return;
    }

    detection_msg.layout.dim.resize(3);
    detection_msg.layout.dim[0].label = "camera";
    detection_msg.layout.dim[0].size = static_cast<uint32_t>(num_cams);
    detection_msg.layout.dim[0].stride = static_cast<uint32_t>(max_det * feature_len);

    detection_msg.layout.dim[1].label = "detection";
    detection_msg.layout.dim[1].size = static_cast<uint32_t>(max_det);
    detection_msg.layout.dim[1].stride = static_cast<uint32_t>(feature_len);

    detection_msg.layout.dim[2].label = "feature";
    detection_msg.layout.dim[2].size = static_cast<uint32_t>(feature_len);
    detection_msg.layout.dim[2].stride = 1;

    const std::size_t total_values = num_cams * max_det * feature_len;
    detection_msg.data.assign(total_values, 0.0f);

    if (max_det == 0) {
        return;
    }

    const auto write_tri = [&](const auto& tri, std::size_t base_idx) {
        detection_msg.data[base_idx + 2] = static_cast<float>(tri(0, 0));
        detection_msg.data[base_idx + 3] = static_cast<float>(tri(0, 1));
        detection_msg.data[base_idx + 4] = static_cast<float>(tri(1, 0));
        detection_msg.data[base_idx + 5] = static_cast<float>(tri(1, 1));
        detection_msg.data[base_idx + 6] = static_cast<float>(tri(2, 0));
        detection_msg.data[base_idx + 7] = static_cast<float>(tri(2, 1));
    };

    const auto write_poly = [&](const auto& poly, std::size_t base_idx) {
        detection_msg.data[base_idx + 8]  = static_cast<float>(poly(0, 0));
        detection_msg.data[base_idx + 9]  = static_cast<float>(poly(0, 1));
        detection_msg.data[base_idx + 10] = static_cast<float>(poly(1, 0));
        detection_msg.data[base_idx + 11] = static_cast<float>(poly(1, 1));
        detection_msg.data[base_idx + 12] = static_cast<float>(poly(2, 0));
        detection_msg.data[base_idx + 13] = static_cast<float>(poly(2, 1));
        detection_msg.data[base_idx + 14] = static_cast<float>(poly(3, 0));
        detection_msg.data[base_idx + 15] = static_cast<float>(poly(3, 1));
    };

    for (std::size_t cam = 0; cam < num_cams; ++cam) {
        const auto& info = detections_info_[cam];
        const std::size_t det_count = std::min<std::size_t>(info.scores.size(), max_det);
        for (std::size_t det = 0; det < det_count; ++det) {
            const std::size_t base = (cam * max_det + det) * feature_len;
            detection_msg.data[base + 0] = 1.0f;
            detection_msg.data[base + 1] = info.scores[det];
            write_tri(info.tri_ptss[det], base);
            write_poly(info.poly4s[det], base);
        }
    }
}

void Model::__copyLSTMOutputsToInputs(){
    // Use device->device copies to avoid host roundtrips
    void* hiddenIn_dev = buffers_->getDeviceBuffer(HIDDEN_IN);
    size_t hiddenTensorSize = buffers_->size(HIDDEN_IN);

    void* cellIn_dev = buffers_->getDeviceBuffer(CELL_IN);
    size_t cellTensorSize = buffers_->size(CELL_IN);

    if (!first_inference_) {
        void* hiddenOut_dev = buffers_->getDeviceBuffer(HIDDEN_OUT);
        CHECK(cudaMemcpyAsync(hiddenIn_dev, hiddenOut_dev, hiddenTensorSize, cudaMemcpyDeviceToDevice, stream_));

        void* cellOut_dev = buffers_->getDeviceBuffer(CELL_OUT);
        CHECK(cudaMemcpyAsync(cellIn_dev, cellOut_dev, cellTensorSize, cudaMemcpyDeviceToDevice, stream_));
    } else {
        // initialize device buffers to zero
        CHECK(cudaMemsetAsync(hiddenIn_dev, 0, hiddenTensorSize, stream_));
        CHECK(cudaMemsetAsync(cellIn_dev, 0, cellTensorSize, stream_));
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
    // (float conf_th=0.15, float nms_iou=0.2, int topk=50);

    std::vector<DetectionInfo>(batch_size_).swap(detections_info_);

    std::vector<DetectionInfo> batch_results(batch_size_);
    std::vector<std::vector<cv::Rect2d>> bboxes_for_nms(batch_size_);

    for(int l = 0; l < strides_.size(); l++){
        auto reg = __toXTensor(name_iter[l][REG]);
        auto obj = __toXTensor(name_iter[l][OBJ]);
        auto cls = __toXTensor(name_iter[l][CLS]);
        float stride = static_cast<float>(strides_[l]);

        for(int b = 0; b < batch_size_; b++){
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
            }
        }
    }

    constexpr double contain_thr = 0.7;
    constexpr double min_area = 20.0;
    constexpr double min_edge = 3.0;
    constexpr double eps = 1e-9;

    for (int b = 0; b < batch_size_; ++b) {
        const auto candidate_count = batch_results[b].scores.size();
        if (candidate_count == 0) {
            continue;
        }

        std::vector<size_t> order(candidate_count);
        std::iota(order.begin(), order.end(), static_cast<size_t>(0));
        std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
            return batch_results[b].scores[lhs] > batch_results[b].scores[rhs];
        });

        std::vector<size_t> kept;
        const int max_keep = (topk > 0) ? std::min(topk, static_cast<int>(candidate_count)) : static_cast<int>(candidate_count);
        if (max_keep > 0) {
            kept.reserve(static_cast<size_t>(max_keep));
        }

        for (size_t idx : order) {
            const auto& bboxA = bboxes_for_nms[b][idx];
            const double areaA = std::max(0.0, bboxA.width) * std::max(0.0, bboxA.height);
            if (areaA <= eps) {
                continue;
            }

            bool suppress = false;
            for (size_t kept_idx : kept) {
                const auto& bboxB = bboxes_for_nms[b][kept_idx];
                const double x0 = std::max(bboxA.x, bboxB.x);
                const double y0 = std::max(bboxA.y, bboxB.y);
                const double x1 = std::min(bboxA.x + bboxA.width, bboxB.x + bboxB.width);
                const double y1 = std::min(bboxA.y + bboxA.height, bboxB.y + bboxB.height);
                const double iw = std::max(0.0, x1 - x0);
                const double ih = std::max(0.0, y1 - y0);
                const double intersection = iw * ih;

                const double areaB = std::max(0.0, bboxB.width) * std::max(0.0, bboxB.height);
                const double union_area = areaA + areaB - intersection;
                const double iou = union_area > eps ? intersection / union_area : 0.0;
                const double ios = intersection / std::max(std::min(areaA, areaB), eps);

                if ((nms_iou > 0.0f && iou >= nms_iou) || (contain_thr > 0.0 && ios >= contain_thr)) {
                    suppress = true;
                    break;
                }
            }

            if (!suppress) {
                kept.push_back(idx);
                if (topk > 0 && static_cast<int>(kept.size()) >= topk) {
                    break;
                }
            }
        }

        for (size_t idx : kept) {
            const auto& poly4 = batch_results[b].poly4s[idx];

            const cv::Point2f p0(poly4(0, 0), poly4(0, 1));
            const cv::Point2f p1(poly4(1, 0), poly4(1, 1));
            const cv::Point2f p2(poly4(2, 0), poly4(2, 1));
            const cv::Point2f p3(poly4(3, 0), poly4(3, 1));

            const cv::Point2f v01 = p1 - p0;
            const cv::Point2f v03 = p3 - p0;
            const double parallelogram_area = std::abs(v01.x * v03.y - v01.y * v03.x);

            const double e0 = cv::norm(p1 - p0);
            const double e1 = cv::norm(p2 - p1);
            const double e2 = cv::norm(p3 - p2);
            const double e3 = cv::norm(p0 - p3);
            const double min_edge_len = std::min(std::min(e0, e1), std::min(e2, e3));

            if (parallelogram_area < min_area || min_edge_len < min_edge) {
                continue;
            }

            detections_info_[b].scores.emplace_back(batch_results[b].scores[idx]);
            detections_info_[b].poly4s.emplace_back(batch_results[b].poly4s[idx]);
            detections_info_[b].tri_ptss.emplace_back(batch_results[b].tri_ptss[idx]);
        }
    }
}
