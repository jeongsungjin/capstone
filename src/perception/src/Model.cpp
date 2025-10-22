#include "perception/Model.h"

#include "perception/utils.hpp"

#include <numeric>

Model::Model(const std::string& pkg_path){
    std::vector<char> engineData = readPlanFile(
        pkg_path + "/engine/model_cm89.plan"
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

void Model::preprocess(const cv::Mat& img){
    input_width_ = img.cols;
    input_height_ = img.rows;

    // @TODO : 일단 이미지 1장에 대한 처리로 진행을 하고
    // 이후에 이미지 4장을 std::vector<cv::Mat>에 담고, 
    // cv::dnn::blobFromImages에 태워야 함.
    cv::Mat model_input = cv::dnn::blobFromImage(
        img,                // src img
        1. / 255.,          // scale factor
        cv::Size(W, H),     // resize size
        cv::Scalar(),       // mean
        true,               // swapRB
        false,              // crop
        CV_16FC3            // output data type
    );

    size_t bytes = model_input.total() * model_input.elemSize();
    size_t expected = buffers_->size(layer_names::INPUT);
    if (expected != samplesCommon::BufferManager::kINVALID_SIZE_VALUE && expected != bytes) {
        std::cerr << "Warning: input bytes (" << bytes << ") != buffer size (" << expected << ")\n";
        return -1;
    }

    void* hostPtr = buffers_->getHostBuffer(layer_names::INPUT);
    std::memcpy(hostPtr, model_input.data, bytes);
}

void Model::inference(){
    buffers_->copyInputToDeviceAsync(stream_);
    context_->enqueueV2(buffers_->getDeviceBindings().data(), stream_, nullptr);
    buffers_->copyOutputToHostAsync(stream_);
    CHECK(cudaStreamSynchronize(stream_));
}

void Model::postprocess(){
    
}
