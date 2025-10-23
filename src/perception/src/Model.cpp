#include "perception/Model.h"

#include "perception/utils.hpp"

#include <numeric>

int W = 1536;
int H = 864;

Model::Model(const std::string& pkg_path){
    std::vector<char> engineData = readPlanFile(
        pkg_path + "/engine/static_model.plan"
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
    size_t expected = buffers_->size(layer_names::INPUT);
    if (expected != samplesCommon::BufferManager::kINVALID_SIZE_VALUE && expected != bytes) {
        std::cerr << "Warning: input bytes (" << bytes << ") != buffer size (" << expected << ")\n";
        return -1;
    }

    void* hostPtr = buffers_->getHostBuffer(layer_names::INPUT);
    std::memcpy(hostPtr, model_input.data, bytes);

    return 0;
}

void Model::inference(){
    buffers_->copyInputToDeviceAsync(stream_);
    context_->enqueueV2(buffers_->getDeviceBindings().data(), stream_, nullptr);
    buffers_->copyOutputToHostAsync(stream_);
    CHECK(cudaStreamSynchronize(stream_));
}

void Model::postprocess(){
    
}
