#include "Model.h"

#include "utils.hpp"

Model::Model(){
    std::vector<char> engineData = readPlanFile("/home/ctrl/capstone/src/perception/engine/model_cm89.plan");

    runtime_ = createInferRuntime(gLogger);
    engine_ = runtime_->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine_) throw std::runtime_error("Failed to deserialize engine");
    
    context_ = engine_->createExecutionContext();
    if (!context_) throw std::runtime_error("Failed to create context");
    
    int total_io_binding = engine_->getNbIOTensors();
    for(int i = 0; i < total_io_binding; i++){
        const char* tensor_name = engine_->getIOTensorName(i);
        const auto io_mode = engine_->getTensorIOMode(tensor_name);
        if(io_mode == TensorIOMode::kNONE)
            throw std::runtime_error("Invalid model");

        int size = getTensorBytesPerComponent(tensor_name);
        Dims shape = engine_->getTensorShape(tensor_name);
        for(int j = 0; j < shape.nbDims; j++){
            size *= shape.d[j];
        }

        if(io_mode == TensorIOMode::kOUTPUT){
            output_names_.push_back(tensor_name);
            output_sizes_.push_back(size);
            output_shapes_.push_back(shape);
        }
        
        else { // io_mode == TensorIOMode::kINPUT
            input_names_.push_back(tensor_name);
            input_sizes_.push_back(size);
            input_shapes_.push_back(shape);
        }
    }

    std::vector<void *>(input_names_.size()).swap(input_buffers_);
    std::vector<void *>(output_names_.size()).swap(output_buffers_);

    for(int input_idx = 0; input_idx < input_sizes_.size(); input_idx++){
        CUDA_CHECK(cudaMalloc(&input_buffers_[input_idx], input_sizes_[input_idx]));
    }

    for(int output_idx = 0; output_idx < output_sizes_.size(); output_idx++){
        CUDA_CHECK(cudaMalloc(&output_buffers_[output_idx], output_sizes_[output_idx]));
    }

    CUDA_CHECK(cudaStreamCreate(&stream_));
}

Model::~Model(){
    cudaStreamDestroy(stream_);

    for(auto in_ptr: input_buffers_){
        CUDA_CHECK(cudaFree(in_ptr));
    }

    for(auto out_ptr: output_buffers_){
        CUDA_CHECK(cudaFree(out_ptr));
    }

    context_->destroy();
    engine_->destroy();
    runtime_->destroy();
}

cv::Mat Model::preprocess(const cv::Mat& img){
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

    io_size_[INPUT_INDEX] = model_input.total() * model_input.elemSize();

    return model_input;
}

void Model::inference(const cv::Mat& model_input){
    CUDA_CHECK(cudaMemcpyAsync(
        input_buffers_[INPUT_INDEX], 
        model_input.data, 
        io_size_[INPUT_INDEX], 
        cudaMemcpyHostToDevice, 
        stream_
    ));

    context_->enqueueV2(input_buffers_, stream_, nullptr);
    cudaStreamSynchronize(stream_);
}

void Model::postprocess(){
    
}
