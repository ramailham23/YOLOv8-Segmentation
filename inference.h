#ifndef INFERENCE_H
#define INFERENCE_H

#include <fstream>
#include <vector>
#include <string>
#include <random>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "yaml-cpp/yaml.h"

struct Detection
{
    int class_id{0};
    std::string className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
    cv::Mat mask;
};

class Inference
{
public:
    Inference(const std::string &onnxModelPath, const cv::Size &modelInputShape, const std::string &yamlPath, const bool &runWithCuda);
    std::vector<Detection> runInference(const cv::Mat &input);

private:
    void loadClassesFromYAML();
    void loadOnnxNetwork();
    cv::Mat formatToSquare(const cv::Mat &source);

    std::string modelPath{};
    std::string yamlPath{};
    bool cudaEnabled{};

    std::vector<std::string> classes;
    cv::Size2f modelShape{};
    float modelConfidenceThreshold{0.25};
    float modelScoreThreshold{0.25};
    float modelNMSThreshold{0.25};
    bool letterBoxForSquare = true;
    cv::dnn::Net net;
};

#endif // INFERENCE_H