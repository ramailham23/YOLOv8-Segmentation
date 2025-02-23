#include "inference.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace dnn;
using namespace std;

// Warna untuk segmentasi
Scalar colors[] = {
    Scalar(0, 0, 255),   // Red (Sawah)
    Scalar(255, 0, 0),   // Blue (Perumahan)
    Scalar(0, 255, 0)    // Green (Sungai)
};

Inference::Inference(const std::string &onnxModelPath, const cv::Size &modelInputShape,
    const std::string &yamlPath, const bool &runWithCuda)
: modelPath(onnxModelPath), yamlPath(yamlPath), cudaEnabled(runWithCuda), modelShape(modelInputShape) {
    loadClassesFromYAML();
    loadOnnxNetwork();
}

void Inference::loadClassesFromYAML() {
    YAML::Node config = YAML::LoadFile(yamlPath);
    if (config["names"]) {
        for (const auto &name : config["names"]) {
            classes.push_back(name.as<string>());
        }
    } else {
        cerr << "❌ Error: Tidak dapat membaca kelas dari " << yamlPath << endl;
    }
}

void Inference::loadOnnxNetwork() {
    net = readNet(modelPath);
    if (net.empty()) {
        cerr << "❌ Error: Model gagal dimuat! Periksa path ke model." << endl;
        return;
    }

    if (cudaEnabled) {
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);
    } else {
        net.setPreferableBackend(DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(DNN_TARGET_CPU);
    }
}

vector<Detection> Inference::runInference(const Mat &input) {
    vector<Detection> detections;
    Mat blob;
    blobFromImage(input, blob, 1.0 / 255.0, modelShape, Scalar(), false, true);
    net.setInput(blob);

    vector<Mat> outputs;
    vector<String> outputNames = net.getUnconnectedOutLayersNames();
    cout << "Output Layers dari Model: " << endl;
    for (const auto &name : outputNames) {
        cout << " - " << name << endl;
    }

    net.forward(outputs, net.getUnconnectedOutLayersNames());
    cout << "✅ Model berhasil melakukan inferensi!" << endl;
    cout << "Dimensi Output0 (deteksi): " << outputs[0].size << endl;
    cout << "Dimensi Output1 (segmentasi): " << outputs[1].size << endl;

    Mat detectionMat = outputs[0];
    Mat maskMat = outputs[1];

    float confThreshold = 0.2;
    for (int i = 0; i < detectionMat.rows; i++) {
        float confidence = detectionMat.at<float>(i, 4);
        if (confidence > confThreshold) {
            int classId = (int)detectionMat.at<float>(i, 5);
            int x = (int)(detectionMat.at<float>(i, 0) * input.cols);
            int y = (int)(detectionMat.at<float>(i, 1) * input.rows);
            int w = (int)(detectionMat.at<float>(i, 2) * input.cols);
            int h = (int)(detectionMat.at<float>(i, 3) * input.rows);

            Detection detection;
            detection.class_id = classId;
            detection.className = classes[classId];
            detection.confidence = confidence;
            detection.box = Rect(x, y, w, h);
            detection.color = colors[classId];

            // Mask segmentation
            Mat mask = maskMat.row(i).reshape(1, 160);
            resize(mask, mask, Size(w, h));
            threshold(mask, mask, 0.5, 255, THRESH_BINARY);
            detection.mask = mask;

            detections.push_back(detection);
        }
    }
    return detections;
}
