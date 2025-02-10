#include "inference.h"

Inference::Inference(const std::string &onnxModelPath, const cv::Size &modelInputShape, const std::string &yamlFilePath, const bool &runWithCuda)
{
    modelPath = onnxModelPath;
    modelShape = modelInputShape;
    yamlPath = yamlFilePath;
    cudaEnabled = runWithCuda;

    loadOnnxNetwork();
    loadClassesFromYAML();
}

void Inference::loadClassesFromYAML()
{
    try {
        YAML::Node config = YAML::LoadFile(yamlPath);
        if (config["names"])
        {
            for (const auto &name : config["names"])
            {
                classes.push_back(name.as<std::string>());
            }
        }
        else
        {
            std::cerr << "⚠️ Warning: Tidak menemukan 'names' di data.yaml! Pastikan file YAML benar. \n";
        }
    } catch (const YAML::Exception &e) {
        std::cerr << "❌ Error membaca data.yaml: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Inference::loadOnnxNetwork()
{
    net = cv::dnn::readNetFromONNX(modelPath);
    if (cudaEnabled)
    {
        std::cout << "\n🚀 Running on CUDA" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        std::cout << "\n💻 Running on CPU" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

cv::Mat Inference::formatToSquare(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

std::vector<Detection> Inference::runInference(const cv::Mat &input)
{
    cv::Mat modelInput = input;
    if (letterBoxForSquare && modelShape.width == modelShape.height)
        modelInput = formatToSquare(modelInput);

    cv::Mat blob;
    cv::dnn::blobFromImage(modelInput, blob, 1.0 / 255.0, modelShape, cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    int rows = outputs[0].size[1];
    int dimensions = outputs[0].size[2];

    bool yolov8 = false;
    if (dimensions > rows)
    {
        yolov8 = true;
        rows = outputs[0].size[2];
        dimensions = outputs[0].size[1];

        outputs[0] = outputs[0].reshape(1, dimensions);
        cv::transpose(outputs[0], outputs[0]);
    }
    float *data = (float *)outputs[0].data;

    float x_factor = modelInput.cols / modelShape.width;
    float y_factor = modelInput.rows / modelShape.height;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<cv::Mat> masks;

    int mask_dim = dimensions - (yolov8 ? 4 : 5) - classes.size();

    for (int i = 0; i < rows; ++i)
    {
        float *classes_scores = yolov8 ? (data + 4) : (data + 5);
        cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore;
        minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > modelScoreThreshold)
        {
            confidences.push_back(maxClassScore);
            class_ids.push_back(class_id.x);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));

            if (mask_dim > 0)
            {
                float *mask_data = data + (yolov8 ? 4 : 5) + classes.size();
                int mask_size = std::sqrt(mask_dim);
                cv::Mat mask(mask_size, mask_size, CV_32FC1, mask_data);
                
                // Normalisasi mask ke [0,255] dan konversi ke 8-bit grayscale
                cv::normalize(mask, mask, 0, 255, cv::NORM_MINMAX);
                mask.convertTo(mask, CV_8UC1);
                
                masks.push_back(mask.clone());
            }
        }
        data += dimensions;
    }

    // 🔹 Terapkan Non-Maximum Suppression (NMS) untuk menghapus duplikasi
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, 0.45, indices);

    std::vector<Detection> detections;
    for (size_t i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];

        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];

        result.color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
        result.className = classes[result.class_id];
        result.box = boxes[idx];
        result.mask = masks[idx];

        detections.push_back(result);
    }

    return detections;
}
