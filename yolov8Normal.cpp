#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "yaml-cpp/yaml.h"
#include "inference.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        cerr << "❌ Penggunaan: ./yo <path_video>" << endl;
        return -1;
    }

    string videoPath = argv[1];
    VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        cerr << "❌ Error: Tidak dapat membuka video " << videoPath << endl;
        return -1;
    }

    YAML::Node config = YAML::LoadFile("/home/ramailham/yolov8_detection/yoloNormal.yaml");
    int m_size = config["size"].as<int>();
    std::string model_path = config["model"].as<std::string>();
    std::string yaml_path = "/home/ramailham/yolov8_detection/data.yaml";

    bool runOnGPU = false;
    Inference inf(model_path, cv::Size(m_size, m_size), yaml_path, runOnGPU);

    Mat frame;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;

        vector<Detection> output = inf.runInference(frame);

        for (const auto &detection : output)
        {
            rectangle(frame, detection.box, detection.color, 2);
            putText(frame, detection.className, Point(detection.box.x, detection.box.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);

            // Jika mask tersedia, gambarkan di atas gambar
            if (!detection.mask.empty())
            {
                cv::Mat mask_resized;
                cv::resize(detection.mask, mask_resized, detection.box.size());

                // 🔴 Pastikan mask dikonversi ke format CV_8UC1 (8-bit grayscale)
                cv::Mat mask_uint8;
                mask_resized.convertTo(mask_uint8, CV_8UC1, 255); // Normalisasi ke 0-255

                // 🔵 Gunakan colormap untuk menampilkan mask dengan warna
                cv::Mat coloredMask;
                applyColorMap(mask_uint8, coloredMask, cv::COLORMAP_JET);

                // 🛠 Perbaikan: Cegah bounding box keluar dari batas gambar
                int x = std::max(0, detection.box.x);
                int y = std::max(0, detection.box.y);
                int width = std::min(frame.cols - x, detection.box.width);
                int height = std::min(frame.rows - y, detection.box.height);

                // Jika width atau height negatif, lewati iterasi ini
                if (width <= 0 || height <= 0)
                    continue;

                cv::Mat roi = frame(cv::Rect(x, y, width, height));
                cv::Mat mask_roi = coloredMask(cv::Rect(0, 0, width, height));

                addWeighted(roi, 0.7, mask_roi, 0.3, 0, roi);
            }
        }

        imshow("YOLOv8 Segmentation", frame);
        if (waitKey(1) == 27)
            break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
