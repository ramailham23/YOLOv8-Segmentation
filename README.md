# 🌾🏡 YOLOv8 Segmentation for Residential & Agricultural Areas

This system utilizes **YOLOv8-Segmentation** to detect **residential 🏠 and agricultural 🌾 areas** from aerial/satellite images.
The system performs the following tasks:
1. **Detects residential and agricultural regions** in aerial imagery.
2. **Generates segmentation masks** for precise boundary identification.
3. **Uses OpenCV DNN with ONNX for efficient inference** on both CPU and GPU.

---

## 📂 **1. Project Structure**
YOLOv8-Detection/

│── models/                        # YOLOv8 ONNX model

│── build/                         # Compiled project binaries

├── inference.h                    # Header file for inference

├── inference.cpp                  # Main inference logic

├── yolov8Normal.cpp               # Main program

│── data.yaml                      # Configuration file for dataset

│── CMakeLists.txt                 # Build configuration file

│── README.md                      # Project documentation


---

## 🔧 **2. Installation Guide**
1. Clone The Repository:
https://github.com/ramailham23/YOLOv8-Segmentation.git
cd yolov8-detection

2. Install Dependencies:
sudo apt update
sudo apt install cmake g++ libopencv-dev

3. Build & Compile:
mkdir build && cd build
cmake ..
make -j$(nproc)

4. Run the Detector:
./yo path/to/video.mp4

*(Replace path/to/video.mp4 with the actual video file path.)*

---

## 🔍 **3. How It Works**
1. The system loads a trained **YOLOv8-Segmentation** model.
2. It processes **satellite/aerial images to detect residential and agricultural regions.**
3. The output includes:
   - **Bounding boxes** for identified areas.
   - **Segmentation masks** to highlight the detected regions.
4. **NMS (Non-Maximum Suppression)** is applied to remove duplicate detections.

---

## 📌 **4. License**
This project is licensed under the **MIT License.**

---

## 📞 **5. Contact**
For any questions or issues, feel free to reach out:

**📩 Email:** mohilhamramadana721@gmail.com

**💬 Telegram:** @Ambatronus
