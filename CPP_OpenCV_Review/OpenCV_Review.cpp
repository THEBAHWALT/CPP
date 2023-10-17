#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn.hpp>
int main() {
    // 設定圖片路徑
    std::string imagePath = "./Dog.jpg";

    // 讀取圖片
    cv::Mat originalImage = cv::imread(imagePath);

    // 檢查圖片是否成功讀取
    if (originalImage.empty()) {
        std::cout << "無法讀取圖片: " << imagePath << std::endl;
        return -1;
    }

    cv::Mat resizedImage;
    cv::resize(originalImage, resizedImage, cv::Size(640, 480));

    // 將圖片轉換為灰階
    cv::Mat grayscaleImage;
    cv::cvtColor(resizedImage, grayscaleImage, cv::COLOR_BGR2GRAY);

    // 顯示原始圖片和灰階圖片
    cv::imshow("resized Image", resizedImage);
    cv::imshow("Grayscale Image", grayscaleImage);

    // 調整曝光值（假設增加50曝光值）
    cv::Mat adjustedImage = grayscaleImage + 50;

    // 顯示調整後的圖片
    cv::imshow("Adjusted Image", adjustedImage);
    cv::cuda::GpuMat dst, src;
    src.upload(grayscaleImage);
    cv::Ptr<cv::cuda::CannyEdgeDetector> cannyDetector = cv::cuda::createCannyEdgeDetector(50.0, 150.0);
    cannyDetector->detect(src, dst);
    cv::Mat result;
    dst.download(result);
    cv::imshow("result", result);

    // 等待使用者按下任意鍵，然後關閉視窗
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}