#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn.hpp>
int main() {
    // �]�w�Ϥ����|
    std::string imagePath = "./Dog.jpg";

    // Ū���Ϥ�
    cv::Mat originalImage = cv::imread(imagePath);

    // �ˬd�Ϥ��O�_���\Ū��
    if (originalImage.empty()) {
        std::cout << "�L�kŪ���Ϥ�: " << imagePath << std::endl;
        return -1;
    }

    cv::Mat resizedImage;
    cv::resize(originalImage, resizedImage, cv::Size(640, 480));

    // �N�Ϥ��ഫ���Ƕ�
    cv::Mat grayscaleImage;
    cv::cvtColor(resizedImage, grayscaleImage, cv::COLOR_BGR2GRAY);

    // ��ܭ�l�Ϥ��M�Ƕ��Ϥ�
    cv::imshow("resized Image", resizedImage);
    cv::imshow("Grayscale Image", grayscaleImage);

    // �վ��n���ȡ]���]�W�[50�n���ȡ^
    cv::Mat adjustedImage = grayscaleImage + 50;

    // ��ܽվ�᪺�Ϥ�
    cv::imshow("Adjusted Image", adjustedImage);
    cv::cuda::GpuMat dst, src;
    src.upload(grayscaleImage);
    cv::Ptr<cv::cuda::CannyEdgeDetector> cannyDetector = cv::cuda::createCannyEdgeDetector(50.0, 150.0);
    cannyDetector->detect(src, dst);
    cv::Mat result;
    dst.download(result);
    cv::imshow("result", result);

    // ���ݨϥΪ̫��U���N��A�M����������
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}