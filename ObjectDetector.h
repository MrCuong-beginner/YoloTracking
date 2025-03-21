#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

#define SCALE (1.0 / 255.0)
#define INPUT_WIDTH 640.0
#define INPUT_HEIGHT 640.0
#define CONFIDENCE_THRESHOLD 0.5
#define SCORE_THRESHOLD 0.4
#define NMS_THRESHOLD 0.2
#define AREAS_THRESHOLD 3600

using namespace std;

struct Predicted
{
    int classId;
    float confidence;
    cv::Rect box;
};
class ObjectDetector
{
public:
    explicit ObjectDetector(const string& modelPath, const string& classFile);
    vector<Predicted> boundingBoxDetector(const cv::Mat& image);
    cv::Mat drawBoundingBoxes(const cv::Mat& image, const vector<Predicted>& detected);
    vector<cv::Rect> getCars(const cv::Mat& image);

private:
    vector<cv::Mat> getOutputs(const cv::Mat& image);
    vector<string> loadClasses(const string& path);
    vector<Predicted> applyNMS(const vector<Predicted>& predictions);


    cv::dnn::Net net;
    vector<string> classList;
};

