#pragma once
#include "ObjectDetector.h"
#include <opencv2/tracking.hpp> 
#define MAX_MISSED_FRAMES 5
#define MATCH_THRESHOLD 0.5
#define MIN_IOU 0.1

struct dataTracker
{
	cv::Rect trackBox;
	int trackMissedCounter;
	cv::Ptr<cv::Tracker> tracker;
};
class ObjectTracker
{
public:
	ObjectTracker();
	~ObjectTracker();
	void detectNewCars(const cv::Mat& frame, const vector<cv::Rect>& newDetections);
	void updateTrackers(const cv::Mat& frame);
	void updateMissedCounts(const vector<cv::Rect>& newDetections);
	cv::Mat drawBoxes(const cv::Mat& image);

private:
	vector<dataTracker> dataTrackers;
	double computeIoU(const cv::Rect& a, const cv::Rect& b);
	int g_carCount = 0;
};

