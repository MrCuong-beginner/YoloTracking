#include "ObjectTracker.h"
//#include <iostream>
using namespace std;
int main()
{
    string modelPath = "D:\\Visual Studio\\data\\yolov5s.onnx";
    string classFile = "D:\\Visual Studio\\data\\coco.names";
    string videoPath = "D:\\Visual Studio\\data\\test.MOV";

    unique_ptr<ObjectDetector> pDetector = make_unique<ObjectDetector>(modelPath, classFile);
    unique_ptr<ObjectTracker> pTracking = make_unique<ObjectTracker>();

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cout << "Cannot open video file!" << endl;
        return -1;
    }
    cv::Mat frame;

    while (cap.read(frame)) {
        if (frame.empty()) {
            cerr << "Could not grab the first frame!" << endl;
            return -1;
        }

        // 新規検出の取得
        vector<cv::Rect> newDetections = pDetector->getCars(frame);



        //新規トラッカーの初期化
        pTracking->detectNewCars(frame, newDetections);
        //pTracking->detectAndUpdateCars(frame, newDetections);

        // 既存トラッカーの状態更新
        pTracking->updateTrackers(frame);

        // 消失カウントの更新と不要トラッカーの削除
        pTracking->updateMissedCounts(newDetections);
        // 可視化処理
        cv::Mat result = pTracking->drawBoxes(frame);
        //cv::Mat result = pDetector->drawBoundingBoxes(frame, pDetector->detect(frame));
        cv::imshow("Tracking", result);

        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
