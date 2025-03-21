#include "ObjectTracker.h"

ObjectTracker::ObjectTracker()
{
}

ObjectTracker::~ObjectTracker()
{
}
/**************************************************************************//**
@関数名         Tracking::computeIoU
@概要           2つの矩形のIoU（Intersection over Union）を計算
@パラメータ[in] a: 矩形1
				b: 突形2
@戻り値         IoU値（0.0～1.0）
@詳細           矩形の重なり度合いを測定するために使用
******************************************************************************/
double ObjectTracker::computeIoU(const cv::Rect& a, const cv::Rect& b) {
	cv::Rect intersection = a & b;
	double intersectionArea = intersection.area();
	double unionArea = a.area() + b.area() - intersectionArea;
	return unionArea == 0 ? 0.0 : intersectionArea / unionArea;
}
/**************************************************************************//**
@関数名           Tracking::detectNewCars
@概要             新しい車両の検出とトラッカーの追加
@パラメータ[in]   frame: 現在のフレーム
                  newDetections: 新しい検出結果
@パラメータ[out]  なし
@戻り値           なし
@詳細             新しい車両を検出し、まだトラッカーがない場合は新しいトラッカーを追加
******************************************************************************/
void ObjectTracker::detectNewCars(const cv::Mat& frame, const vector<cv::Rect>& newDetections)
{
	vector<dataTracker> newTrackers = dataTrackers;
	for (const auto& detection : newDetections)
	{		
		bool isNew = true;
		//cv::Rect detectionRect = cv::boundingRect(detection);
		// IoUで重複を確認
		for (const auto& tracker : dataTrackers) 
		{
			if (computeIoU(detection, tracker.trackBox) > MIN_IOU)
			{
				isNew = false;
				break;
			}
		}
		if (isNew) {
			auto newTracker = cv::TrackerCSRT::create(); // CSRTを使用
			newTracker->init(frame, detection);
			newTrackers.push_back({ detection, 0, newTracker });
			g_carCount++;
		}

	}
	dataTrackers = move(newTrackers);
}
/**************************************************************************//**
@関数名           Tracking::updateTrackers
@概要             トラッカーを更新して新しいフレームでの位置を取得
@パラメータ[in]   frame: 現在のフレーム
@パラメータ[out]  なし
@戻り値           なし
@詳細             現在のフレームでトラッカーの位置を更新する
******************************************************************************/
void ObjectTracker::updateTrackers(const cv::Mat& frame)
{
	vector<dataTracker> newTrackers;
	for (size_t i = 0; i < dataTrackers.size(); ++i) {
		cv::Rect box;
		if (dataTrackers[i].tracker->update(frame, box)) {
			newTrackers.push_back({ box, 0, dataTrackers[i].tracker });
			
		}
	}
	dataTrackers = move(newTrackers);
}
/**************************************************************************//**
@関数名           Tracking::updateMissedCounts
@概要             トラッカーが失敗した回数を更新する
@パラメータ[in]   newDetections: 新たに検出された物体のリスト（バウンディングボックス）
@パラメータ[out]  なし
@戻り値           なし
@詳細             新しい検出結果と現在のトラッカーを比較し、失敗回数を更新する
******************************************************************************/
void ObjectTracker::updateMissedCounts(const vector<cv::Rect>& newDetections)
{
	for (size_t i = 0; i < dataTrackers.size(); ++i) {
		bool isMatched = false;
		for (const auto& detection : newDetections) {
			//cv::Rect detectionRect = cv::boundingRect(detection);
			if (computeIoU(detection, dataTrackers[i].trackBox) > MATCH_THRESHOLD) {
				isMatched = true;
				break;
			}
		}
		dataTrackers[i].trackMissedCounter = isMatched ? 0 : dataTrackers[i].trackMissedCounter + 1;
	}
}
/**************************************************************************//**
@関数名           Tracking::drawBoxes
@概要             物体のトラッキング結果を描画する
@パラメータ[in]   image: 入力画像
@パラメータ[out]  なし
@戻り値           トラッキングされた物体のバウンディングボックスを描画した画像（Mat）
@詳細             物体の位置にバウンディングボックスを描画し、車両数を表示する
******************************************************************************/
cv::Mat ObjectTracker::drawBoxes(const cv::Mat& image)
{
	cv::Mat result = image.clone();

	for (size_t i = 0; i < dataTrackers.size(); ++i) {
		if (dataTrackers[i].trackMissedCounter < MAX_MISSED_FRAMES) {
			int x = dataTrackers[i].trackBox.x;
			int y = dataTrackers[i].trackBox.y - 5;
			// バウンディングボックスを描画
			rectangle(result, dataTrackers[i].trackBox, cv::Scalar(0, 255, 0), 2);
			// トラッカーの幅に合わせてテキストサイズを調整
			double fontScale = std::min(dataTrackers[i].trackBox.width, dataTrackers[i].trackBox.height) / 340.0;
			fontScale = std::max(fontScale, 0.5); // 文字が小さくなりすぎないように0.5以上を維持
			// 車両数を表示
			putText(result, "Car: " + to_string(i + 1), cv::Point(x, y),
				cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 255, 0), 2);
		}
	}
	// 合計車両数を表示
	//putText(result, "Total Car: " + to_string(m_trackedBoxes.size()), cv::Point(10, 30),
	//    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
	putText(result, "Total Car: " + to_string(g_carCount), cv::Point(10, 30),
		cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
	return result;
}