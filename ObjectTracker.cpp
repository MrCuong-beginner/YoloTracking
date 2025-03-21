#include "ObjectTracker.h"

ObjectTracker::ObjectTracker()
{
}

ObjectTracker::~ObjectTracker()
{
}
/**************************************************************************//**
@�֐���         Tracking::computeIoU
@�T�v           2�̋�`��IoU�iIntersection over Union�j���v�Z
@�p�����[�^[in] a: ��`1
				b: �ˌ`2
@�߂�l         IoU�l�i0.0�`1.0�j
@�ڍ�           ��`�̏d�Ȃ�x�����𑪒肷�邽�߂Ɏg�p
******************************************************************************/
double ObjectTracker::computeIoU(const cv::Rect& a, const cv::Rect& b) {
	cv::Rect intersection = a & b;
	double intersectionArea = intersection.area();
	double unionArea = a.area() + b.area() - intersectionArea;
	return unionArea == 0 ? 0.0 : intersectionArea / unionArea;
}
/**************************************************************************//**
@�֐���           Tracking::detectNewCars
@�T�v             �V�����ԗ��̌��o�ƃg���b�J�[�̒ǉ�
@�p�����[�^[in]   frame: ���݂̃t���[��
                  newDetections: �V�������o����
@�p�����[�^[out]  �Ȃ�
@�߂�l           �Ȃ�
@�ڍ�             �V�����ԗ������o���A�܂��g���b�J�[���Ȃ��ꍇ�͐V�����g���b�J�[��ǉ�
******************************************************************************/
void ObjectTracker::detectNewCars(const cv::Mat& frame, const vector<cv::Rect>& newDetections)
{
	vector<dataTracker> newTrackers = dataTrackers;
	for (const auto& detection : newDetections)
	{		
		bool isNew = true;
		//cv::Rect detectionRect = cv::boundingRect(detection);
		// IoU�ŏd�����m�F
		for (const auto& tracker : dataTrackers) 
		{
			if (computeIoU(detection, tracker.trackBox) > MIN_IOU)
			{
				isNew = false;
				break;
			}
		}
		if (isNew) {
			auto newTracker = cv::TrackerCSRT::create(); // CSRT���g�p
			newTracker->init(frame, detection);
			newTrackers.push_back({ detection, 0, newTracker });
			g_carCount++;
		}

	}
	dataTrackers = move(newTrackers);
}
/**************************************************************************//**
@�֐���           Tracking::updateTrackers
@�T�v             �g���b�J�[���X�V���ĐV�����t���[���ł̈ʒu���擾
@�p�����[�^[in]   frame: ���݂̃t���[��
@�p�����[�^[out]  �Ȃ�
@�߂�l           �Ȃ�
@�ڍ�             ���݂̃t���[���Ńg���b�J�[�̈ʒu���X�V����
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
@�֐���           Tracking::updateMissedCounts
@�T�v             �g���b�J�[�����s�����񐔂��X�V����
@�p�����[�^[in]   newDetections: �V���Ɍ��o���ꂽ���̂̃��X�g�i�o�E���f�B���O�{�b�N�X�j
@�p�����[�^[out]  �Ȃ�
@�߂�l           �Ȃ�
@�ڍ�             �V�������o���ʂƌ��݂̃g���b�J�[���r���A���s�񐔂��X�V����
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
@�֐���           Tracking::drawBoxes
@�T�v             ���̂̃g���b�L���O���ʂ�`�悷��
@�p�����[�^[in]   image: ���͉摜
@�p�����[�^[out]  �Ȃ�
@�߂�l           �g���b�L���O���ꂽ���̂̃o�E���f�B���O�{�b�N�X��`�悵���摜�iMat�j
@�ڍ�             ���̂̈ʒu�Ƀo�E���f�B���O�{�b�N�X��`�悵�A�ԗ�����\������
******************************************************************************/
cv::Mat ObjectTracker::drawBoxes(const cv::Mat& image)
{
	cv::Mat result = image.clone();

	for (size_t i = 0; i < dataTrackers.size(); ++i) {
		if (dataTrackers[i].trackMissedCounter < MAX_MISSED_FRAMES) {
			int x = dataTrackers[i].trackBox.x;
			int y = dataTrackers[i].trackBox.y - 5;
			// �o�E���f�B���O�{�b�N�X��`��
			rectangle(result, dataTrackers[i].trackBox, cv::Scalar(0, 255, 0), 2);
			// �g���b�J�[�̕��ɍ��킹�ăe�L�X�g�T�C�Y�𒲐�
			double fontScale = std::min(dataTrackers[i].trackBox.width, dataTrackers[i].trackBox.height) / 340.0;
			fontScale = std::max(fontScale, 0.5); // �������������Ȃ肷���Ȃ��悤��0.5�ȏ���ێ�
			// �ԗ�����\��
			putText(result, "Car: " + to_string(i + 1), cv::Point(x, y),
				cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 255, 0), 2);
		}
	}
	// ���v�ԗ�����\��
	//putText(result, "Total Car: " + to_string(m_trackedBoxes.size()), cv::Point(10, 30),
	//    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
	putText(result, "Total Car: " + to_string(g_carCount), cv::Point(10, 30),
		cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
	return result;
}