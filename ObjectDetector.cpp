
/**************************************************************************//**
@クラス名         Detector
@概要             YOLOモデルを使用した物体検出クラス
@詳細             指定されたONNX形式のYOLOモデルを使用して画像中の物体を検出する。
                  検出結果は信頼度閾値とNMS処理を経て返される。
******************************************************************************/
#include "ObjectDetector.h"


/**************************************************************************//**
@関数名           Detector::Detector
@概要             ONNX形式のYOLOモデルを読み込んで、DNNネットワークを初期化する
@パラメータ[in]   modelPath: モデルファイルのパス（ONNX形式）
                  classFile: クラス名が記載されたファイルのパス
@パラメータ[out]  なし
@戻り値           なし
@詳細             モデルをONNX形式で読み込み、クラス名をファイルからロードする。
                  また、バックエンドやターゲットの設定（コメントアウトされている）を行うことができる。
******************************************************************************/
ObjectDetector::ObjectDetector(const string& modelPath, const string& classFile)
{
    // ONNX形式のモデルを読み込み、DNNネットワークを作成
    net = cv::dnn::readNetFromONNX(modelPath);
    // Cấu hình backend và target (GPU bằng OpenCL, hoặc CPU)
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);  // Hoặc DNN_BACKEND_OPENCL
    net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);  // Hoặc DNN_TARGET_CPU

    // クラス名をファイルからロード
    classList = loadClasses(classFile);
}

/**************************************************************************//**
@関数名           Detector::loadClasses
@概要             クラス名をファイルからロードする
@パラメータ[in]   path: クラス名が記載されたファイルのパス
@パラメータ[out]  なし
@戻り値           クラス名のリスト（vector<string>）
@詳細             ファイルから1行ずつ読み込んで、クラス名を格納する
******************************************************************************/
vector<string> ObjectDetector::loadClasses(const string& path)
{
    vector<string> classList;
    ifstream file(path);
    string line;
    while (getline(file, line))
    {
        classList.push_back(line);
    }
    return classList;
}


/**************************************************************************//**
@関数名           Detector::detect
@概要             画像から物体を検出する
@パラメータ[in]   image: 入力画像
@パラメータ[out]  なし
@戻り値           検出された物体の予測結果（vector<Predicted>）
@詳細             YOLOモデルを使用して物体を検出し、信頼度が指定された閾値を超える物体を選択する
******************************************************************************/
vector<Predicted> ObjectDetector::boundingBoxDetector(const cv::Mat& image)
{
    vector<Predicted> predictions;
    vector<cv::Mat> DataOuts = getOutputs(image);
    float xFactor = static_cast<float>(image.cols) / INPUT_WIDTH;
    float yFactor = static_cast<float>(image.rows) / INPUT_HEIGHT;
    const int rows = DataOuts[0].size[1];
    const int cols = DataOuts[0].size[2];
    float* data = (float*)DataOuts[0].data;

    predictions.reserve(rows); // メモリの使用を最適化するため、予め容量を予約しておく
    for (int i = 0; i < rows; i++, data += cols)
    {
        float confidence = data[4];
        if (confidence > CONFIDENCE_THRESHOLD)
        {
            float cx = data[0] * xFactor;
            float cy = data[1] * yFactor;
            float w = data[2] * xFactor;
            float h = data[3] * yFactor;
            int classId = max_element(data + 5, data + cols) - (data + 5);
            predictions.push_back({ classId, confidence, cv::Rect(cx - w / 2, cy - h / 2, w , h) });
        }
    }

    return applyNMS(predictions);

}

/**************************************************************************//**
@関数名           Detector::applyNMS
@概要             非最大抑制（NMS）を適用し、重複する検出を削除する
@パラメータ[in]   predictions: 検出された物体のリスト（予測結果）
@パラメータ[out]  なし
@戻り値           NMSが適用された後の物体のリスト（vector<Predicted>）
@詳細             重複する物体検出を削除し、最も信頼度の高い検出を残す
******************************************************************************/
vector<Predicted> ObjectDetector::applyNMS(const vector<Predicted>& predictions)
{
    if (predictions.empty()) return {};
    vector<float> scores;
    vector<cv::Rect> boxes;
    vector<int> nms_result;
    vector<Predicted> finalPredictions;

    scores.reserve(predictions.size());
    boxes.reserve(predictions.size());

    for (const auto& object : predictions)
    {
        scores.push_back(object.confidence);
        boxes.push_back(object.box);
    }
    cv::dnn::NMSBoxes(boxes, scores, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);

    for (auto idx : nms_result)
    {
        finalPredictions.push_back(predictions[idx]);
    }
    return finalPredictions;
}

/**************************************************************************//**
@関数名           Detector::drawBoundingBoxes
@概要             画像に物体の検出結果（バウンディングボックス）を描画する
@パラメータ[in]   image: 入力画像
                  detected: 検出された物体のリスト（予測結果）
@パラメータ[out]  なし
@戻り値           物体のバウンディングボックスが描画された画像（Mat）
@詳細             画像にバウンディングボックスとラベルを描画する
******************************************************************************/
cv::Mat ObjectDetector::drawBoundingBoxes(const cv::Mat& image, const vector<Predicted>& detected)
{
    int baseLine;

    cv::Mat result = image.clone();
    for (const auto& idx : detected)
    {
        //string label = classList[idx.classId] + ":" + to_string(idx.confidence);
        string label = classList[idx.classId];
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int x = idx.box.x;
        int y = idx.box.y - baseLine;
        rectangle(result, idx.box, cv::Scalar(0, 255, 0), 2);
        putText(result, label, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }
    return result;
}
/**************************************************************************//**
@関数名           Detector::getOutputs
@概要             画像から物体を検出するための出力を取得する
@パラメータ[in]   image: 入力画像
@パラメータ[out]  なし
@戻り値           モデルの出力結果（vector<Mat>）
@詳細             入力画像に対してYOLOモデルを実行し、出力を取得する
******************************************************************************/
vector<cv::Mat> ObjectDetector::getOutputs(const cv::Mat& image)
{
    cv::Mat blob;
    vector<cv::Mat> outs;
    cv::dnn::blobFromImage(image, blob, SCALE, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);
    net.forward(outs, net.getUnconnectedOutLayersNames());
    return outs;
}


/**************************************************************************//**
@関数名           Detector::getCars
@概要             画像内の車両を検出し、その領域を返す
@パラメータ[in]   image: 入力画像
@パラメータ[out]  なし
@戻り値           車両の矩形領域のリスト（vector<cv::Mat>）
@詳細             画像内で車両を検出し、その領域を返します。車両はクラスIDが2であると仮定しており、領域が閾値（AREAS_THRESHOLD）より大きい場合にのみ検出対象とします。
                  各検出結果の領域面積も表示されます。
******************************************************************************/
vector<cv::Rect> ObjectDetector::getCars(const cv::Mat& image)
{
    vector<cv::Rect> carImages;
    vector<Predicted> predictions = boundingBoxDetector(image);
    for (const auto& idx : predictions)
    {
        if (idx.classId == 2) //car
        {
            carImages.push_back(idx.box);

        }
    }
    return carImages;
}



