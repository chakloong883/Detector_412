#include "image_process.h"

void ImageProcess::cropImage(cv::Mat& image, std::vector<ImagePos>& imagePos, int& cropHeight, int& cropWidth, float& overLap) {
	int rowBias = std::round(cropHeight - (cropHeight * overLap));
	int colBias = std::round(cropWidth - (cropWidth * overLap));

    int rows = image.rows;
    int cols = image.cols;
    int total = (rows + cropHeight - 1) / cropWidth * (cols + cropWidth - 1) / cropWidth;
    int frameNumber = 0;
    bool isLastRow = false;
    bool isLastCol = false;

    int count = 0;
    int count1 = 0;
    int y = 0;
    while (true) {
        int nextRowBiasAdd = y + rowBias + cropHeight > rows ? rows - cropHeight - y : rowBias;
        isLastRow = nextRowBiasAdd == 0;
        int x = 0;
        while (true) {
            count += 1;
            int nextColBiasAdd = x + colBias + cropWidth > cols ? cols - cropWidth - x : colBias;
            isLastCol = nextColBiasAdd == 0;
            cv::Mat cropped = cv::Mat::zeros(cropHeight, cropWidth, image.type());

            // Define the region of interest in the original image, ±ÜÃâ²Ã¼ô³ö½ç
            int width = std::min(cropWidth, cols - x);
            int height = std::min(cropHeight, rows - y);
            cv::Rect roi(x, y, width, height);

            // Copy the region of interest to the cropped image
            image(roi).copyTo(cropped(cv::Rect(0, 0, width, height)));


            imagePos.push_back({
                y,
                x,
                isLastRow && isLastCol
                });

            x += nextColBiasAdd;
            if (isLastCol) {
                break;
            }
        }
        y += nextRowBiasAdd;
        if (isLastRow) {
            break;
        }
    }

}
