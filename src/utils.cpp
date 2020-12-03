
#include "utils.h"


void drawCross(cv::Mat & img, cv::Point center, cv::Scalar color, int d) {
    cv::line(img, cv::Point(center.x - d, center.y - d),
             cv::Point(center.x + d, center.y + d), color, 2, 16, 0);
    cv::line(img, cv::Point(center.x + d, center.y - d),
             cv::Point(center.x - d, center.y + d), color, 2, 16, 0);
}
