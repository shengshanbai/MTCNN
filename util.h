#ifndef UTIL_H
#define UTIL_H
#include <opencv2/opencv.hpp>
int createDir(const   char   *sPathName);
float IoU(cv::Rect& rect1,cv::Rect& rect2);
#endif // UTIL_H
