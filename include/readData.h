//
// Created by yong on 2020/11/30.
//

#ifndef SERIALPORT_KALMAN_FILTER_READDATA_H
#define SERIALPORT_KALMAN_FILTER_READDATA_H

#include "utils.h"

namespace PF
{
    void readData(std::string file_path,std::vector<cv::Point2f> &dataPnts);
}


#endif //SERIALPORT_KALMAN_FILTER_READDATA_H
