

#include "readData.h"

namespace PF
{
    void readData(std::string file_path,std::vector<cv::Point2f> &dataPnts)
    {
        std::ifstream file;
        file.open(file_path.c_str());
        if (!file.is_open())
        {
            std::cout << "open file failed !" << std::endl;
            return;
        }
        std::string s;
        cv::Point2f pnt;
        while(!file.eof())
        {
            getline(file,s);
            stringstream ss;
            ss<<s;
            ss>>pnt.x;
            ss>>pnt.y;
            dataPnts.push_back(pnt);
        }
        file.close();
    }



}
