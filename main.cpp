#include "utils.h"
#include "readData.h"
#include "opencv2/opencv.hpp"

#include "Condensation.h"

//功能
//本demo主要展示了particle filter对于非线性系统的滤波过程。
//欢迎关注我们的公众号：[3D视觉工坊](https://mp.weixin.qq.com/s/xyGndcupuK1Zzmv1AJA5CQ)

//「3D视觉工坊」技术交流群已经成立，目前大约有12000人，方向主要涉及3D视觉、CV&深度学习、SLAM、三维重建、点云后处理、自动驾驶、CV入门、三维测量、VR/AR、3D人脸识别、医疗影像、缺陷检测、行人重识别、目标跟踪、视觉产品落
//地、视觉竞赛、车牌识别、硬件选型、学术交流、求职交流、ORB-SLAM系列源码交流、深度估计等。工坊致力于干货输出，不做搬运工，为计算机视觉领域贡献自己的力量！欢迎大家一起交流成长~

//添加小助手微信：*CV_LAB*，备注学校/公司+姓名+研究方向即可加入工坊一起学习进步。

//3D视觉研习社QQ群：574432628

//particle filter
int main(int argc,char **argv)
{
    //std::ofstream cmdoutfile("../output.txt");
    //std::cout.rdbuf(cmdoutfile.rdbuf());

    std::string file_data = argv[1];
    std::vector<cv::Point2f> dataPnts;
    PF::readData(file_data,dataPnts);

    for(size_t i = 0; i < dataPnts.size(); ++i)
    {
        std::cout<<"x: "<<dataPnts[i].x<<" "
                 <<"y: "<<dataPnts[i].y<<" "<<std::endl;

    }

    std::vector<cv::Point2f> vcurrent_pos,vparticle_pos;

    int DP = 2;
    int nParticles = 200;
    float xRange = 800.0f;
    float flocking = 0.9f;
    float minRange[] = {0.0f,0.0f};
    float maxRange[] = {xRange,xRange};
    cv::Mat_<float> LB(1,DP,minRange);
    cv::Mat_<float> UB(1,DP,maxRange);
    cv::Mat_<float> measurement(1,DP);
    cv::Mat_<float> dyna(cv::Mat_<float>::eye(2,2));

    ConDensation condens(DP,nParticles);

    cv::Mat img((int)xRange,(int)xRange,CV_8UC3);
    cv::namedWindow("particle filter");

    condens.initSampleSet(LB,UB,dyna);

    for(size_t i = 0; i < dataPnts.size(); ++i)
    {
        cv::Point2f cur;
        cv::waitKey(30);
        cur.x = dataPnts[i].x * 100; //由于原始数据单位为m,这里乘以100，换算到cm来观测，比较直观
        cur.y = dataPnts[i].y * 100; //由于原始数据单位为m,这里乘以100，换算到cm来观测，比较直观

        measurement(0) = float(cur.x);
        measurement(1) = float(cur.y);

        cv::Point2f measPt(cur.x,cur.y);
        vcurrent_pos.push_back(measPt);

        //Clear screen
        img = cv::Scalar::all(60);

        //Update and get prediction:
        const cv::Mat_<float> &pred = condens.correct(measurement);

        cv::Point2f statePt(pred(0),pred(1));
        vparticle_pos.push_back(statePt);

        for(int s = 0; s < condens.sampleCount();s++)
        {
            cv::Point2f partPt(condens.sample(s,0), condens.sample(s,1));
            drawCross(img,partPt,cv::Scalar(255,90,(int)(s*255.0)/(float)condens.sampleCount()),2);
        }

        for(size_t i = 0; i < vcurrent_pos.size() - 1; i++)
        {
            cv::line(img,vcurrent_pos[i],vcurrent_pos[i + 1],cv::Scalar(255,255,0),1);
        }

        for(size_t i = 0; i < vparticle_pos.size() - 1; i++)
        {
            cv::line(img,vparticle_pos[i],vparticle_pos[i + 1],cv::Scalar(0,255,0),1);
        }

        drawCross(img,statePt,cv::Scalar(255,255,255),5);
        drawCross(img,measPt,cv::Scalar(0,0,255),5);
        cv::Mat imgshow = cv::Mat::zeros(640,640,CV_8UC3);
        cv::resize(img,imgshow,cv::Size(800,800));
        cv::imshow("particle filter",imgshow);
    }
    return 0;
}

