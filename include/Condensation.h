#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>

class ConDensation
{
public:

	//
	//! All Matrices interfaced here are expected to have dp cols, and are float.
	//
	ConDensation(int dp, int numSamples, float flocking = 0.9f);

	//! Reset. call at least once before correct()
	void initSampleSet(const  cv::Mat & lowerBound, const  cv::Mat & upperBound, const  cv::Mat & dyna = cv::Mat());

	//! Update the state and return prediction.
	const cv::Mat & correct(const  cv::Mat & measurement);

	//! Access single samples(read only).
	int   sampleCount()       { return samples.rows; }
	float sample(int j, int i) { return samples(j, i); }

private:

	int DP;                      //! Sample dimension
	int numSamples;              //! Number of the Samples                 
	float flocking;              //! flocking/congealing factor
	cv::Mat_<float> range;           //! Scaling factor for correction, the upper bound from the orig. samples
	cv::Mat_<float> dynamMatr;       //! Matrix of the linear Dynamics system  
	cv::Mat_<float> samples;         //! Arr of the Sample Vectors             
	cv::Mat_<float> newSamples;      //! Temporary array of the Sample Vectors 
	cv::Mat_<float> confidence;      //! Confidence for each Sample            
	cv::Mat_<float> cumulative;      //! Cumulative confidence                 
	cv::Mat_<float> randomSample;    //! RandomVector to update sample set     
	cv::Mat_<float> state;           //! Predicted state vector
	cv::Mat_<float> mean;            //! Internal mean vector
	cv::Mat_<float> measure;         //! Cached measurement vector

	struct Rand                  //! CvRandState replacement
	{
		cv::RNG r;
		float lo, hi;
		Rand(float l = 0.0f, float h = 1.0f) { set(cv::getTickCount(), l, h); }
		void set(int64 s = 0, float l = 0.0f, float h = 1.0f) { r.state = s; lo = l; hi = h; }
		float uni() { return r.uniform(lo, hi); }
	};
	std::vector<Rand> rng;       //! One rng for each dimension.


	void updateByTime();
};


