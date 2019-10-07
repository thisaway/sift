#pragma once

#include <opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "descriptor.h"
#include "keypoint.h"

//cv::

#define  M_PI 3.1415926535897932384
#define NUM_BINS 36
#define CURVATURE_THRESHOLD 10.0
#define SIFT_DESCR_WIDTH 4
#define SIFT_DESCR_HIST_BINS 8
#define SIFT_DESCR_MAG_THR 0.2
#define SIFT_INT_DESCR_FCTR 512.0

class sift
{
public:
	sift(cv::Mat img, int intervals);
	sift(const char* filename, int intervals);
	~sift() {};

	void dosift();
	vector<descriptor> get_des() { return key_descs; }
	cv::Mat get_img() { return src_img; }



private:
	void GenerateLists();
	void BuildScaleSpace();
	void DetectExtrema();
	void AssignOrientations();
	void GenerateDescriptors();

private:
	cv::Mat src_img;
	unsigned int num_intervals;
	unsigned int num_octaves;
	unsigned int num_keypoints;

	vector<vector<cv::Mat>> g_list;
	vector<vector<cv::Mat>> dog_list;
	double* abs_sigma;

	vector<vector<cv::Mat>> magnitude;
	vector<vector<cv::Mat>> orientation;

	vector<keypoint> key_points;
	vector<descriptor> key_descs;

};