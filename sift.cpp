#include "sift.h"
using namespace cv;

sift::sift(Mat img, int intervals)
{
	src_img = img.clone();
	num_intervals = intervals;
	num_octaves = floor(log(min(src_img.cols*2, src_img.rows*2)) / log(2) - 2);

	GenerateLists();
}

sift::sift(const char* filename, int intervals)
{
	src_img = imread(filename);

	num_intervals = intervals;
	num_octaves = floor(log(min(src_img.cols*2, src_img.rows*2)) / log(2) - 2);

	GenerateLists();
}

void sift::GenerateLists()
{
	unsigned int i = 0;

	g_list.resize(num_octaves);
	for (i = 0; i<num_octaves; i++)
		g_list[i].resize(num_intervals + 3);

	// 储存差分高斯金字塔的数列
	dog_list.resize(num_octaves);
	for (i = 0; i<num_octaves; i++)
		dog_list[i].resize(num_intervals + 2);

	// 生成数列去储存模糊系数
	abs_sigma = new double[num_intervals + 3];
	magnitude.resize(num_octaves);
	orientation.resize(num_octaves);
}

void sift::dosift()
{
	BuildScaleSpace();
	DetectExtrema();
	AssignOrientations();
	GenerateDescriptors();
}