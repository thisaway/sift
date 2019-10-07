#include "sift.h"
using namespace cv;

static void img2float(Mat img1, Mat& img2);

void sift::BuildScaleSpace()
{
	cout << "正在生成尺度空间……" << endl;

	Mat imgGray(src_img.rows, src_img.cols, CV_32FC1);
	Mat imgTemp(src_img.rows, src_img.cols, CV_8UC1);

	if (src_img.channels() == 3)
	{
		cvtColor(src_img, imgTemp, CV_BGR2GRAY);
	}
	else
	{
		imgTemp = src_img;
	}

	imgTemp.convertTo(imgTemp, CV_32FC1);
	
	img2float(imgTemp, imgGray);

	float init_sigma = 1.6;
	float k = pow(2.0, 1.0 / num_intervals);
	abs_sigma[1] = init_sigma*sqrt(k*k - 1);
	//确定高斯模糊系数
	abs_sigma[0] = init_sigma;
	for (int i = 2; i < num_intervals + 3; i++)
	{
		abs_sigma[i] = abs_sigma[i - 1] * k;
	}

	//生成高斯金字塔
	g_list[0][0].create(Size(imgGray.cols * 2, imgGray.rows * 2), CV_32FC1);
	pyrUp(imgGray, g_list[0][0], Size(imgGray.cols * 2, imgGray.rows * 2));
	GaussianBlur(g_list[0][0], g_list[0][0], Size(0, 0), sqrt(pow(init_sigma, 2.0) - pow(0.5, 2.0) * 4));
	for (unsigned int i = 0; i < num_octaves; i++)
	{
		Size currentSize = g_list[i][0].size();
		//g_list[i][0].convertTo(g_list[i][0], CV_8U);
	/*	imshow("每一层第一张图片", g_list[i][0]);*/

		for (unsigned int j = 1; j < num_intervals + 3; j++)
		{
			g_list[i][j].create(currentSize, CV_32FC1);
			dog_list[i][j - 1].create(currentSize, CV_32FC1);

			GaussianBlur(g_list[i][j - 1], g_list[i][j], Size(0, 0), abs_sigma[j]);
			dog_list[i][j - 1] = g_list[i][j] - g_list[i][j - 1];

		}
		if (i < num_octaves - 1)
		{
			currentSize.width /= 2;
			currentSize.height /= 2;

			pyrDown(g_list[i][num_intervals], g_list[i + 1][0], currentSize);
		}
		//cout << dog_list[i][0].at<float>(0,0) << endl;
	/*	for (int i = 0; i < dog_list[i][0].cols; i++)
		{
			for (int j = 0; j < dog_list[i][0].rows; j++)
			{
				cout << dog_list[i][0].at<float>(i, j) << " ";
			}
			cout << endl;
		}*/
		/*namedWindow("每一层的第一张dog", 0);
		imshow("每一层的第一张dog", dog_list[i][0]);
		waitKey(0);*/
	}

}
static void img2float(Mat img1, Mat& img2)
{
	for (int c = 0; c < img1.cols; c++)
	{
		for (int r = 0; r < img1.rows; r++)
		{
			img2.at<float>(c, r) = img1.at<float>(c, r) / 255.0;
		}
	}
}