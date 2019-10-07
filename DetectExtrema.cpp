#include "sift.h"
using namespace cv;

static bool isExtrema(vector<vector<Mat>> dog_list, int octa, int interv, int xi, int yi);

static bool interLocation(vector<vector<Mat>> dog_list, int octa, int &interv, int& xi, int& yi,
	float contr_thr, int num_intervals,double& p);

static bool isEdge(vector<vector<Mat>> dog_list, int octa, int interv, unsigned int xi, unsigned int yi);

void sift::DetectExtrema()
{
	cout << "检测极值……" << endl;

	const float prelim_contr_thr = 0.5*0.04 / num_intervals;

	unsigned int num = 0;
	int con_num = 0, extre_num = 0, inter_num = 0;

	for (int i = 0; i < num_octaves; i++)
	{
		for (int j = 1; j < num_intervals + 1; j++)
		{
			for (int xi = 5; xi < dog_list[i][j].cols - 5; xi++)
			{
				for (int yi = 5; yi < dog_list[i][j].rows - 5; yi++)
				{
					float currentPixel = dog_list[i][j].at<float>(yi, xi);
					if (abs(currentPixel) < prelim_contr_thr)continue;
					con_num++;
					if (!isExtrema(dog_list, i, j, xi, yi))continue;
					extre_num++;
					int true_xi = xi, true_yi = yi, true_interval = j;
					double p;
					if (!interLocation(dog_list, i, true_interval, true_xi, true_yi, 0.04, num_intervals,p))continue;
					inter_num++;
					if (isEdge(dog_list,i,true_interval,true_xi,true_yi))continue;
					num++;
					key_points.push_back(keypoint(i, true_interval, true_xi, true_yi, abs_sigma[0] * pow(2.0, double(true_interval+p) / num_intervals)));
				}
			}
		}
	}
	num_keypoints = num;
	cout << "con_num:" << con_num << endl;
	cout << "extre_num:" << extre_num << endl;
	cout << "inter_num:" << inter_num << endl;
	cout << "找到" << num << "对关键点" << endl;
}
static bool isExtrema(vector<vector<Mat>> dog_list,int octa,int interv, int xi, int yi)
{
	Mat middle = dog_list[octa][interv];
	Mat up = dog_list[octa][interv + 1];
	Mat down = dog_list[octa][interv - 1];
	float currentPixel = middle.at<float>(yi, xi);


	if (currentPixel > middle.at<float>(yi - 1, xi) &&
		currentPixel > middle.at<float>(yi + 1, xi) &&
		currentPixel > middle.at<float>(yi, xi - 1) &&
		currentPixel > middle.at<float>(yi, xi + 1) &&
		currentPixel > middle.at<float>(yi - 1, xi - 1) &&
		currentPixel > middle.at<float>(yi - 1, xi + 1) &&
		currentPixel > middle.at<float>(yi + 1, xi - 1) &&
		currentPixel > middle.at<float>(yi + 1, xi + 1) &&
		currentPixel > up.at<float>(yi, xi) &&
		currentPixel > up.at<float>(yi - 1, xi) &&
		currentPixel > up.at<float>(yi + 1, xi) &&
		currentPixel > up.at<float>(yi, xi - 1) &&
		currentPixel > up.at<float>(yi, xi + 1) &&
		currentPixel > up.at<float>(yi - 1, xi - 1) &&
		currentPixel > up.at<float>(yi - 1, xi + 1) &&
		currentPixel > up.at<float>(yi + 1, xi - 1) &&
		currentPixel > up.at<float>(yi + 1, xi + 1) &&
		currentPixel > down.at<float>(yi, xi) &&
		currentPixel > down.at<float>(yi - 1, xi) &&
		currentPixel > down.at<float>(yi + 1, xi) &&
		currentPixel > down.at<float>(yi, xi - 1) &&
		currentPixel > down.at<float>(yi, xi + 1) &&
		currentPixel > down.at<float>(yi - 1, xi - 1) &&
		currentPixel > down.at<float>(yi - 1, xi + 1) &&
		currentPixel > down.at<float>(yi + 1, xi - 1) &&
		currentPixel > down.at<float>(yi + 1, xi + 1))
		return true;

	if (currentPixel > middle.at<float>(yi - 1, xi) &&
		currentPixel < middle.at<float>(yi + 1, xi) &&
		currentPixel < middle.at<float>(yi, xi - 1) &&
		currentPixel < middle.at<float>(yi, xi + 1) &&
		currentPixel < middle.at<float>(yi - 1, xi - 1) &&
		currentPixel < middle.at<float>(yi - 1, xi + 1) &&
		currentPixel < middle.at<float>(yi + 1, xi - 1) &&
		currentPixel < middle.at<float>(yi + 1, xi + 1) &&
		currentPixel < up.at<float>(yi, xi) &&
		currentPixel < up.at<float>(yi - 1, xi) &&
		currentPixel < up.at<float>(yi + 1, xi) &&
		currentPixel < up.at<float>(yi, xi - 1) &&
		currentPixel < up.at<float>(yi, xi + 1) &&
		currentPixel < up.at<float>(yi - 1, xi - 1) &&
		currentPixel < up.at<float>(yi - 1, xi + 1) &&
		currentPixel < up.at<float>(yi + 1, xi - 1) &&
		currentPixel < up.at<float>(yi + 1, xi + 1) &&
		currentPixel < down.at<float>(yi, xi) &&
		currentPixel < down.at<float>(yi - 1, xi) &&
		currentPixel < down.at<float>(yi + 1, xi) &&
		currentPixel < down.at<float>(yi, xi - 1) &&
		currentPixel < down.at<float>(yi, xi + 1) &&
		currentPixel < down.at<float>(yi - 1, xi - 1) &&
		currentPixel < down.at<float>(yi - 1, xi + 1) &&
		currentPixel < down.at<float>(yi + 1, xi - 1) &&
		currentPixel < down.at<float>(yi + 1, xi + 1))
		return true;

	return false;
}

static bool interLocation(vector<vector<Mat>> dog_list, int octa, int &interv, int& xi, int& yi,
	float contr_thr, int num_intervals,double& p)
{
	Mat middle = dog_list[octa][interv];
	Mat up = dog_list[octa][interv + 1];
	Mat down = dog_list[octa][interv - 1];

	int i = 1;
	Mat deriv3d(3, 1, CV_32FC1);
	Mat hat;
	while (i <= 5)
	{
		i++;
		deriv3d.at<float>(0, 0) = (middle.at<float>(yi, xi + 1) - middle.at<float>(yi, xi - 1)) / 2.0;
		deriv3d.at<float>(1, 0) = (middle.at<float>(yi + 1, xi) - middle.at<float>(yi - 1, xi)) / 2.0;
		deriv3d.at<float>(2, 0) = (up.at<float>(yi, xi) - down.at<float>(yi, xi)) / 2.0;

		Mat hessian3d(3, 3, CV_32FC1);
		hessian3d.at<float>(0, 0) = middle.at<float>(yi, xi + 1) +
			middle.at<float>(yi, xi - 1) - 2 * middle.at<float>(yi, xi);
		hessian3d.at<float>(0, 1) = (middle.at<float>(yi + 1, xi + 1) + middle.at<float>(yi - 1, xi - 1) -
			middle.at<float>(yi - 1, xi + 1) - middle.at<float>(yi + 1, xi - 1)) / 4.0;
		hessian3d.at<float>(0, 2) = (up.at<float>(yi, xi + 1) + down.at<float>(yi, xi - 1) -
			down.at<float>(yi, xi + 1) - up.at<float>(yi, xi - 1)) / 4.0;
		hessian3d.at<float>(1, 0) = hessian3d.at<float>(0, 1);
		hessian3d.at<float>(1, 1) = middle.at<float>(yi + 1, xi) +
			middle.at<float>(yi - 1, xi) - 2 * middle.at<float>(yi, xi);
		hessian3d.at<float>(1, 2) = (up.at<float>(yi + 1, xi) + down.at<float>(yi - 1, xi) -
			down.at<float>(yi + 1, xi) - up.at<float>(yi - 1, xi)) / 4.0;
		hessian3d.at<float>(2, 0) = hessian3d.at<float>(0, 2);
		hessian3d.at<float>(2, 1) = hessian3d.at<float>(1, 2);
		hessian3d.at<float>(2, 2) = up.at<float>(yi, xi) +
			down.at<float>(yi, xi) - 2 * middle.at<float>(yi, xi);

		hat = -hessian3d.inv()*deriv3d;

		if (abs(hat.at<float>(0, 0)) < 0.5 && abs(hat.at<float>(1, 0)) < 0.5 && abs(hat.at<float>(2, 0)) < 0.5)
			break;

		/*if (hat.at<float>(0, 0) < 0 || hat.at<float>(1, 0) < 0)
		return false;*/

		xi += round(hat.at<float>(0, 0));
		yi += round(hat.at<float>(1, 0));
		interv += round(hat.at<float>(2, 0));

		if (interv<1 || interv >= num_intervals+1 || xi<=5 || yi<=5 || xi>middle.cols - 5 || yi>middle.rows - 5)
			return false;
	}

	if (i >5)return false;

	p = hat.at<float>(2, 0);
	float tmep_a = dog_list[octa][interv].at<float>(yi,xi);
	float contr = tmep_a + 0.5*Mat(deriv3d.t()*hat).at<float>(0, 0);
	if (abs(contr) < contr_thr / num_intervals)return false;

	return true;
}

static bool isEdge(vector<vector<Mat>> dog_list, int octa, int interv, unsigned int xi, unsigned int yi)
{
	Mat middle = dog_list[octa][interv];
	Mat up = dog_list[octa][interv + 1];
	Mat down = dog_list[octa][interv - 1];

	float dxx = middle.at<float>(yi - 1, xi) + middle.at<float>(yi + 1, xi) -
		2.0*middle.at<float>(yi, xi);
	float dyy = middle.at<float>(yi, xi - 1) + middle.at<float>(yi, xi + 1) -
		2.0*middle.at<float>(yi, xi);
	float dxy = (middle.at<float>(yi - 1, xi - 1) + middle.at<float>(yi + 1, xi + 1) -
		middle.at<float>(yi + 1, xi - 1) - middle.at<float>(yi - 1, xi + 1)) / 4.0;

	float trH = dxx + dyy;
	float detH = dxx*dyy - dxy*dxy;
	float curvature_threshold = (CURVATURE_THRESHOLD + 1)*(CURVATURE_THRESHOLD + 1) / float(CURVATURE_THRESHOLD);

	float curvature_ratio = trH*trH / detH;
	if (detH <= 0)
		return true;
	if (curvature_ratio < curvature_threshold)return false;
	return true;

}
