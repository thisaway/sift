#include "sift.h"
using namespace cv;

#define interp_hist_peak( l, c, r ) ( 0.5 * ((l)-(r)) / ((l) - 2.0*(c) + (r)) )


static void compute_mag_ori(vector<vector<Mat>> g_list,
	vector<vector<Mat>>&  magnitude, vector<vector<Mat>>& orientation, int num_octaves, int num_intervals);

static void smooth_hist(double* hist_orient, int n);

static double* generate_hist(vector<vector<Mat>> magnitude, vector<vector<Mat>> orientation, keypoint kp);

static void interp_hist(double* hist_orient, keypoint& kp);


void sift::AssignOrientations()
{
	cout << "分配方向……" << endl;

	compute_mag_ori(g_list, magnitude, orientation, num_octaves, num_intervals);

	for (unsigned int kp_index = 0; kp_index < key_points.size(); kp_index++)
	{
		double* hist_orient = generate_hist(magnitude, orientation, key_points[kp_index]);

		interp_hist(hist_orient, key_points[kp_index]);

	}
}


//建立方向和梯度图
static void compute_mag_ori(vector<vector<Mat>> g_list,
	vector<vector<Mat>>&  magnitude, vector<vector<Mat>>& orientation, int num_octaves, int num_intervals)
{
	for (unsigned int i = 0; i < num_octaves; i++)
	{
		magnitude[i].resize(num_intervals+3);
		orientation[i].resize(num_intervals+3);

		for (unsigned int j = 0; j < num_intervals + 3; j++)
		{
			magnitude[i][j] = Mat::zeros(g_list[i][j].size(), CV_32FC1);
			orientation[i][j] = Mat::zeros(g_list[i][j].size(), CV_32FC1);

			for (unsigned int xi = 1; xi < g_list[i][j].cols - 1; xi++)
			{
				for (unsigned int yi = 1; yi < g_list[i][j].rows - 1; yi++)
				{
					//cout << m_gList[i][j].type() << endl;
					double dx = g_list[i][j].at<float>(yi, xi -1) - g_list[i][j].at<float>(yi, xi + 1);
					double dy = g_list[i][j].at<float>(yi + 1, xi) - g_list[i][j].at<float>(yi - 1, xi);

					magnitude[i][j].at<float>(yi, xi) = sqrt(dx*dx + dy*dy);

					double ori = atan2(dy, dx);
					orientation[i][j].at<float>(yi, xi) = ori;
				}
			}
		}
	}
}

//生成直方图
static double* generate_hist(vector<vector<Mat>> magnitude, vector<vector<Mat>> orientation,keypoint kp)
{
	double* hist_orient = new double[NUM_BINS];

	int width = magnitude[kp.octave][kp.interval].cols;
	int height = magnitude[kp.octave][kp.interval].rows;
	/*Mat imgWeight(Size(width, height), CV_32FC1);

	GaussianBlur(magnitude[kp.octave][kp.interval], imgWeight, Size(0, 0), 2 * pow(1.5*kp.scl_octv, 2.0));*/

	int hfsz = round(4.5*kp.scl_octv);

	for (unsigned int k = 0; k < NUM_BINS; k++)
		hist_orient[k] = 0.0;

	for (int kk = -hfsz; kk <= hfsz; kk++)
	{
		for (int tt = -hfsz; tt <= hfsz; tt++)
		{
			if (kp.xi + kk < 0 || kp.xi + kk >= width || kp.yi + tt < 0 || kp.yi + tt >= height)
				continue;

			double sampleOrient = orientation[kp.octave][kp.interval].at<float>(kp.yi + tt, kp.xi + kk);

			if (sampleOrient <= -M_PI || sampleOrient > M_PI)
				cout << "错误的方向：" << sampleOrient << endl;

			sampleOrient += M_PI;
			unsigned int sampleOrientDegrees = sampleOrient * 180 / M_PI;
			double w = exp(-(kk*kk + tt*tt) / (2 * pow(1.5*kp.scl_octv, 2.0)));
			hist_orient[(int)sampleOrientDegrees / (360 / NUM_BINS)] += magnitude[kp.octave][kp.interval].at<float>(kp.yi + tt, kp.xi + kk)*w;

		}
	}
	for (unsigned int step = 0; step < 2; step++)
	{
		smooth_hist(hist_orient, NUM_BINS);
	}

	return hist_orient;
}

static void smooth_hist(double* hist_orient, int n)
{
	double prev, temp, h0 = hist_orient[0];
	prev = hist_orient[n - 1];

	for (unsigned int i = 0; i < n; i++)
	{
		temp = hist_orient[i];
		hist_orient[i] = 0.25*prev + 0.5*hist_orient[i] +
			0.25*((i + 1 == n) ? h0 : hist_orient[i]);
		prev = temp;
	}
}

static void interp_hist(double* hist_orient, keypoint& kp)
{
	vector<double> orien, mag;

	double max_peak = hist_orient[0];
	for (int k = 1; k < NUM_BINS; k++)
	{
		if (hist_orient[k] > max_peak)
		{
			max_peak = hist_orient[k];
		}
	}

	double bin;
	int l, r, k;
	for (int k = 0; k < NUM_BINS; k++)
	{
		l = (k == 0) ? NUM_BINS - 1 : k - 1;
		r = (k + 1) % 10;

		if (hist_orient[k] > hist_orient[l] && hist_orient[k] > hist_orient[r] && hist_orient[k] > 0.8*max_peak)
		{
			bin = k + interp_hist_peak(hist_orient[l], hist_orient[k], hist_orient[r]);
			bin = (bin < 0) ? NUM_BINS + bin : (bin >= NUM_BINS) ? bin - NUM_BINS : bin;
			bin = bin*(2 * M_PI / NUM_BINS);
			bin -= M_PI;
			orien.push_back(bin);
			mag.push_back(hist_orient[k]);
		}
	}
	kp.orien = orien;
	kp.mag = mag;
}