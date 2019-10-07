#include "sift.h"
using namespace cv;

static vector<vector<vector<double>>> descr_hist(keypoint kp,int ori_index,
	vector<vector<Mat>> magnitude, vector<vector<Mat>> orientation);

static void interp_hist_entry(vector<vector<vector<double>>>& hist, double rbin,
	double cbin, double obin, double mag, int d, int n);

static void hist_to_descr(vector<vector<vector<double>>> hist, int d, int n,
	keypoint kp, vector<descriptor>& key_descs);

static void normalize_descr(vector<double>& feature);


void sift::GenerateDescriptors()
{
	cout << "²úÉúÃèÊö×Ó¡­¡­" << endl;


	for (unsigned int kp_index = 0; kp_index < key_points.size(); kp_index++)
	{
		keypoint kp = key_points[kp_index];
		for (unsigned int ori_index = 0; ori_index < kp.orien.size(); ori_index++)
		{
		  vector<vector<vector<double>>> hist;
		  hist = descr_hist(kp,ori_index, magnitude, orientation);

		  hist_to_descr(hist, SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS, key_points[kp_index], key_descs);
	    }
	}
}

static vector<vector<vector<double>>> descr_hist(keypoint kp, int ori_index,
	vector<vector<Mat>> magnitude, vector<vector<Mat>> orientation)
{
	vector<vector<vector<double>>> hist;
	double PI2 = 2.0 * CV_PI;

	int d = SIFT_DESCR_WIDTH;
	int n = SIFT_DESCR_HIST_BINS;

	hist.resize(d);
	for (unsigned int i = 0; i < d; i++)
	{
		hist[i].resize(d);
		for (unsigned int j = 0; j < d; j++)
		{
			hist[i][j].resize(n);
		}
	}

	//Mat weight_img(magnitude[kp.octave][kp.interval].rows, magnitude[kp.octave][kp.interval].cols,CV_32FC1);
	//GaussianBlur(magnitude[kp.octave][kp.interval], weight_img, Size(0, 0), d*d*0.5);



	double hist_width = 3.0*kp.scl_octv;
	int radius = hist_width*sqrt(2)*(d + 1.0)*0.5 + 0.5;

	double bins_per_rad = n / PI2;

		double cos_t = cos(kp.orien[ori_index]);
		double sin_t = sin(kp.orien[ori_index]);

		for (int i = -radius; i <= radius; i++)
		{
			for (int j = -radius; j <= radius; j++)
			{
				double c_rot = (j*cos_t - i*sin_t) / hist_width;
				double r_rot = (j*sin_t + i*cos_t) / hist_width;
				double rbin = r_rot + d / 2 - 0.5;
				double cbin = c_rot + d / 2 - 0.5;

				if (rbin > -1.0&& rbin<d&& cbin>-1.0&&cbin < d)
				{
					if (kp.yi + i<=0 || kp.xi + j<=0 || kp.yi + i>=magnitude[kp.octave][kp.interval].rows-1 || kp.xi + j>= magnitude[kp.octave][kp.interval].cols-1)continue;
					double grad_mag = magnitude[kp.octave][kp.interval].at<float>(kp.yi + i, kp.xi + j);
					double grad_ori = orientation[kp.octave][kp.interval].at<float>(kp.yi + i, kp.xi + j);

					grad_ori -= kp.orien[ori_index];
					while (grad_ori < 0.0)
						grad_ori += PI2;
					while (grad_ori >= PI2)
						grad_ori -= PI2;

					double obin = grad_ori * bins_per_rad;
					double w = exp(c_rot*c_rot + r_rot*r_rot) / (2 * pow(0.5*d*hist_width, 2.0));
					interp_hist_entry(hist, rbin, cbin, obin, grad_mag, d, n);
				}
			}
		}
	return hist;
}

static void interp_hist_entry(vector<vector<vector<double>>>& hist, double rbin,
	double cbin, double obin, double mag, int d, int n)
{
	double d_r, d_c, d_o, v_r, v_c, v_o;
	int r0, c0, o0, rb, cb, ob, r, c, o;

	r0 = cvFloor(rbin);
	c0 = cvFloor(cbin);
	o0 = cvFloor(obin);
	d_r = rbin - r0;
	d_c = cbin - c0;
	d_o = obin - o0;

	for (r = 0; r <= 1; r++)
	{
		rb = r0 + r;
		if (rb >= 0 && rb < d)
		{
			v_r = mag * ((r == 0) ? 1.0 - d_r : d_r);
			for (c = 0; c <= 1; c++)
			{
				cb = c0 + c;
				if (cb >= 0 && cb < d)
				{
					v_c = v_r * ((c == 0) ? 1.0 - d_c : d_c);
					for (o = 0; o <= 1; o++)
					{
						ob = (o0 + o) % n;
						v_o = v_c * ((o == 0) ? 1.0 - d_o : d_o);
						hist[rb][cb][ob] += v_o;
					}
				}
			}
		}
	}
}

static void hist_to_descr(vector<vector<vector<double>>> hist, int d, int n,
	keypoint kp, vector<descriptor>& key_descs)
{
	vector<double> temp_feature;
	for (unsigned int r = 0; r < d; r++)
		for (unsigned int c = 0; c < d; c++)
			for (unsigned int o = 0; o < n; o++)
				temp_feature.push_back(hist[r][c][o]);

	normalize_descr(temp_feature);
	for (unsigned int i = 0; i < temp_feature.size(); i++)
	{
		if (temp_feature[i] > SIFT_DESCR_MAG_THR)
			temp_feature[i] = SIFT_DESCR_MAG_THR;
	}
	normalize_descr(temp_feature);

	/*for (unsigned int j = 0; j < temp_feature.size(); j++)
	{
		int int_val = SIFT_INT_DESCR_FCTR*temp_feature[j];
		temp_feature[j] = MIN(255, int_val);
	}*/

	key_descs.push_back(descriptor(kp.yi*pow(2.0, kp.octave - 1), kp.xi*pow(2.0, kp.octave - 1),
		temp_feature));
}

static void normalize_descr(vector<double>& feature)
{
	double average_sum,value_sum=0.0;
	for (unsigned int i = 0; i < feature.size(); i++)
	{
		value_sum += feature[i]* feature[i];
	}
	average_sum = 1.0 / sqrt(value_sum);
	for (unsigned int j = 0; j < feature.size(); j++)
	{
		feature[j] *= average_sum;
	}
}