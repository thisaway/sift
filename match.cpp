#include "sift.h"
#include <float.h>
#include <Windows.h>

static double mode(vector<double> feature);

vector<vector<cv::Point>> match(sift sift1, sift sift2)
{
	vector<vector<cv::Point>> all_match_point;
	vector<descriptor> des1, des2;
	des1 = sift1.get_des();
	des2 = sift2.get_des();

	const float distRatio = 0.6;
	for (int d2 = 0; d2 < des2.size(); d2++)
	{
		vector<cv::Point> match_point;
		double first_min = DBL_MAX, second_min = DBL_MAX;
		cv::Point target_point;
		for (int d1 = 0; d1 < des1.size(); d1++)
		{
			double temp_sum = 0.0;
			double mode_mul;
			
			for (int f = 0; f < 128; f++)
			{
//				cout << "des1[" << d1 << "].feature[" << f << "]:" << des1[d1].feature[f] << " ";
	//			cout << "des2[" << d2 << "].feature[" << f << "]:" << des2[d2].feature[f] <<endl;
				temp_sum += des1[d1].feature[f] * des2[d2].feature[f];
			}
			//mode_mul = mode(des1[d1].feature)*mode(des2[d2].feature);
			temp_sum = acos(temp_sum);
			//cout << temp_sum << endl;
			if (temp_sum < first_min)
			{
				target_point = cv::Point(des1[d1].xi, des1[d1].yi);
				second_min = first_min;
				first_min = temp_sum;
			}
			else if (temp_sum < second_min)second_min = temp_sum;
		}
		if (first_min < distRatio*second_min)
		{
			match_point.push_back(target_point);
			match_point.push_back(cv::Point(des2[d2].xi, des2[d2].yi));
			all_match_point.push_back(match_point);
		}
	}
	

	for (int i = 0; i < all_match_point.size(); i++)
	{
		cv::Mat img1 = sift1.get_img(), img2 = sift2.get_img();
		cv::Mat cont_img;
		cv::hconcat(img1, img2, cont_img);

		cv::Scalar color = (255 * (rand() / (1.0 + RAND_MAX)), 255 * (rand() / (1.0 + RAND_MAX)), 255 * (rand() / (1.0 + RAND_MAX)));
		circle(cont_img, all_match_point[i][0], 1, color, -1);
		circle(cont_img, cv::Point(all_match_point[i][1].x + img1.cols, all_match_point[i][1].y), 1, color, -1);
		line(cont_img, all_match_point[i][0], cv::Point(all_match_point[i][1].x + img1.cols, all_match_point[i][1].y),
			color, 2);
		imshow("result", cont_img);
		cv::waitKey(0);

	}
	

	return all_match_point;
}

static double mode(vector<double> feature)
{
	double re_value = 0;
	for (int i = 0; i < feature.size(); i++)
	{
		re_value += pow(feature[i],2);
	}
	re_value = sqrt(re_value);

	return re_value;
}