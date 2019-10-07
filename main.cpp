#include "sift.h"
using namespace cv;

vector<vector<Point>> match(sift sift1, sift sift2);

int main()
{
	sift sift1("./ucsb1.jpg", 3);
	sift1.dosift();
	sift sift2("./ucsb2.jpg", 3);
	sift2.dosift();
	vector<vector<Point>> result = match(sift1, sift2);

	//sift.ShowAbsSigma();
	//sift.ShowKeypoints();
	waitKey(0);

	return 0;
}