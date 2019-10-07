#pragma once

#include <vector>

using namespace std;

class descriptor
{
public:
	float xi, yi;//��ԭͼ�ߴ��е�λ��
	vector<double> feature;

	descriptor() {}

	descriptor(float _yi, float _xi, vector<double> _feature)
	{
		xi = _xi;
		yi = _yi;
		feature = _feature;
	}
};