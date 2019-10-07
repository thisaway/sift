#pragma once

#include <vector>

using namespace std;

class keypoint
{
public:
	int octave;
	int interval;
	//在各组尺寸下的位置
	float xi;
	float yi;
	vector<double> mag;
	vector<double> orien;
	float scl_octv;

	keypoint() {}
	keypoint(int _octave, int _interval, float _xi, float _yi, float _scl_octv)
	{
		octave = _octave;
		interval = _interval;
		xi = _xi;
		yi = _yi;
		scl_octv = _scl_octv;
	}

	keypoint(int _octave, int _interval, float _xi, float _yi,
		vector<double> _mag, vector<double> _orien, float _scl_octv)
	{
		octave = _octave;
		interval = _interval;
		xi = _xi;
		yi = _yi;
		mag = _mag;
		orien = _orien;
		scl_octv = _scl_octv;
	}
};