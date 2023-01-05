#include <time.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "CycleTimer.h"
#include "stereo.h"

using namespace cv;

int main(int argc, char *argv[]) {
	const char *fileNameL = argv[1];
	const char *fileNameR = argv[2];

	// read input image
	Mat1f imgL, imgR;
	imread(fileNameL, 0).convertTo(imgL, IMREAD_GRAYSCALE);
	imread(fileNameR, 0).convertTo(imgR, IMREAD_GRAYSCALE);

	Mat1f disparityL;
	// compute disparity
	double startTime = CycleTimer::currentSeconds();
	stereoMatch(imgL, imgR, disparityL); // disparity Left
	double endTime = CycleTimer::currentSeconds();
	printf("time: %f sec\n", (double)  (endTime-startTime));

	// write file
	normalize(disparityL, disparityL, 0, 1, NORM_MINMAX);

	imwrite("disparityL.png", disparityL*255);

	return 0;
}
