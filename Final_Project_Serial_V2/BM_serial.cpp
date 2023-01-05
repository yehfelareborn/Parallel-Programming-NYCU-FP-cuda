#include <time.h>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include "CycleTimer.h"

using namespace cv;

// 用來調整window size 比較不同window size花費的時間
int winSize     = 7;
int searchRange = 100;


// 方便進行block matching
int halfWinSize     = winSize /  2;
int halfSearchRange = searchRange / 2;


// 取得左圖或右圖以某個點為中心的block
Mat1f getBlock(const Mat1f &img, int cx, int cy) {
	Range rangeY(cy - halfWinSize, cy + halfWinSize + 1);
	Range rangeX(cx - halfWinSize, cx + halfWinSize + 1);
	return img(rangeY, rangeX);
}

float computeSadOverBlock(const Mat1f &imgL, const Mat1f &imgR, int cy, int cxSrc, int cxDst) {
	float sad = 0.f;
	for(int y = -halfWinSize; y < halfWinSize+1; y++) {
		for (int x = -halfWinSize; x < halfWinSize+1; x++) {
			sad += abs(imgL(cy+y, cxSrc+x) - imgR(cy+y, cxDst+x));
		}
	}
			
	return sad;
}

void stereoMatch(const Mat1f &imgSrc, const Mat1f &imgDst, Mat1f &disparity) {
	disparity = Mat1f::zeros( imgSrc.size() );
// #pragma omp parallel for num_threads(16)
    for (int cy = halfWinSize; cy < imgSrc.rows-halfWinSize; ++cy) {
		for (int cxSrc = halfWinSize; cxSrc < imgSrc.cols-halfWinSize; ++cxSrc) {
			// left patch
			// Mat1f patchL = getBlock( imgSrc, cxSrc, cy );

			// epipolar line search range
			int cxDstMin = max(cxSrc - halfSearchRange, halfWinSize);
			int cxDstMax = min(cxSrc + halfSearchRange, imgSrc.cols-halfWinSize);

			// find best match disparity
			float minSad        = FLT_MAX;
			float bestDisparity = 0;

        // #pragma omp parallel for num_threads(8)
            for (int cxDst = cxDstMin; cxDst < cxDstMax; ++cxDst) {
				// right patch
				// Mat1f patchR = getBlock( imgDst, cxDst, cy );

				// patch diff: Sum of Absolute Difference
				// Mat1f diff = abs( patchL - patchR );
				// float sad  = sum(diff)[0];
				float sad = computeSadOverBlock(imgSrc, imgDst, cy, cxSrc, cxDst);

				// update best SAD and disparity
				if ( sad < minSad ) {
					minSad        = sad;
					bestDisparity = abs( cxDst - cxSrc );
				}
			}

			disparity(cy, cxSrc) = bestDisparity;
		}
	}
    
}

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

	imshow("disparityL", disparityL);

	imwrite("disparityL.png", disparityL*255);

	waitKey();

	return 0;
}