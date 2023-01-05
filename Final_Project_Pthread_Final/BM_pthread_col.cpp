#include <time.h>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include "CycleTimer.h"
#include <pthread.h>

using namespace cv;

int thread_count;
// arguments for thread function
struct ThreadData{
	Mat1f *imageL;
	Mat1f *imageR;
	Mat1f *disp;
	int start_col;
	int end_col;
	int cy;
};

// 用來調整window size 比較不同window size花費的時間
int winSize     = 7;
int searchRange = 100;


// 方便進行block matching
int halfWinSize     = winSize /  2;
int halfSearchRange = searchRange / 2;

float computeSadOverBlock(const Mat1f &imgL, const Mat1f &imgR, int cy, int cxSrc, int cxDst) {
	float sad = 0.f;
	for(int y = -halfWinSize; y < halfWinSize+1; y++) {
		for (int x = -halfWinSize; x < halfWinSize+1; x++) {
			sad += std::abs(imgL(cy+y, cxSrc+x) - imgR(cy+y, cxDst+x));
		}
	}
			
	return sad;
}

void *stereoMatchCol(void *arg) {
	ThreadData *data = (ThreadData *) arg;
	Mat1f *imageL = data->imageL;
	Mat1f *imageR = data->imageR;
	Mat1f *disp = data->disp;
	int start_col = data->start_col;
	int end_col = data->end_col;
	int cy = data->cy;

	for (int cxSrc = start_col; cxSrc <= end_col; ++cxSrc) {
		// epipolar line search range
		int cxDstMin = max(cxSrc - halfSearchRange, halfWinSize);
		int cxDstMax = min(cxSrc + halfSearchRange, (imageL->cols) - halfWinSize);

		// find best match disparity
		float minSad        = FLT_MAX;
		int bestCxDst = 0;

		for (int cxDst = cxDstMin; cxDst < cxDstMax; ++cxDst) {
			float sad = computeSadOverBlock(*imageL, *imageR, cy, cxSrc, cxDst);
			
			// update best SAD and disparity
			if ( sad < minSad ) {
				minSad        = sad;
				bestCxDst = cxDst;
			}
		}

		(*disp)(cy, cxSrc) = (float) std::abs(bestCxDst - cxSrc);
	}

	pthread_exit(NULL);
}

void stereoMatch(Mat1f &imgSrc, Mat1f &imgDst, Mat1f &disparity) {
	disparity = Mat1f::zeros(imgSrc.size());
    for (int cy = halfWinSize; cy < imgSrc.rows-halfWinSize; ++cy) {

		pthread_t *thread_handles = (pthread_t *) malloc(thread_count * sizeof(pthread_t));
		ThreadData data[thread_count];

		int cols = imgSrc.cols - 2 * halfWinSize;
		int col_per_thread = cols / thread_count;
		int remain_col = cols % thread_count;

		for (int i = 0; i < thread_count; i++) {
			data[i].imageL = &imgSrc;
			data[i].imageR = &imgDst;
			data[i].disp = &disparity;
			data[i].start_col = halfWinSize + i * col_per_thread;
			data[i].cy = cy;

			if (i == thread_count - 1)
				data[i].end_col = min(imgSrc.cols - halfWinSize - 1, halfWinSize + (i+1) * col_per_thread - 1 + remain_col);
			else
				data[i].end_col = halfWinSize + (i+1) * col_per_thread - 1;

			int rc = pthread_create(&thread_handles[i], NULL, stereoMatchCol, (void *) &data[i]);
			if(rc) {
				printf("fail to create thread\n");
			}
		}

		for(int i = 0; i < thread_count; i++)
			pthread_join(thread_handles[i], NULL);


	}
    
}

int main(int argc, char *argv[]) {
	const char *fileNameL = argv[1];
	const char *fileNameR = argv[2];
	if(argv[3] == NULL)
		thread_count = 8;
	else
		thread_count = atoi(argv[3]);

	// read input image
	Mat1f imgL = imread(fileNameL, 0);
	Mat1f imgR = imread(fileNameR, 0);

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