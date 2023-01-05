#include <time.h>
#include <opencv2/opencv.hpp>
#include "CycleTimer.h"
#include <pthread.h>

using namespace cv;

// arguments for thread function
struct ThreadData{
	Mat1f *imageL;
	Mat1f *imageR;
	Mat1f *disp;
	int start_row;
	int end_row;
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

void *stereoMatchThread(void *arg) {
	ThreadData *data = (ThreadData *) arg;
	Mat1f *imageL = data->imageL;
	Mat1f *imageR = data->imageR;
	Mat1f *disp = data->disp;	
	int start_row = data->start_row;
	int end_row = data->end_row;

	for (int cy = start_row; cy <= end_row; ++cy) {
		for (int cxSrc = halfWinSize; cxSrc < (imageL->cols)-halfWinSize; ++cxSrc) {
			// epipolar line search range
			int cxDstMin = max(cxSrc - halfSearchRange, halfWinSize);
			int cxDstMax = min(cxSrc + halfSearchRange, (imageL->cols) - halfWinSize);

			// find best match disparity
			float minSad = FLT_MAX;
			int bestCxDst = 0;

			for (int cxDst = cxDstMin; cxDst < cxDstMax; ++cxDst) {
				float sad = computeSadOverBlock(*imageL, *imageR, cy, cxSrc, cxDst);

				// update best SAD and disparity
				if (sad < minSad) {
					minSad = sad;
					bestCxDst = cxDst;
				}
			}

			(*disp)(cy, cxSrc) = (float) std::abs(bestCxDst - cxSrc);
		}	
	}

	pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
	const char *fileNameL = argv[1];
	const char *fileNameR = argv[2];		
	int thread_count;
	if (argv[3] == NULL)
		thread_count = 8;
	else
		thread_count = atoi(argv[3]);

	// read input image
	Mat1f imgL = imread(fileNameL, 0);
	Mat1f imgR = imread(fileNameR, 0);
	Mat1f dispL = Mat1f::zeros(imgL.size());

	pthread_t *thread_handles;
	thread_handles = (pthread_t * ) malloc(thread_count * sizeof(pthread_t));

	ThreadData data[thread_count];

	int rows = imgL.rows - 2 * halfWinSize;	// total rows to be computed
	int row_per_thread = rows / thread_count;
	int remain_row = rows % thread_count;

	// compute disparity
	double startTime = CycleTimer::currentSeconds();
	
	// Create thread
	for (int i = 0; i < thread_count; i++) {
		data[i].imageL = &imgL;
		data[i].imageR = &imgR;
		data[i].disp = &dispL;
		data[i].start_row = halfWinSize + i * row_per_thread;

		if (i == thread_count - 1)
			data[i].end_row = min(imgL.rows - halfWinSize - 1, halfWinSize + (i+1) * row_per_thread - 1 + remain_row);
		else
			data[i].end_row = halfWinSize + (i+1) * row_per_thread - 1;

		int rc = pthread_create(&thread_handles[i], NULL, stereoMatchThread, (void *) &data[i]);

		if(rc) {
			printf("fail to create thread\n");
			return -1;
		}
	}	

	// Wait for threads to complete their tasks
	for (int i = 0; i < thread_count; i++) 
		pthread_join(thread_handles[i], NULL);

	double endTime = CycleTimer::currentSeconds();
	printf("time: %f sec\n", (double)  (endTime-startTime));

	// write file
	normalize(dispL, dispL, 0, 1, NORM_MINMAX);

	imshow("disparityL", dispL);

	imwrite("disparityL.png", dispL*255);

	waitKey();

	free(thread_handles);

	return 0;
}