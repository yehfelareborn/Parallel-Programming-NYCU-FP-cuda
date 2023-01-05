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
	int start_colDst;
	int end_colDst;
    int cy;
    int cxSrc;
    float minSad;
    int bestCxDst;
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

void *stereoMatchColDst(void *arg) {
    ThreadData *data = (ThreadData *) arg;
    Mat1f *imageL = data->imageL;
    Mat1f *imageR = data->imageR;
    Mat1f *disp = data->disp;
    int start_colDst = data->start_colDst;
    int end_colDst = data->end_colDst;
    int cy = data->cy;
    int cxSrc = data->cxSrc;
    float &minSad = data->minSad;
    int &bestCxDst = data->bestCxDst;
    
    for (int cxDst = start_colDst; cxDst <= end_colDst; ++cxDst) {
        float sad = computeSadOverBlock(*imageL, *imageR, cy, cxSrc, cxDst);

        // update best SAD and disparity
        if ( sad < minSad ) {
            minSad        = sad;
            bestCxDst = cxDst;
        }
    }
    

    pthread_exit(NULL);
}

void stereoMatch(Mat1f &imgSrc, Mat1f &imgDst, Mat1f &disparity) {
	disparity = Mat1f::zeros(imgSrc.size());

    for (int cy = halfWinSize; cy < imgSrc.rows-halfWinSize; ++cy) {
		for (int cxSrc = halfWinSize; cxSrc < imgSrc.cols-halfWinSize; ++cxSrc) {
			// epipolar line search range
			int cxDstMin = max(cxSrc - halfSearchRange, halfWinSize);
			int cxDstMax = min(cxSrc + halfSearchRange, imgSrc.cols-halfWinSize);

            // find best match disparity
            pthread_t *thread_handles = (pthread_t *) malloc(thread_count * sizeof(pthread_t));

            ThreadData data[thread_count];

            int colDst = cxDstMax - cxDstMin;
            int colDst_per_thread = colDst / thread_count;
            int remain_colDst = colDst % thread_count;

            for (int i = 0; i < thread_count; i++) {
                data[i].imageL = &imgSrc;
                data[i].imageR = &imgDst;
                data[i].disp = &disparity;
                data[i].start_colDst = cxDstMin + i * colDst_per_thread;
                data[i].cy = cy;
                data[i].cxSrc = cxSrc;
                data[i].minSad = FLT_MAX;
                data[i].bestCxDst = 0;

                if (i == thread_count - 1)
                    data[i].end_colDst = cxDstMin + (i+1) * colDst_per_thread - 1 + remain_colDst;
                else
                    data[i].end_colDst = cxDstMin + (i+1) * colDst_per_thread - 1;

                int rc = pthread_create(&thread_handles[i], NULL, stereoMatchColDst, (void *) &data[i]);
                if(rc) {
                    printf("fail to create thread\n");
                }
            }

            for (int i = 0; i < thread_count; i++)
                pthread_join(thread_handles[i], NULL);

            float minSad = data[0].minSad;
            int bestCxDst = data[0].bestCxDst;
            for (int i = 1; i < thread_count; i++) {
                if (data[i].minSad < minSad) {
                    minSad = data[i].minSad;
                    bestCxDst = data[i].bestCxDst;
                }
            }

			disparity(cy, cxSrc) = (float) std::abs(bestCxDst - cxSrc);
		}
	}
    
}

int main(int argc, char *argv[]) {
	const char *fileNameL = argv[1];
	const char *fileNameR = argv[2];
    if (argv[3] == NULL)
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