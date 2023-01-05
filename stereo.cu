#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "stereo.h"
#define BLOCK_SIZE 32

// 用來調整window size 比較不同window size花費的時間
int winSize     = 7;
int searchRange = 100;

// 方便進行block matching
int halfWinSize     = winSize /  2;
int halfSearchRange = searchRange / 2;

// GPU 的 Kernel

__global__ void Calculate(float *d_imgSrc, float *d_imgDst, float *d_disparity, int halfWinSize, int halfSearchRange)
{
    // 根據 CUDA 模型，算出當下 thread 對應的 x 與 y
    const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = idx_y * 881 + idx_x;
    if(idx_y>2&&idx_y<402){
        if(idx_x>2&&idx_x<879){
        int cxDstMin = max(idx_x - halfSearchRange, halfWinSize); //left edge
        int cxDstMax = min(idx_x + halfSearchRange, 881-halfWinSize);//right edge

        float minSad        = FLT_MAX;
        float bestDisparity = 0;

        for (int cxDst = cxDstMin; cxDst < cxDstMax; ++cxDst) {
				float sad=0.f ;
                for(int y = -halfWinSize; y < halfWinSize+1; y++) {
                    for (int x = -halfWinSize; x < halfWinSize+1; x++) {
                    int idx_xWindow = idx_x+x;
                    int idx_yWindow = idx_y+y;
                    int idx_Window_Src =idx_yWindow * 881 + idx_xWindow;
                    int idx_Window_Dst =idx_yWindow * 881 + cxDst;
                    sad += abs(d_imgSrc[idx_Window_Src] - d_imgDst[idx_Window_Dst]);
                    }
                }
				if ( sad < minSad ) {
					minSad        = sad;
					bestDisparity = abs( cxDst - idx_x );
				}
        }
        d_disparity[idx]=bestDisparity;
        }
    }

}

void stereoMatch(const cv::Mat1f &imgSrc, const cv::Mat1f &imgDst, cv::Mat1f &disparity) {
    float* d_imgSrc; float* d_imgDst; float* d_disparity;
    cudaMalloc(&d_imgSrc,  imgSrc.total() * sizeof(float));
    cudaMalloc(&d_imgDst,  imgDst.total() * sizeof(float));
    disparity = cv::Mat1f::zeros( imgSrc.size() );
    cudaMalloc(&d_disparity,  disparity.total() * sizeof(float));
    cudaMemcpy(d_imgSrc, imgSrc.ptr<float>(0), imgSrc.total() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imgDst, imgDst.ptr<float>(0), imgDst.total() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_disparity, disparity.ptr<float>(0), disparity.total() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlock(881 / BLOCK_SIZE+1, 400 / BLOCK_SIZE+1);
    Calculate<<<numBlock, blockSize>>>(d_imgSrc, d_imgDst, d_disparity, halfWinSize, halfSearchRange);
    cudaDeviceSynchronize();
    cudaMemcpy(disparity.ptr(), d_disparity, disparity.total() * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_imgDst);
	cudaFree(d_imgSrc);
    cudaFree(d_disparity);
}
