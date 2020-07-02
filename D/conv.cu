#include "conv.h"
#include "util.h"
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void insertZero(float* input, float* input_big, int H, int W, int C) {
  int h = blockDim.x * blockIdx.x + threadIdx.x;
  int w = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(h%2==0 && w%2==0){
		for(int c = 0; c < C; c++){
			input_big[h*(2*W-1)*C+w*C+c] = input[h/2*W*C+w/2*C+c];
		}
	}
	else{
		for(int c = 0; c < C; c++){
			input_big[h*(2*W-1)*C+w*C+c] = 0;
		}
	}
}

__global__ void im2col(float* input_big,float* input_col,int H,int W,int R,int S,int C,int OH,int OW,int stride,int pad){
  int oh = blockDim.x * blockIdx.x + threadIdx.x;
  int ow = blockDim.y * blockIdx.y + threadIdx.y;
	
	for(int r = 0; r < 4; r++){
		for(int s = 0; s < 4; s++){
			int ih = oh * stride - pad + r;
			int iw = ow * stride - pad + s;
			if (ih < 0 || ih >= H || iw < 0 || iw >= W){
				for(int c = 0; c < C; c++)
					input_col[(oh*OW+ow)*R*S*C+(r*S+s)*C+c] = 0;
			}   
			else{
				for(int c = 0; c < C; c++)
					input_col[(oh*OW+ow)*R*S*C+(r*S+s)*C+c] = input_big[ih*W*C+iw*C+c]; 
			}   
		}
	}
}

#define TS_M 4
#define TS_N 32
#define TS_K 32
__global__ void sgemm(float* A, float* B, float* C, int M, int N, int K) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = 4*(blockDim.y * blockIdx.y + threadIdx.y);
	int i_local = threadIdx.x;
	int j_local = 4*threadIdx.y;

	__shared__ float A_local[TS_M*TS_K];
	__shared__ float B_local[TS_K*TS_N];

	float res1 = 0.0;
	float res2 = 0.0;
	float res3 = 0.0;
	float res4 = 0.0;
  for (int kk = 0; kk < (K+TS_K-1)/TS_K; kk++){
    for (int jj = 0; jj < TS_K/TS_N*4; jj++)
      A_local[i_local*TS_K + (TS_K/TS_N)*j_local+jj] = (i < M && kk*TS_K+(TS_K/TS_N)*j_local+jj < K)? A[i*K + kk*TS_K+(TS_K/TS_N)*j_local+jj]:0;
    for (int ii = 0; ii < TS_K/TS_M; ii++){
      for (int jj = 0; jj < 4; jj++){
        B_local[((TS_K/TS_M)*i_local+ii)*TS_N+j_local+jj] = (kk*TS_K+(TS_K/TS_M*i_local+ii) < K && j+jj < N)? B[(kk*TS_K+(TS_K/TS_M*i_local+ii))*N+j+jj]:0;
      }   
    }   

    __syncthreads();

    for(int k = 0; k < TS_K; k++){
      float temp = A_local[i_local*TS_K+k];
      res1 += B_local[k*TS_N+j_local]*temp;
			res2 += B_local[k*TS_N+j_local+1]*temp;
			res3 += B_local[k*TS_N+j_local+2]*temp;
			res4 += B_local[k*TS_N+j_local+3]*temp;
    }   

    __syncthreads();
  }
  if (i < M && j < N)
    C[i*N+j] = res1;
  if (i < M && j+1 < N)
    C[i*N+j+1] = res2;
  if (i < M && j+2 < N)
    C[i*N+j+2] = res3;
  if (i < M && j+3 < N)
    C[i*N+j+3] = res4;
}
 
__global__ void addBias(float* output, float* bias, int OH, int OW, int K){
  int owoh = blockDim.x * blockIdx.x + threadIdx.x;
  int k = blockDim.y * blockIdx.y + threadIdx.y;
	
	output[owoh*K+k] += bias[k];
}

#define NDEV 4

static float *input_d[16], *filter_d[9][16], *bias_d[9][16], *filter_e[9][16], *bias_e[9][16], *input_big[16], *input_col[16], *output_d[16];

void conv_init() {
	for(int i = 0; i < 8; i++){
		cudaSetDevice(i%NDEV);
		cudaMalloc(&input_d[i], 64*64*256*sizeof(float)); // H*W*C
		cudaMalloc(&input_big[i], 4*64*64*256*sizeof(float)); //4*H*W*C
		cudaMalloc(&input_col[i], 256*4*4*128*128*sizeof(float)); // OH*OW*R*S*C
		cudaMalloc(&output_d[i], 128*128*64*sizeof(float)); //OW*OH*K
		for(int iter = 1; iter <= 8; iter++){
			switch(iter){
				case 1:
					cudaMalloc(&filter_e[iter][i], 4*4*3*64*sizeof(float)); //R*S*C*K
					cudaMalloc(&bias_e[iter][i], 64*sizeof(float)); //K
					break;
				case 2:
					cudaMalloc(&filter_e[iter][i], 4*4*64*128*sizeof(float)); //R*S*C*K
					cudaMalloc(&bias_e[iter][i], 128*sizeof(float)); //K
					break;
				case 3:
					cudaMalloc(&filter_e[iter][i], 4*4*128*256*sizeof(float)); //R*S*C*K
					cudaMalloc(&bias_e[iter][i], 256*sizeof(float)); //K
					break;
				case 4:
					cudaMalloc(&filter_e[iter][i], 4*4*256*512*sizeof(float)); //R*S*C*K
					cudaMalloc(&bias_e[iter][i], 512*sizeof(float)); //K
					break;
				case 5:
					cudaMalloc(&filter_e[iter][i], 4*4*512*512*sizeof(float)); //R*S*C*K
					cudaMalloc(&bias_e[iter][i], 512*sizeof(float)); //K
					break;
				case 6:
					cudaMalloc(&filter_e[iter][i], 4*4*512*512*sizeof(float)); //R*S*C*K
					cudaMalloc(&bias_e[iter][i], 512*sizeof(float)); //K
					break;
				case 7:
					cudaMalloc(&filter_e[iter][i], 4*4*512*512*sizeof(float)); //R*S*C*K
					cudaMalloc(&bias_e[iter][i], 512*sizeof(float)); //K
					break;
				case 8:
					cudaMalloc(&filter_e[iter][i], 4*4*512*512*sizeof(float)); //R*S*C*K
					cudaMalloc(&bias_e[iter][i], 512*sizeof(float)); //K
					break;
			}
		}
		for(int iter = 2; iter <= 8; iter++){
			switch(iter){
				case 1:
					cudaMalloc(&filter_d[iter][i], 4*4*3*128*sizeof(float)); //R*S*C*K
					cudaMalloc(&bias_d[iter][i], 3*sizeof(float)); //K
					break;
				case 2:
					cudaMalloc(&filter_d[iter][i], 4*4*64*256*sizeof(float)); //R*S*C*K
					cudaMalloc(&bias_d[iter][i], 64*sizeof(float)); //K
					break;
				case 3:
					cudaMalloc(&filter_d[iter][i], 4*4*128*512*sizeof(float)); //R*S*C*K
					cudaMalloc(&bias_d[iter][i], 128*sizeof(float)); //K
					break;
				case 4:
					cudaMalloc(&filter_d[iter][i], 4*4*256*1024*sizeof(float)); //R*S*C*K
					cudaMalloc(&bias_d[iter][i], 256*sizeof(float)); //K
					break;
				case 5:
					cudaMalloc(&filter_d[iter][i], 4*4*512*1024*sizeof(float)); //R*S*C*K
					cudaMalloc(&bias_d[iter][i], 512*sizeof(float)); //K
					break;
				case 6:
					cudaMalloc(&filter_d[iter][i], 4*4*512*1024*sizeof(float)); //R*S*C*K
					cudaMalloc(&bias_d[iter][i], 512*sizeof(float)); //K
					break;
				case 7:
					cudaMalloc(&filter_d[iter][i], 4*4*512*1024*sizeof(float)); //R*S*C*K
					cudaMalloc(&bias_d[iter][i], 512*sizeof(float)); //K
					break;
				case 8:
					cudaMalloc(&filter_d[iter][i], 4*4*512*512*sizeof(float)); //R*S*C*K
					cudaMalloc(&bias_d[iter][i], 512*sizeof(float)); //K
					break;
			}
		}
	} 
	//cudaDeviceSynchronize();
}

void filterbias_init(float* filter, float* bias, int R, int S, int C, int K, int thread, int iter){
	cudaSetDevice(thread%NDEV);
  cudaMemcpy(filter_e[iter][thread], filter, R*S*C*K*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(bias_e[iter][thread], bias, K*sizeof(float), cudaMemcpyHostToDevice);
}
void filterbias2_init(float* filter, float* bias, int R, int S, int C, int K, int thread, int iter){
	cudaSetDevice(thread%NDEV);
  cudaMemcpy(filter_d[iter][thread], filter, R*S*C*K*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(bias_d[iter][thread], bias, K*sizeof(float), cudaMemcpyHostToDevice);
}

void conv(float *input,float *filter,float *output,float *bias,int H,int W,int C,int R,int S,int K,int OH,int OW,int thread,int iter){
	cudaSetDevice(thread%NDEV);
  cudaMemcpy(input_d[thread], input, H*W*C*sizeof(float), cudaMemcpyHostToDevice);
  //cudaDeviceSynchronize();

  dim3 blockDim(2,2,1);
  dim3 gridDim(OW/2,OH/2,1);
  im2col<<<gridDim, blockDim>>>(input_d[thread], input_col[thread], H, W, R, S, C, OH, OW, 2 ,1);
  //cudaDeviceSynchronize();

	blockDim.x = TS_M;
	blockDim.y = TS_N/4;
	blockDim.z = 1;
	gridDim.x = (OW*OH+TS_M-1)/TS_M;
	gridDim.y = (K+TS_N-1)/TS_N;
	gridDim.z = 1;
  sgemm<<<gridDim, blockDim>>>(input_col[thread], filter_e[iter][thread], output_d[thread], OW*OH, K, R*S*C);
  //cudaDeviceSynchronize();
  
	blockDim.x = 1;
	blockDim.y = 1;
	blockDim.z = 1;
	gridDim.x = OW*OH;
	gridDim.y = K;
	gridDim.z = 1;
  addBias<<<gridDim, blockDim>>>(output_d[thread], bias_e[iter][thread], OW, OH, K);
  //cudaDeviceSynchronize();

	cudaMemcpy(output, output_d[thread], OH*OW*K*sizeof(float), cudaMemcpyDeviceToHost);
  //cudaDeviceSynchronize();
}

void conv_t(float *input,float *filter,float *output,float *bias,int H,int W,int C,int R,int S,int K,int OH,int OW,int thread, int iter){
	cudaSetDevice(thread%NDEV);
  cudaMemcpy(input_d[thread], input, H*W*C*sizeof(float), cudaMemcpyHostToDevice);
  //cudaDeviceSynchronize();

  dim3 blockDim(1,1,1);
  dim3 gridDim(2*H-1,2*W-1,1);
  insertZero<<<gridDim, blockDim>>>(input_d[thread], input_big[thread], H, W, C);
  //cudaDeviceSynchronize();

	blockDim.x = 2;
	blockDim.y = 2;
	blockDim.z = 1;
	gridDim.x = OW/2;
	gridDim.y = OH/2;
	gridDim.z = 1;
  im2col<<<gridDim, blockDim>>>(input_big[thread], input_col[thread], 2*H-1, 2*W-1, R, S, C, OH, OW, 1, 2);
  //cudaDeviceSynchronize();

	blockDim.x = TS_M;
	blockDim.y = TS_N/4;
	blockDim.z = 1;
	gridDim.x = (OW*OH+TS_M-1)/TS_M;
	gridDim.y = (K+TS_N-1)/TS_N;
	gridDim.z = 1;
  sgemm<<<gridDim, blockDim>>>(input_col[thread], filter_d[iter][thread], output_d[thread], OW*OH, K, R*S*C);
  //cudaDeviceSynchronize();
  
	blockDim.x = 1;
	blockDim.y = 1;
	blockDim.z = 1;
	gridDim.x = OW*OH;
	gridDim.y = K;
	gridDim.z = 1;
  addBias<<<gridDim, blockDim>>>(output_d[thread], bias_d[iter][thread], OW, OH, K);
  //cudaDeviceSynchronize();

	cudaMemcpy(output, output_d[thread], OH*OW*K*sizeof(float), cudaMemcpyDeviceToHost);
  //cudaDeviceSynchronize();
}

void mat_mul(float *A, float *B, float *C, int M, int N, int K, int thread, int iter) {
  dim3 blockDim(TS_M, TS_N/4, 1);
  dim3 gridDim((M+TS_M-1)/TS_M, (N+TS_N-1)/TS_N, 1);

	cudaSetDevice(thread%NDEV);
  cudaMemcpy(input_col[thread], A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  //cudaDeviceSynchronize();

  sgemm<<<gridDim, blockDim>>>(input_col[thread], filter_e[iter][thread], output_d[thread], M, N, K);
  //cudaDeviceSynchronize();

	cudaMemcpy(C, output_d[thread], M * N * sizeof(float), cudaMemcpyDeviceToHost);
  //cudaDeviceSynchronize();
}
