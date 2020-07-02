#include "conv.h"
#include <omp.h>
#define BS1 128
#define BS2 1024
#define MIN(A,B) ((A<B)? A:B)

void mat_mul(float *A, float *B, float *C, int M, int N, int K, int num_threads) {
  #pragma omp parallel for num_threads(num_threads)
	for (int i = 0; i < (M+2)/3; ++i) { 
		if(3*i == M-1){
			int x1 = 3*i*N;
			for (int k = 0; k < K; k += BS1){
				for (int j = 0; j < N; j += BS2){
					for (int kk = k; kk < MIN(k+BS1,K); kk++){
						float temp1 = A[3*i*K+kk];
						int y = kk*N;
						for (int jj = j; jj < MIN(j+BS2,N); jj++){
							C[x1 + jj] += temp1 * B[y+jj];
						}
					}
				}
			}
		}
		else if(3*i == M-2){
			int x1 = 3*i*N;
			int x2 = (3*i+1)*N;
			for (int k = 0; k < K; k += BS1){
				for (int j = 0; j < N; j += BS2){
					for (int kk = k; kk < MIN(k+BS1,K); kk++){
						float temp1 = A[3*i*K+kk];
						float temp2 = A[(3*i+1)*K+kk];
						int y = kk*N;
						for (int jj = j; jj < MIN(j+BS2,N); jj++){
							float temp = B[y+jj];
							C[x1 + jj] += temp1 * temp;
							C[x2 + jj] += temp2 * temp;
						}
					}
				}
			}
		}
		else{
			int x1 = 3*i*N;
			int x2 = (3*i+1)*N;
			int x3 = (3*i+2)*N;
			for (int k = 0; k < K; k += BS1){
				for (int j = 0; j < N; j += BS2){
					for (int kk = k; kk < MIN(k+BS1,K); kk++){
						float temp1 = A[3*i*K+kk];
						float temp2 = A[(3*i+1)*K+kk];
						float temp3 = A[(3*i+2)*K+kk];
						int y = kk*N;
						for (int jj = j; jj < MIN(j+BS2,N); jj++){
							float temp = B[y+jj];
							C[x1 + jj] += temp1 * temp;
							C[x2 + jj] += temp2 * temp;
							C[x3 + jj] += temp3 * temp;
						}
					}
				}
			}
		}
  }
}
