#pragma once

void conv_init();
void filterbias_init(float* filter, float* bias, int R, int S, int C, int K, int thread,int iter);
void filterbias2_init(float* filter, float* bias, int R, int S, int C, int K, int thread,int iter);
void mat_mul(float *A, float *B, float *C, int M, int N, int K, int thread, int iter);
void conv_t(float *input,float *filter,float *output,float *bias,int H,int W,int C,int R,int S,int K,int OH,int OW,int thread,int iter);
void conv(float *input,float *filter,float *output,float *bias,int H,int W,int C,int R,int S,int K,int OH,int OW,int thread,int iter);
