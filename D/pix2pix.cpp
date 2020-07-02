#include "pix2pix.h"
#include "util.h"
#include "conv.h"

#include <omp.h>
#include <mpi.h>
#include <string>
#include <map>
#include <cmath>
#include <iostream>

#define NUM_THREADS 4

class Tensor {
public:
  Tensor();
  Tensor(float *buf_, std::vector<size_t> shape_);
  void alloc_once(std::vector<size_t> shape_);
  void set_sz();
  void set_buf(float *buf_, std::vector<size_t> shape_, size_t sz_);
	void print();
	void transpose();
	void reverse();
	bool is_same(Tensor t);

  float* buf;
  std::vector<size_t> shape;
  size_t sz;
};

std::map<std::string, Tensor> weights;
Tensor input;
uint8_t *out;
size_t num_image_glob;

// Helpers
static void register_weight(std::map<std::string, Tensor>& weights, float* (*buf), std::string name, std::vector<size_t> shape);
static std::map<std::string, Tensor> register_weights(float* weight_buf);
static Tensor preprocess(uint8_t *in, size_t num_image);
static void postprocess_one_image(Tensor input, uint8_t *out, size_t idx);
static void get_one_image(Tensor input, Tensor &output, size_t idx);

// Operators
static void conv2d_gpu(Tensor input, Tensor filter, Tensor bias, Tensor &output, int thread, int iter);
static void conv2d_transposed1(Tensor input, Tensor filter, Tensor bias, Tensor &output);
static void conv2d_transposed_gpu(Tensor input, Tensor filter, Tensor bias, Tensor &output, int thread, int iter);
static void leaky_relu(Tensor input, Tensor &output, float alpha);
static void relu(Tensor input, Tensor &output);
static void batchnorm(Tensor input, Tensor scale, Tensor offset, Tensor &output);
static void concat(Tensor input0, Tensor input1, Tensor &output);
static void elem_tanh(Tensor input, Tensor &output);
static void* pix2pix_thread(void *data);

int get_size(){
	int size = 0;
#ifdef USE_MPI
	MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
	return size;
}

void pix2pix_init() {
  conv_init();
}

void pix2pix(uint8_t *input_buf, float *weight_buf, uint8_t *output_buf, size_t num_image) {
	MPI_Request request[7];
	MPI_Status status[7];
	MPI_Request request1[3];
	MPI_Status status1[3];
	int mpi_size = get_size();
	int mpi_rank = get_rank();
	
  MPI_Bcast(&num_image, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	
	int num_start = num_image*mpi_rank/mpi_size;
	int num_end = num_image*(mpi_rank+1)/mpi_size;
	int num_size = num_end - num_start;

	size_t input_sz = num_size*256*256*3;
	size_t weight_sz = size_t(217725452)/sizeof(float);

	if (mpi_rank != 0) {
		input_buf = (uint8_t*) malloc(input_sz*sizeof(uint8_t));
		weight_buf = (float*) malloc(weight_sz*sizeof(float));
		output_buf = (uint8_t*) malloc(input_sz*sizeof(uint8_t));
	}	

	MPI_Bcast(weight_buf, int(weight_sz), MPI_FLOAT, 0, MPI_COMM_WORLD);
	
  if (mpi_rank == 0) {
    for (int i = 1; i < mpi_size; ++i) {
      //MPI_Isend(&input_buf[input_sz*i/mpi_size], input_sz/mpi_size, MPI_UINT8_T, i, 0, MPI_COMM_WORLD, &request1[i-1]);
      MPI_Isend(&input_buf[((num_image*i)/mpi_size)*256*256*3], ((num_image*(i+1))/mpi_size-(num_image*i)/mpi_size)*256*256*3, MPI_UINT8_T, i, 0, MPI_COMM_WORLD, &request1[i-1]);
    }   
		MPI_Waitall(mpi_size-1,request1,status1);
  } else {
		MPI_Recv(input_buf, num_size*256*256*3, MPI_UINT8_T, 0, 0, MPI_COMM_WORLD, &status[1]);
  }

  weights = register_weights(weight_buf);
	input = preprocess(input_buf, num_size);
		
	out = output_buf;	
	num_image_glob = num_size;

	int num[NUM_THREADS];
	pthread_t thread[NUM_THREADS];
	for(int i=0;i<NUM_THREADS;i++){
		num[i]=i;
		pthread_create(&thread[i],NULL,pix2pix_thread,&num[i]);
	}
	for(int i=0;i<NUM_THREADS;i++){
		pthread_join(thread[i],NULL);
	}

  if (mpi_rank == 0) {
    for (int i = 1; i < mpi_size; ++i) {
      MPI_Irecv(&output_buf[((num_image*i)/mpi_size)*256*256*3], ((num_image*(i+1))/mpi_size-(num_image*i)/mpi_size)*256*256*3, MPI_UINT8_T, i, 0, MPI_COMM_WORLD, &request[2+i]);
    }   
		MPI_Waitall(mpi_size-1,&request[3],&status[3]);
  } else {
    MPI_Send(output_buf, num_size*256*256*3, MPI_UINT8_T, 0, 0, MPI_COMM_WORLD);
  }
}

static void* pix2pix_thread(void *data){
	Tensor one_image;
	Tensor encoder_layer_input[9];
	Tensor encoder_layer_rectified[9];
	Tensor encoder_layer_convolved[9];
	Tensor encoder_layer[9];
	Tensor decoder_layer_input[9];
	Tensor decoder_layer_rectified[9];
	Tensor decoder_layer_convolved[9];
	Tensor decoder_layer[9];
	Tensor filter_reversed[9];
	int *p = (int*)data;
	int thread = *p;

	for (int i = 1; i <= 8; i++) {
		auto scope = "generator/decoder_" + std::to_string(i);
		auto filter = weights[scope + "/conv2d_transpose/kernel"];
		filter_reversed[i].alloc_once(filter.shape);
		size_t R = filter.shape[0];
		size_t S = filter.shape[1];
		size_t X = filter.shape[2]*filter.shape[3];
		for(size_t r = 0; r < R; r++){
			for(size_t s = 0; s < S; s++){
				for(size_t x = 0; x < X; x++){
					filter_reversed[i].buf[r*S*X+s*X+x] = filter.buf[(R-1-r)*S*X+(S-1-s)*X+x];
				}
			}
		}
		filter_reversed[i].transpose();
	}

  for(int i = 1; i <= 8; i++){
    auto scope = "generator/encoder_" + std::to_string(i);
    auto filter = weights[scope + "/conv2d/kernel"];
    auto bias = weights[scope + "/conv2d/bias"];
    size_t R,S,C,K;
    R = filter.shape[0]; 
		S = filter.shape[1]; 
		C = filter.shape[2]; 
		K = filter.shape[3];
		filterbias_init(filter.buf,bias.buf,R,S,C,K,thread,i);
	}

  for(int i = 1; i <= 8; i++){
    auto scope = "generator/decoder_" + std::to_string(i);
		auto filter = weights[scope + "/conv2d_transpose/kernel"];
    auto bias = weights[scope + "/conv2d_transpose/bias"];
    size_t R,S,C,K;
    R = filter.shape[0]; 
		S = filter.shape[1]; 
		C = filter.shape[2]; 
		K = filter.shape[3];
		filterbias2_init(filter_reversed[i].buf,bias.buf,R,S,C,K,thread,i);
	}

  for (size_t img_idx = thread; img_idx < num_image_glob; img_idx+=NUM_THREADS) {	
		get_one_image(input, one_image, img_idx);
		//if(get_rank()==3) printf("image: %lu\n",img_idx);
		/*
		 * Encoding phase
		 */
		conv2d_gpu(one_image, weights["generator/encoder_1/conv2d/kernel"], weights["generator/encoder_1/conv2d/bias"], encoder_layer[1], thread, 1);
		for (int i = 2; i <= 8; ++i) {
    	auto scope = "generator/encoder_" + std::to_string(i);
			// Encoder i : leaky_relu => conv2d => batchnorm
			encoder_layer_input[i] = encoder_layer[i - 1];
			leaky_relu(encoder_layer_input[i], encoder_layer_rectified[i], 0.2);
			conv2d_gpu(encoder_layer_rectified[i], weights[scope + "/conv2d/kernel"], weights[scope + "/conv2d/bias"], encoder_layer_convolved[i], thread, i);
			batchnorm(encoder_layer_convolved[i], weights[scope + "/batch_normalization/gamma"], weights[scope + "/batch_normalization/beta"], encoder_layer[i]);
		}
		/*
		 * Decoding phase
		 */
		for (int i = 8; i >= 1; --i) {
    	auto scope = "generator/decoder_" + std::to_string(i);
			if (i == 8) {
				decoder_layer_input[i] = encoder_layer[8];
			} else {
				concat(decoder_layer[i + 1], encoder_layer[i], decoder_layer_input[i]);
			}
			relu(decoder_layer_input[i], decoder_layer_rectified[i]);
			if (i == 1){
				conv2d_transposed1(decoder_layer_rectified[i], weights[scope + "/conv2d_transpose/kernel"], weights[scope + "/conv2d_transpose/bias"], decoder_layer_convolved[i]);
				//conv2d_transposed_gpu(decoder_layer_rectified[i], filter_reversed[i], weights[scope + "/conv2d_transpose/bias"], decoder_layer_convolved[i], thread, i);
			} else{
				conv2d_transposed_gpu(decoder_layer_rectified[i], filter_reversed[i], weights[scope + "/conv2d_transpose/bias"], decoder_layer_convolved[i], thread, i);
			}
			if (i == 1) break;
			batchnorm(decoder_layer_convolved[i], weights[scope + "/batch_normalization/gamma"], weights[scope + "/batch_normalization/beta"], decoder_layer[i]);
		}
		elem_tanh(decoder_layer_convolved[1], decoder_layer[1]);
		postprocess_one_image(decoder_layer[1], out, img_idx);
	}
	return NULL; 
}

Tensor::Tensor() : buf(NULL) {}

// If buf is given, use it. If not, allocate new one.
Tensor::Tensor(float *buf_, std::vector<size_t> shape_) : buf(buf_), shape(shape_) {
  set_sz();
  if (buf == NULL) {
    buf = (float*)malloc(sz * sizeof(float));
  }
}

// If buf is not allocated, allocate new one.
void Tensor::alloc_once(std::vector<size_t> shape_) {
  if (buf == NULL) {
    shape = shape_;
    set_sz();
    buf = (float*)malloc(sz * sizeof(float));
  }
}

void Tensor::set_buf(float *buf_, std::vector<size_t> shape_, size_t sz_) {
  buf = buf_;
  shape = shape_;
  sz = sz_;
}


void Tensor::set_sz() {
  sz = 1;
  for (auto x : shape) {
    sz *= x;
  }
}

void Tensor::print() {
  for (size_t i = 0; i < shape[0]; ++i) {
  	for (size_t j = 0; j < shape[1]; ++j) {
			for (size_t k = 0; k < sz/shape[0]/shape[1]; ++k) {
				printf("%.1f ",buf[i*(sz/shape[0])+j*(sz/shape[0]/shape[1])+k]);
			}
		printf("\n");
		}
		printf("\n");
  }
}

void Tensor::transpose() {
  float* transposed = (float*)malloc(sz * sizeof(float));
	size_t R = shape[0];
	size_t S = shape[1];
	size_t K = shape[2];
	size_t C = shape[3];
  for (size_t r = 0; r < R; ++r) {
  	for (size_t s = 0; s < S; ++s) {
			for (size_t c = 0; c < C; ++c) {	
				for (size_t k = 0; k < K; ++k) {	
					transposed[r*S*C*K+s*C*K+c*K+k] = buf[r*S*K*C+s*K*C+k*C+c];
				}
			}	
		}
  }
	buf = transposed;
	shape[2] = C;
	shape[3] = K;
}
	
bool Tensor::is_same(Tensor t){
	if(sz != t.sz) return false;
	for(size_t i = 0; i < sz; i++){
		if(buf[i] != t.buf[i]) return false;
	}
	return true;
}

// Make a new tensor from buffer and put the tensor into map. Advance buffer pointer by size.
void register_weight(std::map<std::string, Tensor>& weights, float* (*buf), std::string name, std::vector<size_t> shape) {
  Tensor tensor(*buf, shape);
  weights[name] = tensor;
  *buf += tensor.sz;
}

// Put all predefined weights into map. Order should not be changed.
std::map<std::string, Tensor> register_weights(float* weight_buf) {
  std::map<std::string, Tensor> weights;
  // auto generated
  register_weight(weights, &weight_buf, "generator/decoder_1/conv2d_transpose/bias", {3});
  register_weight(weights, &weight_buf, "generator/decoder_1/conv2d_transpose/kernel", {4, 4, 3, 128});
  register_weight(weights, &weight_buf, "generator/decoder_2/batch_normalization/beta", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/batch_normalization/gamma", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/batch_normalization/moving_mean", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/batch_normalization/moving_variance", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/conv2d_transpose/bias", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/conv2d_transpose/kernel", {4, 4, 64, 256});
  register_weight(weights, &weight_buf, "generator/decoder_3/batch_normalization/beta", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/batch_normalization/gamma", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/batch_normalization/moving_mean", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/batch_normalization/moving_variance", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/conv2d_transpose/bias", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/conv2d_transpose/kernel", {4, 4, 128, 512});
  register_weight(weights, &weight_buf, "generator/decoder_4/batch_normalization/beta", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/batch_normalization/gamma", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/batch_normalization/moving_mean", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/batch_normalization/moving_variance", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/conv2d_transpose/bias", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/conv2d_transpose/kernel", {4, 4, 256, 1024});
  register_weight(weights, &weight_buf, "generator/decoder_5/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/conv2d_transpose/bias", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/conv2d_transpose/kernel", {4, 4, 512, 1024});
  register_weight(weights, &weight_buf, "generator/decoder_6/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/conv2d_transpose/bias", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/conv2d_transpose/kernel", {4, 4, 512, 1024});
  register_weight(weights, &weight_buf, "generator/decoder_7/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/conv2d_transpose/bias", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/conv2d_transpose/kernel", {4, 4, 512, 1024});
  register_weight(weights, &weight_buf, "generator/decoder_8/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/conv2d_transpose/bias", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/conv2d_transpose/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_1/conv2d/bias", {64});
  register_weight(weights, &weight_buf, "generator/encoder_1/conv2d/kernel", {4, 4, 3, 64});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/beta", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/gamma", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/moving_mean", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/moving_variance", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/conv2d/bias", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/conv2d/kernel", {4, 4, 64, 128});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/beta", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/gamma", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/moving_mean", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/moving_variance", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/conv2d/bias", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/conv2d/kernel", {4, 4, 128, 256});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/conv2d/kernel", {4, 4, 256, 512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/conv2d/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/conv2d/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/conv2d/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/conv2d/kernel", {4, 4, 512, 512});
  return weights;
}

// Convert 8-bit depth images (value range [0, 255]) into floating-point ones (value range [-1, 1])
Tensor preprocess(uint8_t *in, size_t num_image) {
  Tensor out(NULL, {num_image, 256, 256, 3});
  for (size_t i = 0; i < out.sz; ++i) {
    out.buf[i] = in[i] / 255.0f * 2 - 1;
  }
  return out;
}

// Inverse of preprocess
void postprocess_one_image(Tensor input, uint8_t *out, size_t idx) {
  // input shape = (height, width, channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  for (size_t i = 0; i < H * W * C; ++i) {
    float x = (input.buf[i] + 1) / 2 * 255;
    out[idx * (H * W * C) + i] = x < 0 ? 0 : (x > 255 ? 255 : x);
  }
}

// Pick single image from images
void get_one_image(Tensor input, Tensor &output, size_t idx) {
  // input shape = (num_image, height, width, channels)
  // output shape = (height, width, channels)
  size_t H = input.shape[1], W = input.shape[2], C = input.shape[3];
  output.alloc_once({H, W, C});
  for (size_t i = 0; i < H * W * C; ++i) {
    output.buf[i] = input.buf[idx * H * W * C + i];
  }
}

void conv2d_gpu(Tensor input, Tensor filter, Tensor bias, Tensor &output, int thread, int iter){
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  size_t R = filter.shape[0], S = filter.shape[1], K = filter.shape[3];
  size_t OH = H/2, OW = W/2;

  output.alloc_once({OH, OW, K});
	conv(input.buf,filter.buf,output.buf,bias.buf,int(H),int(W),int(C),int(R),int(S),int(K),int(OH),int(OW),thread,iter);
}

// Transposed convolution (2-dimension, stride = 2, pad = 1)
void conv2d_transposed1(Tensor input, Tensor filter, Tensor bias, Tensor &output) {
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  size_t R = filter.shape[0], S = filter.shape[1]; // K = filter.shape[2] = 3
  const size_t stride = 2, pad = 1;
  size_t OH = H * stride, OW = W * stride;
  output.alloc_once({OH, OW, 3});

	for (size_t oh = 0; oh < OH; ++oh) {
		for (size_t ow = 0; ow < OW; ++ow) {
			float x = bias.buf[0];
			float y = bias.buf[1];
			float z = bias.buf[2];
			for (size_t r = (oh+pad)%stride; r < R; r+=stride) {
				for (size_t s = (ow+pad)%stride; s < S; s+=stride) {
					size_t ih = (oh-r+pad)/stride;
					size_t iw = (ow-s+pad)/stride;
					if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
					size_t temp = r*S*3*C+s*3*C;
					for (size_t c = 0; c < C; ++c) {
						float ii = input.buf[ih*W*C+iw*C+c];
						x += ii * filter.buf[temp+c];
						y += ii * filter.buf[temp+C+c];
						z += ii * filter.buf[temp+2*C+c];
					}
				}
			}
			// output (oh, ow, k)
			output.buf[oh*OW*3+ow*3] = x;
			output.buf[oh*OW*3+ow*3+1] = y;
			output.buf[oh*OW*3+ow*3+2] = z;
		}
	}
}

// Transposed convolution (2-dimension, stride = 2, pad = 1)
void conv2d_transposed_gpu(Tensor input, Tensor filter, Tensor bias, Tensor &output, int thread, int iter) {
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  size_t R = filter.shape[0], S = filter.shape[1], K = filter.shape[3];
  size_t OH = H*2, OW = W*2;

  output.alloc_once({OH, OW, K});
	conv_t(input.buf,filter.buf,output.buf,bias.buf,int(H),int(W),int(C),int(R),int(S),int(K),int(OH),int(OW),thread,iter);
}

// Leaky ReLU
void leaky_relu(Tensor input, Tensor &output, float alpha) {
  // input shape = (height, width, channels)
  // output shape = (height, width, channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  output.alloc_once({H, W, C});
  for (size_t i = 0; i < H * W * C; ++i) {
    output.buf[i] = input.buf[i] >= 0 ? input.buf[i] : alpha * input.buf[i];
  }
}

// ReLU
void relu(Tensor input, Tensor &output) {
  // input shape = (height, width, channels)
  // output shape = (height, width, channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  output.alloc_once({H, W, C});
  for (size_t i = 0; i < H * W * C; ++i) {
    output.buf[i] = input.buf[i] >= 0 ? input.buf[i] : 0;
  }
}

// Batch normalization (channel-wise)
void batchnorm(Tensor input, Tensor scale, Tensor offset, Tensor &output) {
  // input shape = (height, width, channels)
  // scale shape = (channels)
  // offset shape = (channels)
  // output shape = (height, width, channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  output.alloc_once({H, W, C});
  for (size_t c = 0; c < C; ++c) {
    float sum = 0;
    for (size_t h = 0; h < H; ++h) {
      for (size_t w = 0; w < W; ++w) {
        float ii = input.buf[h * W * C + w * C + c];
        sum += ii;
      }
    }
    float mean = sum / (H * W);

    float sqsum = 0;
    for (size_t h = 0; h < H; ++h) {
      for (size_t w = 0; w < W; ++w) {
        float ii = input.buf[h * W * C + w * C + c];
        sqsum += (ii - mean) * (ii - mean);
      }
    }
    float variance = sqsum / (H * W);

    const float epsilon = 1e-5;
    for (size_t h = 0; h < H; ++h) {
      for (size_t w = 0; w < W; ++w) {
        size_t idx = h * W * C + w * C + c;
        output.buf[idx] = offset.buf[c] + (input.buf[idx] - mean) * scale.buf[c] / sqrtf(variance + epsilon);
      }
    }
  }
}

// Concatenation (along channel dimension)
void concat(Tensor input0, Tensor input1, Tensor &output) {
  // input0 shape = (height, width, channels0)
  // input1 shape = (height, width, channels1)
  // output shape = (height, width, channels0 + channels1)
  size_t H = input0.shape[0], W = input0.shape[1], C0 = input0.shape[2];
  size_t C1 = input1.shape[2];
  output.alloc_once({H, W, C0 + C1});
  for (size_t h = 0; h < H; ++h) {
    for (size_t w = 0; w < W; ++w) {
      for (size_t c = 0; c < C0; ++c) {
        output.buf[h * W * (C0 + C1) + w * (C0 + C1) + c] = input0.buf[h * W * C0 + w * C0 + c];
      }
      for (size_t c = 0; c < C1; ++c) {
        output.buf[h * W * (C0 + C1) + w * (C0 + C1) + (C0 + c)] = input1.buf[h * W * C1 + w * C1 + c];
      }
    }
  }
}

// Elementwise tanh
void elem_tanh(Tensor input, Tensor &output) {
  // input shape = (height, width, channels)
  // output shape = (height, width, channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  output.alloc_once({H, W, C});
  for (size_t i = 0; i < H * W * C; ++i) {
    output.buf[i] = tanhf(input.buf[i]);
  }
}
