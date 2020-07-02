Pix2Pix 모델 추론 가속

딥러닝 모델의 추론에서 대부분의 시간을 차지하는 것은 convolution 연산이다. convolution 연산을 가속시키기 위한 방법으로 image to column 방법을 이용했다. image to column 방식은 convolution 연산의 input matrix를 하나의 큰 matrix로 바꾼 뒤, matrix multlication 연산을 통해 convolution 연산을 해결하는 방법이다. matrix multiplication은 규칙적인 memory access를 통해 연산이 이루어지므로 실행 속도가 빠르고 최적화하기 쉽기 때문에 해당 방법을 채택했다.
transposed convolution 연산도 같은 방법으로 해결하기 위해 convolution 연산으로 치환해서 해결했다. 치환 방법으로는 input matrix의 간 행과 열 사이에 zero padding을 하여, input matrix를 2배로 늘려주는 연산을 진행한 뒤, C 와 K dimension을 transpose 한 filter를 이용했다. 정리하자면 아래와 같다.
(1) Convolution = ‘image to column’ + ‘matrix multiplication’ + ‘add bias’
(2) Transposed Convolution = ‘inside padding’ + ‘image to column’
      + ‘matrix multiplication’ + ‘add bias’ (using transposed filter)
위의 병렬화 방법은 convolution 연산을 gpu에서 빨리 돌리기 위해 matrix multiplication 문제로 치환하고자 사용한 방법이다. 따라서 gpu를 사용하지 않은 A 에서는 사용하지 않았다.
A. Use CPU only
cpu만 사용하는 A 버전에서는 pthread 와 openMP를 이용해서 병렬화를 진행했다. 각 image를 16개의 pthread를 이용해서 16개의 thread가 병렬적으로 하나의 이미지를 처리하도록 했다. 또한 각 convolution 연산들을 가속화시키기 위해 openMP를 적용했고, register blocking 을 통해 most inner loop에서의 memory 연산의 비율을 줄이는 방법으로 최적화를 했다.
실행 결과는 1600 image 기준으로 5.5 img/sec 이다.
B. CPU + single GPU
gpu를 사용하는 B 버전에서는 cuda를 이용했다. 앞서 설명한 방법대로 convolution 연산을 진행했고, 각 operation에 대한 커널함수를 작성한 다음 “conv.cu” 파일에 추가했다. 다만 gpu가 1대뿐인 B 버전에서는 모든 convolution 연산에 대해 gpu를 사용하면 오히려 속도가 안좋았다. 따라서 8번의 convolution과 8번의 transposed convolution 중 각 3번만 gpu를 사용하고 나머지는 cpu를 이용해 연산을 진행했다.
실행 결과는 1600 image 기준으로 12.4 img/sec 이다.
C. CPU + 4 GPU
    gpu를 4대 사용하는 C 버전에서는 16개의 thread 연산을 각 gpu에 4개씩 할당했다. CUDA 커널을 호출할 때마다 thread id를 확인하여 cudaSetDevice를 통해 각 thread에 할당된 gpu로 실행시켰다. 또한 B 버전 과는 달리, 한 gpu당 실행시켜야 하는 thread 숫자가 줄었기 때문에 transposded convolution의 마지막 layer를 제외한 모든 연산을 gpu로 돌렸다.
    실행 결과는 1600 image 기준으로 37.5 img/sec 이다.
D. (CPU + 4 GPU) * 4 nodes
    4개의 node를 사용하는 D 버전에서는 mpi를 사용했다. mpi를 이용해 image input을 4개로 나누어 각 node가 연산을 진행하는 방식으로 병령화를 진행했다. 이 외의 병렬화 기법은 C 버전과 동일하다.
    실행 결과는 1600 image 기준으로 72 img/sec 이다.
