#include <stdlib.h>
#include <stdio.h>
#include <boost/timer.hpp>

__global__
void rotate(unsigned char *A,unsigned char *res,int N){
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x < N && y < N){
    res[y + (N - 1 - x) * N] = A[x + y * N];
  }
}

void rotateCPU(unsigned char *A,unsigned char *res,int N){
  for(int i = 0; i < N; ++i){
    for(int j = 0; j < N; ++j){
      res[i + (N - 1 - j) * N] =  A[j + i * N];
    }
  }
}

void print(unsigned char *A,int N){
  for(int i = 0; i < N; ++i){
    for(int j = 0; j < N; ++j){
      printf("%d ",A[j + i * N]);
    }
    printf("\n");
  }
}

int main(int argc, char *argv[])
{
  const int N = atoi(argv[1]);
  const int memsize = sizeof(unsigned char) * N * N;
  unsigned char *A = (unsigned char*)malloc(memsize);
  unsigned char *res = (unsigned char*)malloc(memsize);
  unsigned char *res2 = (unsigned char*)malloc(memsize);
  for(int i = 0; i < N; ++i){
    for(int j = 0; j < N; ++j){
      unsigned char n = rand() % 255;
      A[j + i * N] = n;
    }
  }
  boost::timer t;
  rotateCPU(A,res,N);
  printf("rotate cpu took %f seconds.\n", t.elapsed());
  
  unsigned char *dA,*dres;
  cudaError_t error;  
  error = cudaMalloc((void **)&dA, memsize);
  if (error != cudaSuccess)
    {
      puts("cant allocate dA.");
      exit(EXIT_FAILURE);
    }
  
  error = cudaMalloc((void **)&dres, memsize);
  if (error != cudaSuccess)
    {
      puts("cant allocate dres.");      
      exit(EXIT_FAILURE);
    }
  
  error =  cudaMemcpy(dA,A,memsize,cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
    {
      exit(EXIT_FAILURE);
    }
  
  error = cudaMemcpy(dres,res2,memsize,cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
    {
      exit(EXIT_FAILURE);
    }
  
  const int block_width = 16;
  dim3 threadNum(block_width, block_width);  
  dim3 blockNum(N/block_width, N/block_width);
  t.restart();
  rotate<<<blockNum, threadNum>>>(dA,dres,N);
  printf("gpu version took %f seconds.\n", t.elapsed());  
  error = cudaMemcpy(res2,dres,memsize,cudaMemcpyDeviceToHost);
  if (error != cudaSuccess)
    {
      exit(EXIT_FAILURE);
    }

  // print(res);
  // printf("\n");
  for(int i = 0; i < N*N; ++i){
    if(res[i] != res2[i])
      printf("WRONG!!\n");
  }
  
  cudaFree(dA);
  cudaFree(dres);
  free(A);
  free(res);
  free(res2);
  return 0;
}
