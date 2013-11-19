#include <iostream>
#include <kernel.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

int
main(int argc, char **argv)
{
    callKernel(argc, argv);
    cudaDeviceReset();
    return 0;
}
