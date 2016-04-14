#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
//初始化CUDA

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}
// end of GPU Architecture definitions


bool InitCUDA()
{
	int count=0;
	printf("Start to detecte devices.........\n");//显示检测到的设备数
	cudaGetDeviceCount(&count);//检测计算能力大于等于1.0 的设备数
	if(count == 0){
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	printf("%d device/s detected.\n",count);//显示检测到的设备数
	int i;
	for(i = 0; i < count; i++){//依次验证检测到的设备是否支持CUDA
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {//获得设备属性并验证是否正确
			if(prop.major >= 1)//验证主计算能力，即计算能力的第一位数是否大于1
			{
				printf("Device %d: %s supports CUDA %d.%d.\n",i+1,prop.name,prop.major,prop.minor);//显示检测到的设备支持的CUDA 版本

				int driverVersion = 0, runtimeVersion = 0;
				cudaDriverGetVersion(&driverVersion);
				cudaRuntimeGetVersion(&runtimeVersion);
				printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
				printf("  CUDA Capability Major/Minor version number:    %d.%d\n", prop.major, prop.minor);
				printf("\n");

				printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
						prop.multiProcessorCount,
						_ConvertSMVer2Cores(prop.major,prop.minor),
						prop.multiProcessorCount*_ConvertSMVer2Cores(prop.major,prop.minor));
				printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n", prop.clockRate * 1e-3f, prop.clockRate * 1e-6f);
				printf("  Memory Clock rate:                             %.0f Mhz\n", prop.memoryClockRate * 1e-3f);
				printf("\n");

				printf("  Memory Bus Width:                              %d-bit\n",   prop.memoryBusWidth);
				if (prop.l2CacheSize)
				{
					printf("  L2 Cache Size:                                 %d bytes\n", prop.l2CacheSize);
				}
				printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",(float)prop.totalGlobalMem/1048576.0f, (unsigned long long) prop.totalGlobalMem);
				printf("  Total amount of constant memory:               %lu bytes\n", prop.totalConstMem);
				printf("  Total amount of shared memory per block:       %lu bytes\n", prop.sharedMemPerBlock);
				printf("  Total number of registers available per block: %d\n", prop.regsPerBlock);
				printf("  Warp size:                                     %d\n", prop.warpSize);
				printf("\n");

				printf("  Maximum number of threads per multiprocessor:  %d\n", prop.maxThreadsPerMultiProcessor);
				printf("  Maximum number of threads per block:           %d\n", prop.maxThreadsPerBlock);
				printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
					   prop.maxThreadsDim[0],
					   prop.maxThreadsDim[1],
					   prop.maxThreadsDim[2]);
				printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
					   prop.maxGridSize[0],
					   prop.maxGridSize[1],
					   prop.maxGridSize[2]);
				printf("\n");

				printf("  Maximum memory pitch:                          %lu bytes\n", prop.memPitch);
				printf("  Texture alignment:                             %lu bytes\n", prop.textureAlignment);
				printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (prop.deviceOverlap ? "Yes" : "No"), prop.asyncEngineCount);
				printf("  Run time limit on kernels:                     %s\n", prop.kernelExecTimeoutEnabled ? "Yes" : "No");
				printf("  Integrated GPU sharing Host Memory:            %s\n", prop.integrated ? "Yes" : "No");
				printf("  Support host page-locked memory mapping:       %s\n", prop.canMapHostMemory ? "Yes" : "No");
				printf("  Alignment requirement for Surfaces:            %s\n", prop.surfaceAlignment ? "Yes" : "No");
				printf("  Device has ECC support:                        %s\n", prop.ECCEnabled ? "Enabled" : "Disabled");
				break;
			}
		}
	}
	if(i == count) {//没有支持CUDA1.x 的设备
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}
	cudaSetDevice(i);//设置设备为主叫线程的当前设备
	return true;
}
//查询支持CUDA的设备
void queryDevice()
{
	if(!InitCUDA()) {//初始化失败返回系统int argc, char** argv
		return;
	}
	printf("Hello GPU! CUDA has been initialized.\n\n");
	//exit(argc ? EXIT_SUCCESS : EXIT_FAILURE);
	return;//返回系统
}