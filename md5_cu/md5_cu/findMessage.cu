//findMessage.cpp

#include "stdafx.h"
#include "deviceMemoryDef.h"
#include "gpuMD5.h"

/**
搜索明文
min：明文最小长度
max：明文最大长度
searchScope：搜索空间
*/
pair<bool, string> findMessage(size_t min, size_t max, string searchScope) {
	bool isFound = false;
	size_t h_isFound = -1; size_t * d_isFound;    //搜索结果标识 
	uchar* d_message; uchar h_message[16];	//明文，最大支持长度为16
	string message = "";
	
	//GoForce GT650M 比较优秀的设置：1024*1024
	int nBlocks = 512;
	int nThreadsPerBlock = 512;
	size_t nTotalThreads = nBlocks * nThreadsPerBlock; // 总线程数
	size_t charsetLength = searchScope.length();  //搜索空间字符数长度

	cudaError_t error;
	error = cudaMalloc((void**)&d_isFound, sizeof(size_t));
	if (error != cudaSuccess){
		printCudaError(error,"分配（搜索结果标识）显存出错", __FILE__, __LINE__);
    }
	error = cudaMemcpy(d_isFound, &h_isFound,  sizeof(size_t), cudaMemcpyHostToDevice);
	if (error != cudaSuccess){
		printCudaError(error,"拷贝（搜索结果标识）至显存出错", __FILE__, __LINE__);
    }
	error = cudaMalloc((void**)&d_message, 16 * sizeof(uchar));
	if (error != cudaSuccess){
		printCudaError(error,"分配搜索结果（明文）显存出错", __FILE__, __LINE__);
    }

	//分配每个线程的搜索起始地址
	float* h_startNumbers = new float[nTotalThreads];
	float* d_startNumbers;
	error = cudaMalloc((void**)&d_startNumbers, nTotalThreads * sizeof(float));
	if (error != cudaSuccess){
		printCudaError(error,"分配线程的搜索起始地址出错", __FILE__, __LINE__);
    }

	for (size_t size = min; size <= max; ++size) {
		cout<<"当前搜索长度："<<size<<endl;
		float maxValue = pow((float)charsetLength, (float)size);  //最大匹配数
		float nIterations = ceil(maxValue / (nBlocks * nThreadsPerBlock));//每个线程分配的任务数,即每个线程需要遍历的个数
		for (size_t i = 0; i != nTotalThreads; ++i) {
		  h_startNumbers[i] = i * nIterations;
		}
		error = cudaMemcpy(d_startNumbers, h_startNumbers, nTotalThreads * sizeof(float), cudaMemcpyHostToDevice);
		if (error != cudaSuccess){
			printCudaError(error,"拷贝 线程的搜索起始地址 到显存出错", __FILE__, __LINE__);
		}
		clock_t start = clock();
		//开始运算
		searchMD5<<< nBlocks, nThreadsPerBlock >>>(d_startNumbers, 
			nIterations, charsetLength, size, d_isFound, d_message);
    
		cudaThreadSynchronize();

		cout<<"耗时："<<(clock()-start)/CLK_TCK<<endl;

		printf("%s\n", cudaGetErrorString(cudaGetLastError()));
		cudaMemcpy(&h_isFound, d_isFound, sizeof(int), cudaMemcpyDeviceToHost);
		printf("####################### h_isFound = %d\n", h_isFound);

		if (h_isFound != -1) {
		  printf("h_isFound=%d\n", h_isFound);
		  cudaMemcpy(h_message, d_message, 16 * sizeof(uchar), cudaMemcpyDeviceToHost);
        
		  for (size_t i = 0; i != size; ++i){
			message.push_back(h_message[i]);
		  }
		  isFound = true;
		  cout << message << endl;
		  break;
		}
	}


	//释放内存和显存
	cudaFree(d_targetDigest);
	cudaFree(d_powerSymbols);
	cudaFree(d_powerValues);
	cudaFree(d_isFound);
	cudaFree(d_message);
	cudaFree(d_startNumbers);

	delete(h_startNumbers);
	cout<<"释放内存完毕..."<<endl;
	return make_pair(isFound, message);
}