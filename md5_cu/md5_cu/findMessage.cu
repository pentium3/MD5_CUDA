//findMessage.cpp

#include "stdafx.h"
#include "deviceMemoryDef.h"
#include "gpuMD5.h"

/**
��������
min��������С����
max��������󳤶�
searchScope�������ռ�
*/
pair<bool, string> findMessage(size_t min, size_t max, string searchScope) {
	bool isFound = false;
	size_t h_isFound = -1; size_t * d_isFound;    //���������ʶ 
	uchar* d_message; uchar h_message[16];	//���ģ����֧�ֳ���Ϊ16
	string message = "";
	
	//GoForce GT650M �Ƚ���������ã�1024*1024
	int nBlocks = 512;
	int nThreadsPerBlock = 512;
	size_t nTotalThreads = nBlocks * nThreadsPerBlock; // ���߳���
	size_t charsetLength = searchScope.length();  //�����ռ��ַ�������

	cudaError_t error;
	error = cudaMalloc((void**)&d_isFound, sizeof(size_t));
	if (error != cudaSuccess){
		printCudaError(error,"���䣨���������ʶ���Դ����", __FILE__, __LINE__);
    }
	error = cudaMemcpy(d_isFound, &h_isFound,  sizeof(size_t), cudaMemcpyHostToDevice);
	if (error != cudaSuccess){
		printCudaError(error,"���������������ʶ�����Դ����", __FILE__, __LINE__);
    }
	error = cudaMalloc((void**)&d_message, 16 * sizeof(uchar));
	if (error != cudaSuccess){
		printCudaError(error,"����������������ģ��Դ����", __FILE__, __LINE__);
    }

	//����ÿ���̵߳�������ʼ��ַ
	float* h_startNumbers = new float[nTotalThreads];
	float* d_startNumbers;
	error = cudaMalloc((void**)&d_startNumbers, nTotalThreads * sizeof(float));
	if (error != cudaSuccess){
		printCudaError(error,"�����̵߳�������ʼ��ַ����", __FILE__, __LINE__);
    }

	for (size_t size = min; size <= max; ++size) {
		cout<<"��ǰ�������ȣ�"<<size<<endl;
		float maxValue = pow((float)charsetLength, (float)size);  //���ƥ����
		float nIterations = ceil(maxValue / (nBlocks * nThreadsPerBlock));//ÿ���̷߳����������,��ÿ���߳���Ҫ�����ĸ���
		for (size_t i = 0; i != nTotalThreads; ++i) {
		  h_startNumbers[i] = i * nIterations;
		}
		error = cudaMemcpy(d_startNumbers, h_startNumbers, nTotalThreads * sizeof(float), cudaMemcpyHostToDevice);
		if (error != cudaSuccess){
			printCudaError(error,"���� �̵߳�������ʼ��ַ ���Դ����", __FILE__, __LINE__);
		}
		clock_t start = clock();
		//��ʼ����
		searchMD5<<< nBlocks, nThreadsPerBlock >>>(d_startNumbers, 
			nIterations, charsetLength, size, d_isFound, d_message);
    
		cudaThreadSynchronize();

		cout<<"��ʱ��"<<(clock()-start)/CLK_TCK<<endl;

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


	//�ͷ��ڴ���Դ�
	cudaFree(d_targetDigest);
	cudaFree(d_powerSymbols);
	cudaFree(d_powerValues);
	cudaFree(d_isFound);
	cudaFree(d_message);
	cudaFree(d_startNumbers);

	delete(h_startNumbers);
	cout<<"�ͷ��ڴ����..."<<endl;
	return make_pair(isFound, message);
}