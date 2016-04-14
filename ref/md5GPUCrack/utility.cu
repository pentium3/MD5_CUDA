//utility.cu
#include "utility.h"

void printCudaError(cudaError_t error, string msg, string fileName, int line)
{
	cout<<msg<<"�������룺"<<error<<"���ļ�("<<fileName<<")������("<<line<<")."<<endl;
	exit(EXIT_FAILURE);
}