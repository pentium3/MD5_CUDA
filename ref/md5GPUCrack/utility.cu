//utility.cu
#include "utility.h"

void printCudaError(cudaError_t error, string msg, string fileName, int line)
{
	cout<<msg<<"，错误码："<<error<<"，文件("<<fileName<<")，行数("<<line<<")."<<endl;
	exit(EXIT_FAILURE);
}