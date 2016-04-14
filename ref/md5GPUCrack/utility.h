//utility.h
/** 
*实用工具类
*/
#ifndef __UTILITY_H__
#define __UTILITY_H__
#include "stdafx.h"

#define CUDA_MALLOC_ERROR 1  //CUDA内存分配错误
#define CUDA_MEM_CPY_ERROR 2  //CUDA内存拷贝错误

/*
*打印CUDA错误信息
*参数:
	error 错误码
	msg 错误信息
	errorType 错误类型
	fileName 出错的文件名
	line 错误在文件中的行数
*/
void printCudaError(cudaError_t error, string msg,string fileName, int line);

#endif // !__UTILITY_H__