//utility.h
/** 
*ʵ�ù�����
*/
#ifndef __UTILITY_H__
#define __UTILITY_H__
#include "stdafx.h"

#define CUDA_MALLOC_ERROR 1  //CUDA�ڴ�������
#define CUDA_MEM_CPY_ERROR 2  //CUDA�ڴ濽������

/*
*��ӡCUDA������Ϣ
*����:
	error ������
	msg ������Ϣ
	errorType ��������
	fileName ������ļ���
	line �������ļ��е�����
*/
void printCudaError(cudaError_t error, string msg,string fileName, int line);

#endif // !__UTILITY_H__