//gupMD5.H
/**
*GPUƥ��MD5�㷨
*/

#include "stdafx.h"
#include "deviceMemoryDef.h"

/**
GPU��ʼ��
������
	targetDigest ����
	searchScope �������ַ���
*/
void initGPU(string targetDigest, string searchScope);

/*
GPU��������
������
	d_startSearchWord ���ÿ���̵߳���ʼ�����ռ�
	useThreadNum ʵ��ʹ�õ��߳�����
	charsetLength �����ռ䳤��
	size ��ǰ��������
	d_isFound �������
	d_message ����
*/
__global__ void searchMD5(float*, float, size_t, size_t, size_t*, uchar*);

