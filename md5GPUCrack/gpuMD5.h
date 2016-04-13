//gupMD5.H
/**
*GPU匹配MD5算法
*/

#include "stdafx.h"
#include "deviceMemoryDef.h"

/**
GPU初始化
参数：
	targetDigest 密文
	searchScope 搜索的字符集
*/
void initGPU(string targetDigest, string searchScope);

/*
GPU运算搜索
参数：
	d_startSearchWord 存放每个线程的起始搜索空间
	useThreadNum 实际使用的线程数量
	charsetLength 搜索空间长度
	size 当前搜索长度
	d_isFound 搜索结果
	d_message 明文
*/
__global__ void searchMD5(float*, float, size_t, size_t, size_t*, uchar*);

