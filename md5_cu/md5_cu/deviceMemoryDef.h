/**
*设备公用显存变量定义
*/
#include "stdafx.h"

#ifndef __DEVICE_MEMORY_DEF_H__
#define __DEVICE_MEMORY_DEF_H__

// 比对的目标数组(a b c d)，只能由GPU设备调用
#define NUM_DIGEST_SIZE 4
__device__ __constant__ uint d_targetDigest[NUM_DIGEST_SIZE];

// 搜索字符数组 包含 a-z A-Z 0-9 ~!@#$%^&*()_+-=[]\|;:"'<,>.?/ 
#define NUM_POWER_SYMBOLS 96
__device__ __constant__ uchar  d_powerSymbols[NUM_POWER_SYMBOLS];

//搜索长度的组合数量
#define NUM_POWER_VALUES 16
__constant__ float d_powerValues[NUM_POWER_VALUES];

#endif // !__DEVICE_MEMORY_DEF_H__