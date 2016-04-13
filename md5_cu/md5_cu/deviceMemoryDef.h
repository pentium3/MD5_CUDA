/**
*�豸�����Դ��������
*/
#include "stdafx.h"

#ifndef __DEVICE_MEMORY_DEF_H__
#define __DEVICE_MEMORY_DEF_H__

// �ȶԵ�Ŀ������(a b c d)��ֻ����GPU�豸����
#define NUM_DIGEST_SIZE 4
__device__ __constant__ uint d_targetDigest[NUM_DIGEST_SIZE];

// �����ַ����� ���� a-z A-Z 0-9 ~!@#$%^&*()_+-=[]\|;:"'<,>.?/ 
#define NUM_POWER_SYMBOLS 96
__device__ __constant__ uchar  d_powerSymbols[NUM_POWER_SYMBOLS];

//�������ȵ��������
#define NUM_POWER_VALUES 16
__constant__ float d_powerValues[NUM_POWER_VALUES];

#endif // !__DEVICE_MEMORY_DEF_H__