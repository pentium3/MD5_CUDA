//gupMD5.cu
#include "gpuMD5.h"

/**    MD5ɢ�к���    **/
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z))) 
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))
#define FF(a, b, c, d, x, s, ac) \
{(a) += F ((b), (c), (d)) + (x) + (uint)(ac); \
	(a) = ROTATE_LEFT ((a), (s)); \
	(a) += (b); \
}
#define GG(a, b, c, d, x, s, ac) \
{(a) += G ((b), (c), (d)) + (x) + (uint)(ac); \
	(a) = ROTATE_LEFT ((a), (s)); \
	(a) += (b); \
}
#define HH(a, b, c, d, x, s, ac) \
{(a) += H ((b), (c), (d)) + (x) + (uint)(ac); \
	(a) = ROTATE_LEFT ((a), (s)); \
	(a) += (b); \
}
#define II(a, b, c, d, x, s, ac) \
{(a) += I ((b), (c), (d)) + (x) + (uint)(ac); \
	(a) = ROTATE_LEFT ((a), (s)); \
	(a) += (b); \
}

// char ת��Ϊ uchar
uchar c2c (char c){ return (uchar)((c > '9') ? (c - 'a' + 10) : (c - '0')); }

void initGPU(string targetDigest, string searchScope)
{
	uint h_targetDigest[4];	//�ڴ��еıȶ�Ŀ��
	for(int c=0;c<targetDigest.size();c+=8) {
		uint x = c2c(targetDigest[c]) <<4 | c2c(targetDigest[c+1]); 
		uint y = c2c(targetDigest[c+2]) << 4 | c2c(targetDigest[c+3]);
		uint z = c2c(targetDigest[c+4]) << 4 | c2c(targetDigest[c+5]);
		uint w = c2c(targetDigest[c+6]) << 4 | c2c(targetDigest[c+7]);
		h_targetDigest[c/8] = w << 24 | z << 16 | y << 8 | x;
	}
	cout<<"h_targetDigest[0]="<<h_targetDigest[0]<<endl;
	cout<<"h_targetDigest[1]="<<h_targetDigest[1]<<endl;
	cout<<"h_targetDigest[2]="<<h_targetDigest[2]<<endl;
	cout<<"h_targetDigest[3]="<<h_targetDigest[3]<<endl;

	float charsetLen = searchScope.length();
	cudaError_t error;
	//��Ŀ��ɢ������ �������������豸�����洢��
	error = cudaMemcpyToSymbol(d_targetDigest, h_targetDigest, NUM_DIGEST_SIZE * sizeof(uint));
	if (error != cudaSuccess){
		printCudaError(error,"������Ŀ��ɢ�����飩�����豸�����洢��������", __FILE__, __LINE__);
    }

	uchar h_powerSymbols[NUM_POWER_SYMBOLS];
	for (size_t i = 0; i != charsetLen; ++i)
	{
		h_powerSymbols[i] = searchScope[i];
	}
	// ���������ռ��ַ����鵽 �豸�����洢��
	error = cudaMemcpyToSymbol(d_powerSymbols, h_powerSymbols, NUM_POWER_SYMBOLS * sizeof(uchar));
	if (error != cudaSuccess){
		printCudaError(error,"�����������ռ��ַ����飩�����豸�����洢������", __FILE__, __LINE__);
    }

	float h_powerValues[NUM_POWER_VALUES];
	for (size_t i = 0; i != NUM_POWER_VALUES; ++i)
	h_powerValues[i] = pow(charsetLen, (float)(NUM_POWER_VALUES - i - 1));
	cudaMemcpyToSymbol(d_powerValues, h_powerValues, NUM_POWER_VALUES * sizeof(float));

}

__global__ void searchMD5(float* d_startNumbers, float nIterations, size_t charsetLength, size_t size, size_t* d_isFound, uchar* message){
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	float maxValue = powf(__uint2float_rz(charsetLength), __uint2float_rz(size));//��������
  
	uint in[17];
  
	for (size_t i = 0; i != 17; ++i){
		in[i] = 0x00000000;
	}
	in[14] = size << 3;
	uchar* toHashAsChar = (uchar*)in;
  
	for (size_t i = 0; i != size; ++i){
		toHashAsChar[i] = d_powerSymbols[0];
	}
  
	toHashAsChar[size] = 0x80;
	float numberToConvert = d_startNumbers[idx];//��ȡ��ʼƥ���ַ
	size_t toHashAsCharIndices[17];//��¼��ǰ�߳���Ҫ������ַ����������ռ������λ��
  
	if(numberToConvert < maxValue) {
		//�õ����̵߳���ʼ������ַ
		for(size_t i = 0; i != size; ++i) {
			//�õ����߳���ʼ��ַ�ڵ�ǰ������Χ�еı��ʣ�Ȼ��ȡ��
			toHashAsCharIndices[i] = __float2uint_rz(floorf(numberToConvert / d_powerValues[NUM_POWER_VALUES - size + i]));
			//�õ��������λ��
			numberToConvert = floorf(fmodf(numberToConvert, d_powerValues[NUM_POWER_VALUES - size + i]));
		}
		/*printf("�߳�%d����ʼ������ַ��",idx);
		for (size_t i = 0; i != size; ++i){
			toHashAsChar[i] = d_powerSymbols[toHashAsCharIndices[i]];
			printf("%c",toHashAsChar[i]);
		}
		printf("\n");*/

		#pragma unroll 5
		for(float iterationsDone = 0; iterationsDone != nIterations; ++iterationsDone){
			if (*d_isFound == 1) break;

			for (size_t i = 0; i != size; ++i){
				toHashAsChar[i] = d_powerSymbols[toHashAsCharIndices[i]];//�����ַ�λ��ȡ���ַ�
			}
			//MD5 HASH
			uint h0 = 0x67452301;
			uint h1 = 0xEFCDAB89;
			uint h2 = 0x98BADCFE;
			uint h3 = 0x10325476;

			uint a = h0;
			uint b = h1;
			uint c = h2;
			uint d = h3;

			/* Round 1 */
			#define S11 7
			#define S12 12
			#define S13 17
			#define S14 22
			FF ( a, b, c, d, in[ 0], S11, 3614090360); /* 1 */
			FF ( d, a, b, c, in[ 1], S12, 3905402710); /* 2 */
			FF ( c, d, a, b, in[ 2], S13,  606105819); /* 3 */
			FF ( b, c, d, a, in[ 3], S14, 3250441966); /* 4 */
			FF ( a, b, c, d, in[ 4], S11, 4118548399); /* 5 */
			FF ( d, a, b, c, in[ 5], S12, 1200080426); /* 6 */
			FF ( c, d, a, b, in[ 6], S13, 2821735955); /* 7 */
			FF ( b, c, d, a, in[ 7], S14, 4249261313); /* 8 */
			FF ( a, b, c, d, in[ 8], S11, 1770035416); /* 9 */
			FF ( d, a, b, c, in[ 9], S12, 2336552879); /* 10 */
			FF ( c, d, a, b, in[10], S13, 4294925233); /* 11 */
			FF ( b, c, d, a, in[11], S14, 2304563134); /* 12 */
			FF ( a, b, c, d, in[12], S11, 1804603682); /* 13 */
			FF ( d, a, b, c, in[13], S12, 4254626195); /* 14 */
			FF ( c, d, a, b, in[14], S13, 2792965006); /* 15 */
			FF ( b, c, d, a, in[15], S14, 1236535329); /* 16 */
      
			/* Round 2 */
			#define S21 5
			#define S22 9
			#define S23 14
			#define S24 20
			GG ( a, b, c, d, in[ 1], S21, 4129170786); /* 17 */
			GG ( d, a, b, c, in[ 6], S22, 3225465664); /* 18 */
			GG ( c, d, a, b, in[11], S23,  643717713); /* 19 */
			GG ( b, c, d, a, in[ 0], S24, 3921069994); /* 20 */
			GG ( a, b, c, d, in[ 5], S21, 3593408605); /* 21 */
			GG ( d, a, b, c, in[10], S22,   38016083); /* 22 */
			GG ( c, d, a, b, in[15], S23, 3634488961); /* 23 */
			GG ( b, c, d, a, in[ 4], S24, 3889429448); /* 24 */
			GG ( a, b, c, d, in[ 9], S21,  568446438); /* 25 */
			GG ( d, a, b, c, in[14], S22, 3275163606); /* 26 */
			GG ( c, d, a, b, in[ 3], S23, 4107603335); /* 27 */
			GG ( b, c, d, a, in[ 8], S24, 1163531501); /* 28 */
			GG ( a, b, c, d, in[13], S21, 2850285829); /* 29 */
			GG ( d, a, b, c, in[ 2], S22, 4243563512); /* 30 */
			GG ( c, d, a, b, in[ 7], S23, 1735328473); /* 31 */
			GG ( b, c, d, a, in[12], S24, 2368359562); /* 32 */

			/* Round 3 */
			#define S31 4
			#define S32 11
			#define S33 16
			#define S34 23
			HH ( a, b, c, d, in[ 5], S31, 4294588738); /* 33 */
			HH ( d, a, b, c, in[ 8], S32, 2272392833); /* 34 */
			HH ( c, d, a, b, in[11], S33, 1839030562); /* 35 */
			HH ( b, c, d, a, in[14], S34, 4259657740); /* 36 */
			HH ( a, b, c, d, in[ 1], S31, 2763975236); /* 37 */
			HH ( d, a, b, c, in[ 4], S32, 1272893353); /* 38 */
			HH ( c, d, a, b, in[ 7], S33, 4139469664); /* 39 */
			HH ( b, c, d, a, in[10], S34, 3200236656); /* 40 */
			HH ( a, b, c, d, in[13], S31,  681279174); /* 41 */
			HH ( d, a, b, c, in[ 0], S32, 3936430074); /* 42 */
			HH ( c, d, a, b, in[ 3], S33, 3572445317); /* 43 */
			HH ( b, c, d, a, in[ 6], S34,   76029189); /* 44 */
			HH ( a, b, c, d, in[ 9], S31, 3654602809); /* 45 */
			HH ( d, a, b, c, in[12], S32, 3873151461); /* 46 */
			HH ( c, d, a, b, in[15], S33,  530742520); /* 47 */
			HH ( b, c, d, a, in[ 2], S34, 3299628645); /* 48 */
      
			/* Round 4 */
			#define S41 6
			#define S42 10
			#define S43 15
			#define S44 21
			II ( a, b, c, d, in[ 0], S41, 4096336452); /* 49 */
			II ( d, a, b, c, in[ 7], S42, 1126891415); /* 50 */
			II ( c, d, a, b, in[14], S43, 2878612391); /* 51 */
			II ( b, c, d, a, in[ 5], S44, 4237533241); /* 52 */
			II ( a, b, c, d, in[12], S41, 1700485571); /* 53 */
			II ( d, a, b, c, in[ 3], S42, 2399980690); /* 54 */
			II ( c, d, a, b, in[10], S43, 4293915773); /* 55 */
			II ( b, c, d, a, in[ 1], S44, 2240044497); /* 56 */
			II ( a, b, c, d, in[ 8], S41, 1873313359); /* 57 */
			II ( d, a, b, c, in[15], S42, 4264355552); /* 58 */
			II ( c, d, a, b, in[ 6], S43, 2734768916); /* 59 */
			II ( b, c, d, a, in[13], S44, 1309151649); /* 60 */
			II ( a, b, c, d, in[ 4], S41, 4149444226); /* 61 */
			II ( d, a, b, c, in[11], S42, 3174756917); /* 62 */
			II ( c, d, a, b, in[ 2], S43,  718787259); /* 63 */
			II ( b, c, d, a, in[ 9], S44, 3951481745); /* 64 */
      
			a += h0;
			b += h1;
			c += h2;
			d += h3;

			//���ɢ��ֵ�Ƿ�ƥ��
			if (a == d_targetDigest[0] && b == d_targetDigest[1] && c == d_targetDigest[2] && d == d_targetDigest[3]){
				*d_isFound = 1;
				for (size_t i = 0; i != size; ++i){//ȡ�����
					message[i] = toHashAsChar[i];
				}
			}else {
				size_t i = size - 1;
				bool incrementNext = true;//�Ƿ���������޷��������λ
				while (incrementNext){//�������޷���λ�����ָ���Ƶ�ǰһλ��λ����[115]->[121]
					if (toHashAsCharIndices[i] < (charsetLength - 1)) {
						++toHashAsCharIndices[i];
						incrementNext = false;
					}
					else {
						if (toHashAsCharIndices[i] >= charsetLength) {
							*d_isFound = 3;
						}
						toHashAsCharIndices[i] = 0;
						if (i == 0) {
							incrementNext = false;
						}
						else {
							--i;
						}
					}
				}
			}
		}
	}
}