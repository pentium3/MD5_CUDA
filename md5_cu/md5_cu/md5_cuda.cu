#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"cstring"
#include"cstdio"

typedef unsigned long int UINT4;

/* Data structure for MD5 (Message Digest) computation */
typedef struct {
  UINT4 i[2];                   /* number of _bits_ handled mod 2^64 */
  UINT4 buf[4];                                    /* scratch buffer */
  unsigned char in[64];                              /* input buffer */
  unsigned char digest[16];     /* actual digest after MD5Final call */
} MD5_CTX;


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void cuMDString()
{
	MD5_CTX mdContext;
	
	
}


void MDString(unsigned char *inString)
{
  const char* p = (const char*)(char*)inString;
  unsigned int len = strlen (p);

}

