
#include "stdafx.h"
#include "gpuMD5.h"
#include "findMessage.h"
#include "queryDevice.h"

/**
*主函数
*/
int main()
{
	string target = "edbd0effac3fcc98e725920a512881e0";

	string searchScope = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";	//搜索空间

	queryDevice();
	initGPU(target, searchScope);

	size_t startNum = 5, endNum = 8;
	pair<bool, string> result = findMessage(startNum, endNum, searchScope);

	if(result.first){
		cout<<"找到明文："<<result.second<<endl;
	}else{
		cout<<"未搜索到相应明文."<<endl;
	}

	getchar();
	return 0;
}