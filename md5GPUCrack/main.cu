/**
**主程序入口
**/

#include "stdafx.h"
#include "gpuMD5.h"
#include "findMessage.h"

/**
*主函数
*/
int main()
{
	//123456 = e10adc3949ba59abbe56e057f20f883e
	// 999999 = 52c69e3a57331081823331c4e69d3f2e
	//adc12d4 = 32f3db39aa85fac25c19c0c8b555dc83
	string target = "5cb8158e007f9de55a6497ffde212bd4";
	//0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
	string searchScope = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";	//搜索空间

	initGPU(target, searchScope);

	size_t startNum = 5, endNum = 8;
	pair<bool, string> result = findMessage(startNum, endNum, searchScope);

	if(result.first){
		cout<<"找到明文："<<result.second<<endl;
	}else{
		cout<<"未搜索到相应明文."<<endl;
	}
	return 0;
}