/**
**���������
**/

#include "stdafx.h"
#include "gpuMD5.h"
#include "findMessage.h"

/**
*������
*/
int main()
{
	//123456 = e10adc3949ba59abbe56e057f20f883e
	// 999999 = 52c69e3a57331081823331c4e69d3f2e
	//adc12d4 = 32f3db39aa85fac25c19c0c8b555dc83
	string target = "5cb8158e007f9de55a6497ffde212bd4";
	//0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
	string searchScope = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";	//�����ռ�

	initGPU(target, searchScope);

	size_t startNum = 5, endNum = 8;
	pair<bool, string> result = findMessage(startNum, endNum, searchScope);

	if(result.first){
		cout<<"�ҵ����ģ�"<<result.second<<endl;
	}else{
		cout<<"δ��������Ӧ����."<<endl;
	}
	return 0;
}