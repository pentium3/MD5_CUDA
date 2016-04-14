
#include "stdafx.h"
#include "gpuMD5.h"
#include "findMessage.h"
#include "queryDevice.h"

/**
*������
*/
int main()
{
	string target = "edbd0effac3fcc98e725920a512881e0";

	string searchScope = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";	//�����ռ�

	queryDevice();
	initGPU(target, searchScope);

	size_t startNum = 5, endNum = 8;
	pair<bool, string> result = findMessage(startNum, endNum, searchScope);

	if(result.first){
		cout<<"�ҵ����ģ�"<<result.second<<endl;
	}else{
		cout<<"δ��������Ӧ����."<<endl;
	}

	getchar();
	return 0;
}