#include "turH.h"

void memoryInfo(void)
{
	size_t free;
	size_t total;
	
	cudaCheck(cudaMemGetInfo (&free,&total),"MemInfo11");
	
	printf("\n");
	printf("\nRANK=%d\n",RANK);
	printf("\nGPU total memory = % .2f MB\n",(float)total/1e6);
	printf("\nGPU free  memory = % .2f MB\n",(float)free/1e6);

}




