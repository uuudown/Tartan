#include "turH.h"

void kernelCheck( cudaError_t error, const char* function, int a=1)
{
	if(error !=cudaSuccess)
	{
		const char* error_string= cudaGetErrorString(error);
		printf("\n error  %s : %s \n", function, error_string);
		exit(1);
	}
		
	
	if(a!=0)
	{
	error= cudaGetLastError();			
		if(error !=cudaSuccess)
		{
			const char* error_string= cudaGetErrorString(error);
			printf("\n error  %s : %s \n", function, error_string);
			exit(1);
		}
	}

	return;
}



