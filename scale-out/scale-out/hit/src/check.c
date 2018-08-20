
#include "turH.h"

void cudaCheck( cudaError_t error, const char* function)
{
	if(error !=cudaSuccess)
	{
		const char* error_string= cudaGetErrorString(error);
		printf("\n error  %s : %s \n", function, error_string);
		exit(1);
	}
		

	return;
}



void mpiCheck( int error, const char* function)
{
	if(error !=0)
	{
		//printf("\n error_MPI %s \n",(char*)function);
		printf("error_mpi");		
		exit(1);
	}
		
	

	return;
}



