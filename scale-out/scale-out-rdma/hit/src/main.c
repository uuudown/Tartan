
#include "turH.h"

//GLOBALS

int RANK;
int SIZE;

int IGLOBAL;
int NXSIZE;
int NYSIZE;

cudaError_t RET;

float2* AUX;

//CUDA FUNCTIONS

int main(int argc, char *argv[])
{ 

	//MPI initialize
	MPI_Init(NULL,NULL);
	H5open();
		
	MPI_Comm_size(MPI_COMM_WORLD, &SIZE);
 	MPI_Comm_rank(MPI_COMM_WORLD, &RANK);

	//ERROR
	if(NX%SIZE!=0 || NY%SIZE!=0){
	printf("\nError: no se puede partir dominio\n");
	exit(1);
	}	

	//Set sizes and positions
	NXSIZE=NX/SIZE;
	NYSIZE=NY/SIZE;
	IGLOBAL=NXSIZE*RANK;
	
	//printf("\n(RANK,IGLOBAL)=(%d,%d)\n",RANK,IGLOBAL);
	
	//CUDA SET UP
	setUp();
	

	//start
	starSimulation();
	
	H5close();
	MPI_Finalize();

return 0;

}


