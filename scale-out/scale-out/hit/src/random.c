#include"turH.h"



int randomNumberGen(int T)
{
	
	unsigned int randomNumber;
	randomNumber=rand()%T;
	return randomNumber;

}

void seedGen(void)
{
  
	struct timeval t1;
	gettimeofday(&t1, NULL); 
//	srand(t1.tv_usec * t1.tv_sec);
      srand(1000);

	return;
}

void genDelta(float* Delta)
{

	seedGen();
	
	for(int i=0;i<3;i++){
	Delta[i]=(float)randomNumberGen(10000);
	Delta[i]=Delta[i]/(10001.0f);}

	MPI_Bcast(Delta,3,MPI_FLOAT,0,MPI_COMM_WORLD);


	return;
}
