#include <stdio.h>
#include <cuda.h>
#include "mpi.h"

//------------------------------------------------------------------------------------------------------------------------------------------

#define BLOCKSIZE 16

//--------------------------------------------------------------------------------------------------------------------------------------------

int IntializingMatrixVectors(float **, float **, float **, int , int , int , int);
int CheckDevice(int );

//--------------------------------------------------------------------------------------------------------------------------------------------

//Pragma routine to report the detail of cuda error

#define CUDA_SAFE_CALL(call)                                                         \
    do{                                                                      \
        cudaError_t err = call;                                             \
        if(err != cudaSuccess)                                              \
        {                                                                   \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                    __FILE__, __LINE__, cudaGetErrorString( err) );             \
            exit(1);                                                    \
        }                                                                   \
    } while (0)                                                           \

//----------------------------------------------------------------------------------------------------------------------------------------


//Kernel that performs Matrix Vector Multiplication
__global__ void MatrixVectorMultiplication(float *Matrix,float *Vector,float *Solution, int RowsNo, int ColsNo, int RowsNo2, int ColsNo2, int VectorLength, int ScatterSize, int ThreadDim, int MyRank, int NumberofProcessors)
{  	
    int tidx = threadIdx.x;


    int count,ThreadColumnIndex,pass = 0 ;
    float TempResult = 0.0f;

    for (int i = 0; i < RowsNo/NumberofProcessors; i++) {
        for (tidx = 0; tidx < ColsNo2; tidx++) {
            float sum = 0.0;
            for (int k = 0; k < RowsNo2; k++)
                sum = sum + Matrix[i * ColsNo + k] * Vector[k * ColsNo2 + tidx];

            Solution[i * ColsNo2 + tidx] = sum;

        }

    }

    __syncthreads();
}//End of Matrix Vector Multiplication Device Function
//---------------------------------------------------------------------------------------------------------------------------------------



int main(int argc, char **argv)
{
    int MyRank, NumberOfProcessors;
    int Root = 0, Index, Status = 1;
    float *MatrixA, *VectorB, *ResultVector, *MatrixB, *ResultMatrix;
    float *MyMatrixA, *MyResultMatrix;
    float *DeviceMyMatrixA, *DeviceMyResultVector, *DeviceVectorB, *DeviceMatrixB, *CPUResultVector;
    int RowsNo, ColsNo, RowsNo2, ColsNo2, VectorSize, ScatterSize, IndexCol, IndexValue, DeviceStatus;
    int matrixbsize, pinned;
    int print =0;
    int verify =0;


    //MPI Intialization
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
    MPI_Comm_size(MPI_COMM_WORLD, &NumberOfProcessors);



    //Checking if valid number of arguements have been passed
    if(argc < 6)
    {
        if(MyRank == Root)
            printf("Usage:< mpirun >< -n >< Number of processors >< ./Program Name >< Number of Rows of Matri x>< Number of Columns of Matrix >< Rows of Matrix 2 > <Coloumns of Matrix 1>  <1 for pinned memory, 2 for unpinnned>  <-v if verification is required>  <-p if print is required>\n");
        MPI_Finalize();
        exit(-1);
    }
    if ((argc >= 7 && strcmp(argv[6],"-v") == 0))  {
        verify=1;}

    if ((argc ==8 && strcmp(argv[7],"-p") == 0) || (argc == 7 && strcmp(argv[6],"-p")==0))  {
        print=1;}






    //Assigning values to RowsNo, ColsNo, VectorSize from the arguements passed
    RowsNo = atoi( argv[1] );
    ColsNo = atoi( argv[2] );
    RowsNo2= atoi( argv[3] );
    ColsNo2= atoi( argv[4] );
    pinned = atoi( argv[5]);




    matrixbsize=RowsNo2*ColsNo2;
    if (MyRank==0)
        printf("Resultant Matrix Number of Elements is %d\n\n\n",matrixbsize);


    int elements;

    //Checking if columns is equal to vector size
    if( ColsNo != RowsNo2)
    {
        if(MyRank == Root)
            printf("Entered wrong input, Number of columns of matrix should be equal to number of rows \n");
        MPI_Finalize();
        exit(-1);
    }

    if(RowsNo < NumberOfProcessors)
    {
        if(MyRank == Root)
            printf("Given number of Rows of the matrix should be more than number of processors \n");
        MPI_Finalize();
        exit(-1);
    }

    //Checking if Matrix can be distributed evenly to all the nodes
    if(RowsNo % NumberOfProcessors != 0)
    {
        if(MyRank == Root)
            printf("The Rows of the matrix can not be distributed evenly among processors \n");
        MPI_Finalize();
        exit(-1);
    }

    //Root node intializes the Matrix, Vector and Result Vector
    if(MyRank == Root)
        Status = IntializingMatrixVectors(&MatrixA, &MatrixB, &ResultVector, RowsNo, ColsNo, RowsNo2, ColsNo2);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&Status, 1, MPI_INT, Root, MPI_COMM_WORLD);


    cudaSetDevice(MyRank);

    CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceMatrixB, matrixbsize*sizeof(float) ) );

    if (MyRank == Root) //root copy to device
    {
        cudaMemcpy( (void *)DeviceMatrixB, (void *)MatrixB,  matrixbsize*sizeof(float), cudaMemcpyHostToDevice );
    }

    //Broad casting the Vector to all the nodes from root node
    MPI_Bcast(DeviceMatrixB, matrixbsize, MPI_FLOAT, Root, MPI_COMM_WORLD);

    //Calculating the Scatter size of the Matrix
    ScatterSize = RowsNo / NumberOfProcessors;
    elements = (RowsNo*ColsNo2)/NumberOfProcessors;

    CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceMyMatrixA, ScatterSize * ColsNo * sizeof(float) ) );
    CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceMyResultVector, elements * sizeof(float) ) );

    //Distributing the Matrix among to all the nodes
    MPI_Scatter(MatrixA, ScatterSize * ColsNo, MPI_FLOAT, DeviceMyMatrixA, ScatterSize * ColsNo, MPI_FLOAT, Root, MPI_COMM_WORLD);


    DeviceStatus = CheckDevice(MyRank);


    //Calling the kernel which performs Matrix Vector Product
    MatrixVectorMultiplication<<<1, 256>>>(DeviceMyMatrixA, DeviceMatrixB, DeviceMyResultVector, RowsNo, ColsNo, RowsNo2, ColsNo2, ColsNo, ScatterSize, BLOCKSIZE, MyRank, NumberOfProcessors);	


    MPI_Barrier(MPI_COMM_WORLD);

    //Root processor gathering from all nodes to get the final result vector
    MPI_Gather(DeviceMyResultVector,elements, MPI_FLOAT, ResultVector, elements, MPI_FLOAT, Root, MPI_COMM_WORLD);


    //To verify:

    //Compute on CPU


    if (MyRank==0 && verify==1){
        printf("\n\n\nVerification Done\n\n");
        CPUResultVector = (float *)malloc(RowsNo * ColsNo2 * sizeof(float));
        for (int i = 0; i < RowsNo; i++) {
            for (int j = 0; j < ColsNo2; j++) {
                float sum = 0.0;
                for (int k = 0; k < RowsNo2; k++)
                    sum = sum + MatrixA[i * ColsNo + k] * MatrixB[k * ColsNo2 + j];

                CPUResultVector[i * ColsNo2 + j] = sum;

            }
        }

        for(Index = 0; Index < ColsNo2 * RowsNo; Index++)

        {
            int a = ResultVector[Index];
            int b = CPUResultVector[Index];
            if (a!=b)
            {printf("Error in computation and values are %f and %f",ResultVector[Index], CPUResultVector[Index] );}
        }

    }







    //Root processor printing the resultant vector if print specified
    if(MyRank == Root && print==1)
    {
        printf("The resultant vector with size %d  is \n",RowsNo*ColsNo2);
        for(Index = 0; Index < ColsNo2 * RowsNo; Index++)
            printf(" %f \n", ResultVector[Index]);

        //freeing the Vectors allocated by the root node
        free(MatrixA);
        free(ResultVector);
    }





    if (MyRank==0)
        printf("\n\n Computation Done .....\n Exiting \n\n");


    //Freeing the host memory	
    //free(MyMatrixA);
    //free(MatrixB);

    /*//Freeing the device memory
      CUDA_SAFE_CALL( cudaFree( DeviceMyMatrixA ) );
      CUDA_SAFE_CALL( cudaFree( DeviceMatrixB ) );
      CUDA_SAFE_CALL( cudaFree( DeviceMyResultVector ) );*/


    MPI_Finalize();



    return(0);
}//End of Main function
//---------------------------------------------------------------------------------------------------------------------------------------

int IntializingMatrixVectors(float **MatrixA, float **MatrixB, float **ResultVector, int RowsNo, int ColsNo, int RowsNo2, int ColsNo2)
{
    float *TempMatrixA, *TempVectorB, *TempResultVector, *TempMatrixB;
    int Status, Index;

    //Allocating memory on the host
    TempMatrixA = (float *)malloc(RowsNo * ColsNo * sizeof(float));
    if(TempMatrixA == NULL)
        Status = 0;
    TempMatrixB = (float *)malloc(RowsNo2 * ColsNo2 * sizeof(float));
    if(TempMatrixB == NULL)
        Status = 0;
    TempResultVector = (float *)malloc(RowsNo * ColsNo2 * sizeof(float));
    if(TempResultVector == NULL)
        Status = 0;

    //Intializing the Matrix and the Vectors

    int a=10;
    for(Index = 0; Index < RowsNo*ColsNo; Index++)
    {

        TempMatrixA[Index] = (float)rand()/(float)(RAND_MAX/a);

    }
    printf("Matrix A initialized");		
    printf("\n\n\n\n\n");

    for(Index = 0; Index < RowsNo2 * ColsNo2; Index++)
    {		TempMatrixB[Index] = (float)rand()/(float)(RAND_MAX/a);

    }
    printf("Matrix B initilized");
    printf("\n\n\n\n\n\n\n");

    for(Index = 0; Index < ColsNo2 * RowsNo; Index++)
    {
        TempResultVector[Index] = 0.0f;
        //	printf("%f\t", TempResultVector[Index]);
    }


    *MatrixA = TempMatrixA;
    *MatrixB = TempMatrixB;
    *ResultVector = TempResultVector;

    return(Status);
}//End of the function
//-------------------------------------------------------------------------------------------------------------------------------------


int CheckDevice(int MyRank)
{
    int DeviceCount, Device;
    struct cudaDeviceProp Properties;

    cudaGetDeviceCount(&DeviceCount);
    if(DeviceCount >= 1)
    {
        cudaGetDevice(&Device);
        cudaGetDeviceProperties(&Properties, Device);
        printf("Processor with  rank %d has the Device by name %s and computation is done on this device \n",MyRank, Properties.name);
    }

    return(DeviceCount);

}//End of function
//--------------------------------------------------------------------------------------------------------------





