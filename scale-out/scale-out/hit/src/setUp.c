#include "turH.h"

char host_name[MPI_MAX_PROCESSOR_NAME];
char mybus[16];
int pipe_xfer;
int min_kb_xfer;
int together;
int SMCOUNT;

MPI_Request *send_requests;
MPI_Request *recv_requests;
MPI_Status *send_status;
MPI_Status *recv_status;

static vectorField u;
static vectorField u_host;


int read_parallel_double(char *filename, double *x, int Nx, int Ny, int Nz,
			 int rank, int size);
int wrte_parallel_double(char *filename, double *x, int Nx, int Ny, int Nz,
			 int rank, int size);

#include <cstring>

int stringCmp( const void *a, const void *b)
{
     return strcmp((const char*)a,(const char*)b);
}


void setUp(void){

   //ASSIGN CUDA DEVICE
   int rank=0, local_rank=0, deviceCount;
   char (*host_names)[MPI_MAX_PROCESSOR_NAME];
   MPI_Comm nodeComm;
   int n, namelen, color, nprocs, local_procs;
   size_t bytes;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Get_processor_name(host_name,&namelen);
   bytes = nprocs * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
   host_names = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc(bytes);
   strcpy(host_names[rank], host_name);
   for (n=0; n<nprocs; n++)
   {
      MPI_Bcast(&(host_names[n]),MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n, MPI_COMM_WORLD);
   }
   qsort(host_names, nprocs,  sizeof(char[MPI_MAX_PROCESSOR_NAME]), stringCmp);
   color = 0;
   for (n=0; n<nprocs; n++)
   {
      if(n>0&&strcmp(host_names[n-1], host_names[n])) color++;
      if(strcmp(host_name, host_names[n]) == 0) break;
   }
   MPI_Comm_split(MPI_COMM_WORLD, color, 0, &nodeComm);
   MPI_Comm_rank(nodeComm, &local_rank);
   MPI_Comm_size(nodeComm, &local_procs);
   free(host_names);
   CHECK_CUDART( cudaGetDeviceCount(&deviceCount) );
   CHECK_CUDART( cudaSetDevice(local_rank%deviceCount) );
   int dev;
   struct cudaDeviceProp deviceProp;
   CHECK_CUDART( cudaGetDevice(&dev) );
   CHECK_CUDART( cudaGetDeviceProperties(&deviceProp, dev) );
   sprintf(&mybus[0], "0000:%02x:%02x.0", deviceProp.pciBusID, deviceProp.pciDeviceID);
   printf("rank: %d host: %s device: %s IGLOBAL: %d\n",RANK,host_name,mybus,IGLOBAL); 
   fflush(stdout);
   SMCOUNT = deviceProp.multiProcessorCount;
   MPI_Barrier(MPI_COMM_WORLD);
   if(rank==0){
     printf("\n# Running on '%s'\n", deviceProp.name);
     printf("# SMs = %d\n", deviceProp.multiProcessorCount);
     printf("# clock = %d\n", deviceProp.clockRate);
   }
   pipe_xfer = 1;
   const char *name0= "PIPE_XFER";
   char *value0;
   value0 = getenv(name0);
   if (value0 != NULL ){
       pipe_xfer = atoi(value0);
       if(rank==0) printf ("PIPE_XFER env var: %d \n",pipe_xfer);
   }
   min_kb_xfer = 512;
   const char *name1= "MIN_KB_XFER";
   char *value1;
   value1 = getenv(name1);
   if (value1 != NULL ){
       min_kb_xfer = atoi(value1);
       if(rank==0) printf ("MIN_KB_XFER env var: %d \n",min_kb_xfer);
   }
   together = 1;
   const char *name2= "TOGETHER";
   char *value2;
   value2 = getenv(name2);
   if (value2 != NULL ){
       together = atoi(value2);
       if(rank==0) printf ("TOGETHER env var: %d \n",together);
   }

  send_requests = (MPI_Request*)malloc(SIZE*sizeof(MPI_Request));
  recv_requests = (MPI_Request*)malloc(SIZE*sizeof(MPI_Request));
  send_status = (MPI_Status*)malloc(SIZE*sizeof(MPI_Status));  
  recv_status = (MPI_Status*)malloc(SIZE*sizeof(MPI_Status));

   //test cuda error check
   //CHECK_CUDART( cudaMemcpy( NULL, NULL, 1, cudaMemcpyDeviceToHost) );	

	/*
	if(SIZE%Ndevices!=0){
	printf("Error_ndevices");
	exit(1);
	}

	int A=SIZE/Ndevices;
	*/	

/*
 if(RANK==0){
        int cur_rank;
        for(cur_rank=0; cur_rank<SIZE; cur_rank++){
   printf("\n");
       //int together=1;
       //while( (NZ*(NX/SIZE)*(NY/SIZE)*sizeof(float2)*together) < min_kb_xfer*1024 ) together *= 2;
  printf("together = %d \n",together);
        int ii,ii_chunk;
        for(ii_chunk=0; ii_chunk<(SIZE+together-1)/together; ii_chunk++){
          if(ii_chunk==0){
	      for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              int dest=ii^cur_rank;
              printf("iter %d: rank %d --> %d  \n",ii,cur_rank,dest);
            }
          }else{
            int min_dest=999999;
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              int dest=ii^cur_rank;
              if(dest<min_dest) min_dest=dest;
            }
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              int dest=ii^cur_rank;
              printf("iter %d: rank %d --> %d min_dest: %d \n",ii,cur_rank,dest,min_dest);
            }
          }
        }
        }

}
*/
	//Setups
	fftSetup();
	setFftAsync();
	
	//RK2setup();
	RK3setup();
	
	size_t size=NXSIZE*NY*NZ*sizeof(float2);
	cudaCheck(cudaMalloc( (void**)&AUX,size),"malloc_t1");		

	setTransposeCudaMpi();	

	return;

}

void starSimulation(void){

        config_t config;
	config_setting_t *read;
	config_setting_t *write;
	const char *str;
	
	// Read configuration file
	config_init(&config);
  
	if (! config_read_file(&config, "run.conf")){
	  fprintf(stderr, "%s:%d - %s\n", config_error_file(&config),
		  config_error_line(&config), config_error_text(&config));
	  config_destroy(&config);
	  return;
	}
	case_config_t case_config = {
	  (float) config_setting_get_float(config_lookup(&config,"application.CFL")),
	  (float) config_setting_get_float(config_lookup(&config,"application.time")),
	  (float) config_setting_get_float(config_lookup(&config,"application.RES")),
	  (int) config_setting_get_bool(config_lookup(&config,"application.forcing")),
	  (int) config_setting_get_bool(config_lookup(&config,"application.tauS")),
	  (int) config_setting_get_int(config_lookup(&config,"application.stats_every")),
	  (char *) config_setting_get_string(config_lookup(&config,"application.read.U")),
	  (char *) config_setting_get_string(config_lookup(&config,"application.read.V")),
	  (char *) config_setting_get_string(config_lookup(&config,"application.read.W")),
	  (char *) config_setting_get_string(config_lookup(&config,"application.statfile")),
	  (char *) config_setting_get_string(config_lookup(&config,"application.path")),
	  (char *) config_setting_get_string(config_lookup(&config,"application.write.U")),
	  (char *) config_setting_get_string(config_lookup(&config,"application.write.V")),
	  (char *) config_setting_get_string(config_lookup(&config,"application.write.W")),
	};

	//Size 
		
	size_t size=NXSIZE*NY*NZ*sizeof(float2);


	// Memory Alloc	

	u_host.x=(float2*)malloc(size);
	u_host.y=(float2*)malloc(size);
	u_host.z=(float2*)malloc(size);


	// Allocate memory in device and host 	

	cudaCheck(cudaMalloc( (void**)&u.x,size),"malloc_t1");
	cudaCheck(cudaMalloc( (void**)&u.y,size),"malloc_t1");
	cudaCheck(cudaMalloc( (void**)&u.z,size),"malloc_t1");

	//MPI COPY to nodes
        if (strcmp(case_config.readU, "-") == 0){
            // If the file name is -, then a dummy field is created
            if (RANK == 0){ printf("Creating dummy file.\n");}
            mpiCheck(create_parallel_float((float*)u_host.x,NX,NY,2*NZ,RANK,SIZE),"read");
            mpiCheck(create_parallel_float((float*)u_host.y,NX,NY,2*NZ,RANK,SIZE),"read");
            mpiCheck(create_parallel_float((float*)u_host.z,NX,NY,2*NZ,RANK,SIZE),"read");
          }
        else{
          if (RANK == 0){
            printf("Reading start files \n");
            printf("%s \n", case_config.readU);
          }
          mpiCheck(read_parallel_float(case_config.readU,(float*)u_host.x,NX,NY,2*NZ,RANK,SIZE),"read");
          mpiCheck(read_parallel_float(case_config.readV,(float*)u_host.y,NX,NY,2*NZ,RANK,SIZE),"read");
          mpiCheck(read_parallel_float(case_config.readW,(float*)u_host.z,NX,NY,2*NZ,RANK,SIZE),"read");
        }


	//COPY to GPUs

	cudaCheck(cudaMemcpy(u.x,u_host.x, size, cudaMemcpyHostToDevice),"MemInfo1_A");
	cudaCheck(cudaMemcpy(u.y,u_host.y, size, cudaMemcpyHostToDevice),"MemInfo1_A");
	cudaCheck(cudaMemcpy(u.z,u_host.z, size, cudaMemcpyHostToDevice),"MemInfo1_A");

	//U sep up

	dealias(u);
	projectFourier(u);

	//RK integration
	

	float time = (float) config_setting_get_float(config_lookup(&config,"application.time"));

	int counter=0;
	
	counter=RK3step(u,&time,&case_config);

	int mpierr = MPI_Barrier(MPI_COMM_WORLD);

	if (RANK == 0){ printf("RK iterations finished.\n");}

	//COPY to CPU

	cudaCheck(cudaMemcpy(u_host.x,u.x,size,cudaMemcpyDeviceToHost),"MemInfo1_B");
	cudaCheck(cudaMemcpy(u_host.y,u.y,size,cudaMemcpyDeviceToHost),"MemInfo1_B");
	cudaCheck(cudaMemcpy(u_host.z,u.z,size,cudaMemcpyDeviceToHost),"MemInfo1_B");
		

	//MPI COPY to nodes
        if (strcmp(case_config.writeU, "-") == 0){
           if (RANK == 0){ printf("Skipping output .\n");}
         }
        else{
            if (RANK == 0){ printf("Writing output.\n");}
        mpiCheck(wrte_parallel_float(case_config.writeU,(float*)u_host.x,NX,NY,2*NZ,RANK,SIZE),"write");
        mpiCheck(wrte_parallel_float(case_config.writeV,(float*)u_host.y,NX,NY,2*NZ,RANK,SIZE),"write");
        mpiCheck(wrte_parallel_float(case_config.writeW,(float*)u_host.z,NX,NY,2*NZ,RANK,SIZE),"write");
        }

	if (RANK == 0){ printf("Nothing important left to do.\n");}

	config_destroy(&config);

return;

}


