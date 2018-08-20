
#include "turH.h"


//RK2 CODE

static vectorField uw;
static vectorField r;


void RK2setup(void)
{
	

	size_t size=NXSIZE*NY*NZ*sizeof(float2);

	cudaCheck(cudaMalloc( (void**)&uw.x,size),"malloc_t1");
	cudaCheck(cudaMalloc( (void**)&uw.y,size),"malloc_t1");
	cudaCheck(cudaMalloc( (void**)&uw.z,size),"malloc_t1");
	
	set2zero(uw.x);
	set2zero(uw.y);
	set2zero(uw.z);

	cudaCheck(cudaMalloc( (void**)&r.x,size),"malloc_t1");
	cudaCheck(cudaMalloc( (void**)&r.y,size),"malloc_t1");
	cudaCheck(cudaMalloc( (void**)&r.z,size),"malloc_t1");

	set2zero(r.x);
	set2zero(r.y);
	set2zero(r.z);

	return;

}


static void collect_statistics(int step, float time, vectorField uw, vectorField u, case_config_t *config){

  float* E=(float*)malloc(sizeof(float));
  float* D=(float*)malloc(sizeof(float));
  
  calc_E(u,AUX,E);
  calc_D(u,AUX,D);
  
  float u_p=sqrt((2.0f/3.0f)*E[0]);	
  float omega_p=sqrt(REYNOLDS*D[0]);
  
  float lambda=sqrt(15.0f*u_p*u_p/(omega_p*omega_p));
  float eta=pow(REYNOLDS,-3.0f/4.0f)*pow(D[0],-1.0f/4.0f);
  
  float Rl=u_p*lambda*REYNOLDS;
  int kmax=sqrt(2.0f)/3.0f*N;	

	
  if(RANK == 0){
    FILE *statfilep = fopen(config->statfile,"a");
    printf("Appending file %s\n",config->statfile);
    fprintf(statfilep,"%e,%e,%e,%e,%e,%e,%e,%e,%e\n",
    	    time,E[0],D[0],u_p,omega_p,eta,lambda,Rl,eta*kmax);
    fclose(statfilep);
  }

  free(D);
  free(E);
}

static float calcDt(vectorField uw,vectorField u, case_config_t *config){	
	
	const float cfl = config->CFL;
	float dt=0.0f;

	float dtc=0.0f;	
	float dtf=0.0f;
	float dtv=0.0f;	
	
	float N3=(float)N*(float)N*(float)N;
	
	float* umax=(float*)malloc(3*sizeof(float));
	
	calcUmax(uw,umax,umax+1,umax+2);

	float c=(fabs(umax[0]/N3)+fabs(umax[1]/N3)+fabs(umax[2]/N3));
	
	dtc=cfl/((N/3)*c);	
	dtv=cfl*REYNOLDS/((N/3)*(N/3));
	/*
	if(RANK == 0){
	printf("\nVmax=(%f,%f,%f)\n",umax[0]/N3,umax[1]/N3,umax[2]/N3);
	}
	*/
	dt=fmin(dtc,dtv);
	//dt=fmin(dt,dtf);

	free(umax);

	return dt;

}


int RK2step(vectorField u,float* time, case_config_t *config)
{
	
	static float time_elapsed=0.0f;
	static int counter=0;	

	float pi=acos(-1.0f);
	float om=2.0f*pi/N;	
	
	float* Delta=(float*)malloc(3*sizeof(float));
	float* Delta_1=(float*)malloc(3*sizeof(float));
	float* Delta_2=(float*)malloc(3*sizeof(float));	

	int frec=2000;

	int kf=2;
		
	float dt=0.0f;
	float Cf;	

	//RK2 time steps	

	while(time_elapsed < *time){

	//Calc forcing	
	  if(config->forcing){
	    Cf=caclCf(u,AUX,kf,config);
	  }
	  else{
	    Cf = 0.0;
	  }   

	//Initial dealiasing

	dealias(u);	
	
	//Generate delta for dealiasing	
	
	genDelta(Delta);
	

	//printf("\n%f,%f,%f\n",Delta[0],Delta[1],Delta[2]);	
	
	for(int i=0;i<3;i++){
	Delta_1[i]=om*Delta[i];
	Delta_2[i]=om*(Delta[i]+0.5f);}
	
	//First substep

	copyVectorField(uw,u);	

	F(uw,r,Delta_1); 

	dt=calcDt(uw,u,config);	

	if( counter%config->stats_every == 0 ){
	  if (RANK == 0){ printf("Computing statistics.\n");}
	  collect_statistics(counter,time_elapsed,uw,u,config);
	}

	RK2_step_1(uw,u,r,REYNOLDS,dt,Cf,kf);

	//Second substep
	
	RK2_step_05(u,uw,REYNOLDS,dt,Cf,kf);	
	
	F(uw,r,Delta_2); 

	RK2_step_2(u,r,REYNOLDS,dt,Cf,kf);	 

	projectFourier(u);
	if(counter%1000){	
	imposeSymetry(u);}	

	counter++;
	time_elapsed+=dt;

	if(RANK==0){
	  printf("Timestep: %d, ",counter);
	  printf("Simulation time: %f, ",time_elapsed);
	  printf("Forcing coefficient: %f\n",Cf);
	}

	//End of step
	}
	
	*time=time_elapsed;

	free(Delta);
	free(Delta_1);
	free(Delta_2);

	if (RANK == 0){ printf("RK iterations finished.\n");}

	return counter;	
	
}

