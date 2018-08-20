#include "turH.h"

static void calc_uuS(vectorField u,vectorField A,vectorField B,float2* OUT,float alpha,int d){

	size_t size=NXSIZE*NY*NZ*sizeof(float2);

	//Filter u1,u2,u3

	cudaCheck(cudaMemcpy(A.x,u.x, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(A.y,u.y, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(A.z,u.z, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	
	//High pass filter
	gaussFilter_High(A,alpha);

	fftBackward(A.x);
	fftBackward(A.y);
	fftBackward(A.z);

	calcUU(A,d);	
	
	//Calc Sij

	cudaCheck(cudaMemcpy(B.x,u.x, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(B.y,u.y, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(B.z,u.z, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	
	//Low pass filter
	gaussFilter(B,alpha);

	calcS(B,d);
	
	fftBackward(B.x);
	fftBackward(B.y);
	fftBackward(B.z);
	
	//Calc Sij
	calc_tauS_cuda(OUT,A,B,0);

	return;
}

static void calc_udTau(vectorField u,vectorField A,vectorField B,float2* OUT,float alpha,int d){

	size_t size=NXSIZE*NY*NZ*sizeof(float2);	

	//Calc G(u)G(u)

	cudaCheck(cudaMemcpy(A.x,u.x, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(A.y,u.y, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(A.z,u.z, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	
	
	gaussFilter(A,alpha);

	fftBackward(A.x);
	fftBackward(A.y);
	fftBackward(A.z);

	calcUU(A,d);	
	
	//Calc G(uu)

	cudaCheck(cudaMemcpy(B.x,u.x, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(B.y,u.y, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(B.z,u.z, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	

	fftBackward(B.x);
	fftBackward(B.y);
	fftBackward(B.z);

	calcUU(B,d);

	fftForward(B.x);
	fftForward(B.y);
	fftForward(B.z);
	
	gaussFilter(B,alpha);

	normalize(B);

	fftBackward(B.x);
	fftBackward(B.y);
	fftBackward(B.z);

	//Guardado en A: G(uu)-G(u)G(u)

	calcL(A,B);

	fftForward(A.x);
	fftForward(A.y);
	fftForward(A.z);

	calc_dTau(A,d);
	
	fftBackward(A.x);
	fftBackward(A.y);
	fftBackward(A.z);	
	
	cudaCheck(cudaMemcpy(B.x,u.x, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(B.y,u.y, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(B.z,u.z, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	
	
	gaussFilter_High(B,alpha);

	normalize(B);

	fftBackward(B.x);
	fftBackward(B.y);
	fftBackward(B.z);

	//PROBLEMA //CHECK !!!!!

	calc_tauS_cuda(OUT,A,B,d);
	
	return;
}

static void calc_tauSii(vectorField u,vectorField A,vectorField B,float2* OUT,float alpha,int d){
	
	size_t size=NXSIZE*NY*NZ*sizeof(float2);

	//Calc G(u)G(u)

	cudaCheck(cudaMemcpy(A.x,u.x, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(A.y,u.y, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(A.z,u.z, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	
	
	
	gaussFilter(A,alpha);


	calcUU(A,d);	
		
	//Calc G(uu)

	cudaCheck(cudaMemcpy(B.x,u.x, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(B.y,u.y, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(B.z,u.z, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	

	fftBackward(B.x);
	fftBackward(B.y);
	fftBackward(B.z);

	calcUU(B,d);

	fftForward(B.x);
	fftForward(B.y);
	fftForward(B.z);
	
	gaussFilter(B,alpha);
	normalize(B);

	fftBackward(B.x);
	fftBackward(B.y);
	fftBackward(B.z);
	

	//Guardado en A: G(uu)-G(u)G(u)

	calcL(A,B);

	//Calc Sij
	

	cudaCheck(cudaMemcpy(B.x,u.x, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(B.y,u.y, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(B.z,u.z, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	
	
	gaussFilter(B,alpha);

	calcS(B,d);
		
	fftBackward(B.x);
	fftBackward(B.y);
	fftBackward(B.z);
	
	//Calc Sij

	//CHECK	

	calc_tauS_cuda(OUT,A,B,0);


}


// Using 13 float2 buffers + the buffer for velocity; 

float calc_T(vectorField u,vectorField A,vectorField B,float2* aux,float alpha)
{

	float uuS_1;
	float uuS_2;

	calc_uuS(u,A,B,aux,alpha,0);
		
	uuS_1=sumElements(aux);

	calc_uuS(u,A,B,aux,alpha,1);
		
	uuS_2=sumElements(aux);
	
	float uuS=uuS_1+2.0f*uuS_2;
	
	//Calc udtau

	float udTau_1;
	float udTau_2;
	float udTau_3;

	calc_udTau(u,A,B,aux,alpha,0);	

	udTau_1=sumElements(aux);

	calc_udTau(u,A,B,aux,alpha,1);

	udTau_2=sumElements(aux);

	calc_udTau(u,A,B,aux,alpha,2);
	
	udTau_3=sumElements(aux);

	float udTau=udTau_1+udTau_2+udTau_3;	

	float T=(udTau-uuS)/((float)N*N*N);

	if(RANK==0){
	printf("\nudTau=(%e,%e,%e)",udTau_1,udTau_2,udTau_3);
	printf("\nudTau=(%e,%e)",uuS_1,uuS_2);
	}
	return T;		
	
}

float calc_tauS(vectorField u,vectorField A,vectorField B,float2* aux,float alpha)
{

	float tauS_1;
	float tauS_2;

	calc_tauSii(u,A,B,aux,alpha,0);
		
	tauS_1=sumElements(aux);

	calc_tauSii(u,A,B,aux,alpha,1);
		
	tauS_2=sumElements(aux);
		
	float tauS=(tauS_1+2.0f*tauS_2)/((float)N*N*N);
	
	return tauS;		
	
}

