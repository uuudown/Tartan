#include "turH.h"

extern float2* aux_dev[6];
extern float* umax;
extern float* umax_d;
//extern cudaStream_t d2h_stream;
//extern cudaEvent_t events[1000];

void copyVectorField(vectorField u1,vectorField u2){

	// Copy to u
	
	size_t size=NXSIZE*NY*NZ*sizeof(float2);	
	
	cudaCheck(cudaMemcpy(u1.x,u2.x, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(u1.y,u2.y, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	cudaCheck(cudaMemcpy(u1.z,u2.z, size, cudaMemcpyDeviceToDevice),"MemInfo1");
	
	return;

}

float Fdt( vectorField u, vectorField r,float* Delta,float Cf)
{
        const float cfl=0.5;
        float dt=0.0f;

        float dtc=0.0f;
        float dtf=0.0f;
        float dtv=0.0f;

        float N3= (float) N*(float)N*(float)N;

START_RANGE("calc_u_w",1)
        calc_U_W(u,r);
END_RANGE
START_RANGE("shift",2)
        shift(u,Delta);
END_RANGE
START_RANGE("dealias",3)
        dealias(u);
END_RANGE
START_RANGE("FFT",4)
        fftBack1T_A(u.x,0);
        fftBack1T_A(u.y,1);

        shift(r,Delta);
        dealias(r);

        fftBack1T_A(r.x,3);
        fftBack1T_A(r.y,4);

        fftBack1T_A(u.z,2);
        fftBack1T_A(r.z,5);


        fftBack1T_B(u.x,0);
        fftBack1T_B(u.y,1);
        fftBack1T_B(r.x,3);
        fftBack1T_B(r.y,4);

        calc_conv_rotor_3(r,u);
        fftForw1T_A(r.z,4);

        fftBack1T_B(u.z,2);
///*
        CHECK_CUDART( cudaMemsetAsync(umax_d, 0, 3*sizeof(float), compute_stream) );
        calc_Umax2(u, umax_d);
        CHECK_CUDART( cudaEventRecord(events[0],compute_stream) );
        CHECK_CUDART( cudaStreamWaitEvent(d2h_stream,events[0],0) );
        CHECK_CUDART( cudaMemcpyAsync(umax,umax_d,3*sizeof(float),cudaMemcpyDeviceToHost, d2h_stream) );
        CHECK_CUDART( cudaEventRecord(events[999],d2h_stream) );
//cudaDeviceSynchronize();
//*/
        fftBack1T_B(aux_dev[0],5);

        calc_conv_rotor_12(r,u,aux_dev[0]);

        fftForw1T_A(r.x,0);
        fftForw1T_A(r.y,1);

        fftForw1T_B(r.z,4);

        fftForw1T_B(r.x,0);
        fftForw1T_B(r.y,1);
END_RANGE
START_RANGE("dealias",3)
        dealias(r);
END_RANGE
START_RANGE("SHIFT_delta",6)
        Delta[0]=-Delta[0];
        Delta[1]=-Delta[1];
        Delta[2]=-Delta[2];
END_RANGE
START_RANGE("SHIFT_func",1)
        shift(r,Delta);
END_RANGE
//START_RANGE("UMAX",5)
//        CHECK_CUDART(cudaMemcpy(umax_d,aux_dev[0],3*sizeof(float),cudaMemcpyDeviceToDevice) );
//        calc_Umax2(u, umax_d);
//END_RANGE
//        CHECK_CUDART( cudaMemcpy(umax,umax_d,3*sizeof(float),cudaMemcpyDeviceToHost) );
START_RANGE("REDUCE",4)
        //MPI REDUCE
        CHECK_CUDART( cudaEventSynchronize(events[999]) );
//cudaDeviceSynchronize();
        reduceMAX(&umax[0],&umax[1],&umax[2]);
END_RANGE
        float c=(fabs(umax[0]/N3)+fabs(umax[1]/N3)+fabs(umax[2]/N3));

        dtc=cfl/((N/3.0f)*c);
        dtv=cfl*REYNOLDS/((N/3.0f)*(N/3.0f));
        dtf=cfl/Cf;

        if(RANK == 0 /*&& counter%10==0*/){
        printf("\nVmax=( %3.8f, %3.8f, %3.8f )\n",umax[0]/N3,umax[1]/N3,umax[2]/N3);
        }

        dt=fmin(dtc,dtv);
        dt=fmin(dt,dtf);
        return dt;        
}


void F( vectorField u, vectorField r,float* Delta)
{
	
	// Calculate vorticity w and move u to u_w workarray	
	// s -> u
	// r -> w
START_RANGE("calc_u_w",1)
	calc_U_W(u,r);
END_RANGE
	// Phase shift U and W
START_RANGE("shift",2)
//	shift(r,Delta);
	shift(u,Delta);		
END_RANGE
	
	// Fill with zeros	

START_RANGE("dealias",3)
        dealias(u);
//        dealias(r);
END_RANGE
	
	// Transform u to real space
/*
	fftBackward(u.x);
	fftBackward(u.y);
	fftBackward(u.z);
*/
#if 0
START_RANGE("FFTbt",4)
	fftBack1T(u.x);
END_RANGE
START_RANGE("FFTbt",4)
	fftBack1T(u.y);
END_RANGE
START_RANGE("FFTbt",4)
	fftBack1T(u.z);
END_RANGE
#endif
#if 1
START_RANGE("FFT",4)
        fftBack1T_A(u.x,0);
        fftBack1T_A(u.y,1);

        shift(r,Delta);
        dealias(r);

        fftBack1T_A(r.x,3);
        fftBack1T_A(r.y,4);

        fftBack1T_A(u.z,2);
        fftBack1T_A(r.z,5);


        fftBack1T_B(u.x,0);
        fftBack1T_B(u.y,1);
        fftBack1T_B(r.x,3);
        fftBack1T_B(r.y,4);

        calc_conv_rotor_3(r,u);
        fftForw1T_A(r.z,4);

//        fftBack1T_A(u.z,2);
//        fftBack1T_A(r.z,5);

//        fftBack1T_B(u.z,2);
//        fftBack1T_B(r.z,5);

        fftBack1T_B(u.z,2);
        fftBack1T_B(aux_dev[0],5);

        calc_conv_rotor_12(r,u,aux_dev[0]);


//        fftForw1T_B(r.z,4);


//END_RANGE_ASYNC

//START_RANGE("conv_rotor_123",2)
	//calc_conv_rotor(r,u);
	//calc_conv_rotor_3(r,u,aux_dev[0]);
	//calc_conv_rotor_12(r,u,aux_dev[0]);
//END_RANGE
#endif
#if 0
        fftBack1T_A(u.x,0);
        fftBack1T_B(u.x,0);
cudaDeviceSynchronize();
        fftBack1T_A(u.y,1);
        fftBack1T_B(u.y,1);
cudaDeviceSynchronize();
        fftBack1T_A(u.z,2);
        fftBack1T_B(u.z,2);
cudaDeviceSynchronize();
#endif
	// Fill with zeros
//START_RANGE_ASYNC("dealias",3)	
//	dealias(r);	
//END_RANGE_ASYNC	
	// Transform w to real space

/*	
	fftBackward(r.x);
	fftBackward(r.y);	
	fftBackward(r.z);
*/
#if 0
START_RANGE("FFTbt",4)
	fftBack1T(r.x);
END_RANGE
START_RANGE("FFTbt",4)
	fftBack1T(r.y);
END_RANGE
START_RANGE("FFTbt",4)
	fftBack1T(r.z);	
END_RANGE	
#endif
#if 1
/*START_RANGE("FFT_BK",4)
        fftBack1T_A(r.x,0);
        fftBack1T_A(r.y,1);
        fftBack1T_A(r.z,2);
        fftBack1T_B(r.x,0);
        fftBack1T_B(r.y,1);
        fftBack1T_B(r.z,2);
END_RANGE*/
#endif
#if 0
        fftBack1T_A(r.x,0);
        fftBack1T_B(r.x,0);
cudaDeviceSynchronize();
        fftBack1T_A(r.y,1);
        fftBack1T_B(r.y,1);
cudaDeviceSynchronize();
        fftBack1T_A(r.z,2);
        fftBack1T_B(r.z,2);
cudaDeviceSynchronize();

#endif
	// Calculate the convolution rotor of u and w

//START_RANGE("conv_rotor",2)
//	calc_conv_rotor(r,u);
//END_RANGE
	// Transform rotor to fourier space

/*	
	fftForward(r.x);
	fftForward(r.y);
	fftForward(r.z);
*/
#if 0
START_RANGE("FFTft",2)
	fftForw1T(r.x);
END_RANGE
START_RANGE("FFTft",2)
	fftForw1T(r.y);
END_RANGE
START_RANGE("FFTft",2)
	fftForw1T(r.z);	
END_RANGE
#endif
#if 1
//START_RANGE("FFT_FW",5)
        fftForw1T_A(r.x,0);
        fftForw1T_A(r.y,1);
        //fftForw1T_A(r.z,2);
        
        fftForw1T_B(r.z,4);

        fftForw1T_B(r.x,0);
        fftForw1T_B(r.y,1);
//        fftForw1T_B(r.z,2);
END_RANGE
#endif
#if 0
        fftForw1T_A(r.x,0);
        fftForw1T_B(r.x,0);
cudaDeviceSynchronize();
        fftForw1T_A(r.y,1);
        fftForw1T_B(r.y,1);
cudaDeviceSynchronize();
        fftForw1T_A(r.z,2);
        fftForw1T_B(r.z,2);
cudaDeviceSynchronize();
#endif
	// Dealiase high frecuencies	
START_RANGE("dealias",3)
	dealias(r);
END_RANGE
START_RANGE("SHIFT_delta",6)
	//shift back
	Delta[0]=-Delta[0];
	Delta[1]=-Delta[1];
	Delta[2]=-Delta[2];
END_RANGE
START_RANGE("SHIFT_func",1)	
	shift(r,Delta);	
END_RANGE
}

