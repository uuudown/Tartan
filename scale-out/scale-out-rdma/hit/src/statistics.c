
#include "turH.h"


void calc_E( vectorField u, float2* t,float* E){


	calc_E_kernel(u,t);
		
	*E=sumElements2(t);



return;

}

void calc_D( vectorField u, float2* t,float* D){


	calc_D_kernel(u,t);

	*D=sumElements2(t);


	return;
}
