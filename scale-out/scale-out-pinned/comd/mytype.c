
#include "mytype.h"
#include "memUtils.h"
#include <stdlib.h> 
#include <string.h> 

void malloc_vec(vec_t *ptr, int nEntries)
{
        ptr->x = (real_t*) malloc(sizeof(real_t) * nEntries);
        ptr->y = (real_t*) malloc(sizeof(real_t) * nEntries);
        ptr->z = (real_t*) malloc(sizeof(real_t) * nEntries);
}

void free_vec(vec_t *ptr)
{
        comdFree(ptr->x);
        comdFree(ptr->y);
        comdFree(ptr->z);
}

void zeroVec(vec_t* a, int iOff)
{
   a->x[iOff] = 0.0;
   a->y[iOff] = 0.0;
   a->z[iOff] = 0.0;
}

void zeroVecAll(vec_t* a, int n)
{
   memset(a->x,  0, n*sizeof(real_t));
   memset(a->y,  0, n*sizeof(real_t));
   memset(a->z,  0, n*sizeof(real_t));
}

void copyVec(vec_t* a, int iOff, int jOff)
{
   a->x[jOff] = a->x[iOff];
   a->y[jOff] = a->y[iOff];
   a->z[jOff] = a->z[iOff];
}
