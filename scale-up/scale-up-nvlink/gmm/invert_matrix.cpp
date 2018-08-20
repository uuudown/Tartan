#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "invert_matrix.h"
#include "gaussian.h"


static float double_abs(float x);

static int 
ludcmp(float *a,int n,int *indx,float *d);

static void 
lubksb(float *a,int n,int *indx,float *b);

/*
 * Inverts a square matrix (stored as a 1D float array)
 * 
 * actualsize - the dimension of the matrix
 *
 * written by Mike Dinolfo 12/98
 * version 1.0
 */
void invert_cpu(float* data, int actualsize, float* log_determinant)  {
    int maxsize = actualsize;
    int n = actualsize;
    *log_determinant = 0.0;

    /*DEBUG("\n\nR matrix before inversion:\n");
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            DEBUG("%.4f ",data[i*n+j]);
        }
        DEBUG("\n");
    }*/
    
    if (actualsize == 1) { // special case, dimensionality == 1
        *log_determinant = logf(data[0]);
        data[0] = 1.0 / data[0];
    } else if(actualsize >= 2) { // dimensionality >= 2
        for (int i=1; i < actualsize; i++) data[i] /= data[0]; // normalize row 0
        for (int i=1; i < actualsize; i++)  { 
            for (int j=i; j < actualsize; j++)  { // do a column of L
                float sum = 0.0;
                for (int k = 0; k < i; k++)  
                    sum += data[j*maxsize+k] * data[k*maxsize+i];
                data[j*maxsize+i] -= sum;
            }
            if (i == actualsize-1) continue;
            for (int j=i+1; j < actualsize; j++)  {  // do a row of U
                float sum = 0.0;
                for (int k = 0; k < i; k++)
                    sum += data[i*maxsize+k]*data[k*maxsize+j];
                data[i*maxsize+j] = 
                    (data[i*maxsize+j]-sum) / data[i*maxsize+i];
            }
        }

        for(int i=0; i<actualsize; i++) {
            *log_determinant += log10(fabs(data[i*n+i]));
            //printf("log_determinant: %e\n",*log_determinant); 
        }
        //printf("\n\n");
        for ( int i = 0; i < actualsize; i++ )  // invert L
            for ( int j = i; j < actualsize; j++ )  {
                float x = 1.0;
                if ( i != j ) {
                    x = 0.0;
                    for ( int k = i; k < j; k++ ) 
                        x -= data[j*maxsize+k]*data[k*maxsize+i];
                }
                data[j*maxsize+i] = x / data[j*maxsize+j];
            }
        for ( int i = 0; i < actualsize; i++ )   // invert U
            for ( int j = i; j < actualsize; j++ )  {
                if ( i == j ) continue;
                float sum = 0.0;
                for ( int k = i; k < j; k++ )
                    sum += data[k*maxsize+j]*( (i==k) ? 1.0 : data[i*maxsize+k] );
                data[i*maxsize+j] = -sum;
            }
        for ( int i = 0; i < actualsize; i++ )   // final inversion
            for ( int j = 0; j < actualsize; j++ )  {
                float sum = 0.0;
                for ( int k = ((i>j)?i:j); k < actualsize; k++ )  
                    sum += ((j==k)?1.0:data[j*maxsize+k])*data[k*maxsize+i];
                data[j*maxsize+i] = sum;
            }

        /*DEBUG("\n\nR matrix after inversion:\n");
        for(int i=0; i<n; i++) {
            for(int j=0; j<n; j++) {
                DEBUG("%.2f ",data[i*n+j]);
            }
            DEBUG("\n");
        }*/
    } else {
        PRINT("Error: Invalid dimensionality for invert(...)\n");
    }
 }


/*
 * Another matrix inversion function
 * This was modified from the 'cluster' application by Charles A. Bouman
 */
int invert_matrix(float* a, int n, float* determinant) {
    int  i,j,f,g;
   
    float* y = (float*) malloc(sizeof(float)*n*n);
    float* col = (float*) malloc(sizeof(float)*n);
    int* indx = (int*) malloc(sizeof(int)*n);

    printf("\n\nR matrix before LU decomposition:\n");
    for(i=0; i<n; i++) {
        for(j=0; j<n; j++) {
            printf("%.2f ",a[i*n+j]);
        }
        printf("\n");
    }

    *determinant = 0.0;
    if(ludcmp(a,n,indx,determinant)) {
        printf("Determinant mantissa after LU decomposition: %f\n",*determinant);
        printf("\n\nR matrix after LU decomposition:\n");
        for(i=0; i<n; i++) {
            for(j=0; j<n; j++) {
                printf("%.2f ",a[i*n+j]);
            }
            printf("\n");
        }
       
      for(j=0; j<n; j++) {
        *determinant *= a[j*n+j];
      }
     
      printf("determinant: %E\n",*determinant);
     
      for(j=0; j<n; j++) {
        for(i=0; i<n; i++) col[i]=0.0;
        col[j]=1.0;
        lubksb(a,n,indx,col);
        for(i=0; i<n; i++) y[i*n+j]=col[i];
      }

      for(i=0; i<n; i++)
      for(j=0; j<n; j++) a[i*n+j]=y[i*n+j];
     
      printf("\n\nMatrix at end of clust_invert function:\n");
      for(f=0; f<n; f++) {
          for(g=0; g<n; g++) {
              printf("%.2f ",a[f*n+g]);
          }
          printf("\n");
      }
      free(y);
      free(col);
      free(indx);
      return(1);
    }
    else {
        *determinant = 0.0;
        free(y);
        free(col);
        free(indx);
        return(0);
    }
}

static float double_abs(float x)
{
       if(x<0) x = -x;
       return(x);
}

#define TINY 1.0e-20

static int
ludcmp(float *a,int n,int *indx,float *d)
{
    int i,imax,j,k;
    float big,dum,sum,temp;
    float *vv;

    vv= (float*) malloc(sizeof(float)*n);
   
    *d=1.0;
   
    for (i=0;i<n;i++)
    {
        big=0.0;
        for (j=0;j<n;j++)
            if ((temp=fabsf(a[i*n+j])) > big)
                big=temp;
        if (big == 0.0)
            return 0; /* Singular matrix  */
        vv[i]=1.0/big;
    }
       
   
    for (j=0;j<n;j++)
    {  
        for (i=0;i<j;i++)
        {
            sum=a[i*n+j];
            for (k=0;k<i;k++)
                sum -= a[i*n+k]*a[k*n+j];
            a[i*n+j]=sum;
        }
       
        /*
        int f,g;
        printf("\n\nMatrix After Step 1:\n");
        for(f=0; f<n; f++) {
            for(g=0; g<n; g++) {
                printf("%.2f ",a[f*n+g]);
            }
            printf("\n");
        }*/
       
        big=0.0;
        dum=0.0;
        for (i=j;i<n;i++)
        {
            sum=a[i*n+j];
            for (k=0;k<j;k++)
                sum -= a[i*n+k]*a[k*n+j];
            a[i*n+j]=sum;
            dum=vv[i]*fabsf(sum);
            //printf("sum: %f, dum: %f, big: %f\n",sum,dum,big);
            //printf("dum-big: %E\n",fabs(dum-big));
            if ( (dum-big) >= 0.0 || fabs(dum-big) < 1e-3)
            {
                big=dum;
                imax=i;
                //printf("imax: %d\n",imax);
            }
        }
       
        if (j != imax)
        {
            for (k=0;k<n;k++)
            {
                dum=a[imax*n+k];
                a[imax*n+k]=a[j*n+k];
                a[j*n+k]=dum;
            }
            *d = -(*d);
            vv[imax]=vv[j];
        }
        indx[j]=imax;
       
        /*
        printf("\n\nMatrix after %dth iteration of LU decomposition:\n",j);
        for(f=0; f<n; f++) {
            for(g=0; g<n; g++) {
                printf("%.2f ",a[f][g]);
            }
            printf("\n");
        }
        printf("imax: %d\n",imax);
        */


        /* Change made 3/27/98 for robustness */
        if ( (a[j*n+j]>=0)&&(a[j*n+j]<TINY) ) a[j*n+j]= TINY;
        if ( (a[j*n+j]<0)&&(a[j*n+j]>-TINY) ) a[j*n+j]= -TINY;

        if (j != n-1)
        {
            dum=1.0/(a[j*n+j]);
            for (i=j+1;i<n;i++)
                a[i*n+j] *= dum;
        }
    }
    free(vv);
    return(1);
}

#undef TINY

static void
lubksb(float *a,int n,int *indx,float *b)
{
    int i,ii,ip,j;
    float sum;

    ii = -1;
    for (i=0;i<n;i++)
    {
        ip=indx[i];
        sum=b[ip];
        b[ip]=b[i];
        if (ii >= 0)
            for (j=ii;j<i;j++)
                sum -= a[i*n+j]*b[j];
        else if (sum)
            ii=i;
        b[i]=sum;
    }
    for (i=n-1;i>=0;i--)
    {
        sum=b[i];
        for (j=i+1;j<n;j++)
            sum -= a[i*n+j]*b[j];
        b[i]=sum/a[i*n+i];
    }
}

