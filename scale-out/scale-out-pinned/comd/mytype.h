/// \file
/// Frequently needed typedefs.

#ifndef __MYTYPE_H_
#define __MYTYPE_H_

/// \def SINGLE determines whether single or double precision is built
#ifdef COMD_SINGLE
typedef float real_t;  //!< define native type for CoMD as single precision
  #define FMT1 "%g"    //!< /def format argument for floats 
  #define EMT1 "%e"    //!< /def format argument for eng floats
#else
typedef double real_t; //!< define native type for CoMD as double precision
  #define FMT1 "%lg"   //!< \def format argument for doubles 
  #define EMT1 "%le"   //!< \def format argument for eng doubles 
#endif

typedef real_t real3_old[3]; //!< a convenience vector with three real_t 

typedef struct vec_t {
  real_t* x;
  real_t* y;
  real_t* z;
} vec_t;

typedef struct real3_t {
  real_t x;
  real_t y;
  real_t z;
} real3_t;

typedef struct int3_t {
  int x;
  int y;
  int z;
} int3_t;

static void zeroReal3_old(real3_old a)
{
   a[0] = 0.0;
   a[1] = 0.0;
   a[2] = 0.0;
}

void malloc_vec(vec_t *ptr, int nEntries);

void free_vec(vec_t *ptr);

void zeroVec(vec_t* a, int iOff);

void zeroVecAll(vec_t* a, int n);

void copyVec(vec_t* a, int iOff, int jOff);


#define screenOut stdout

#endif
