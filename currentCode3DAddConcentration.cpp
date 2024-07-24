///// negative temperature -0.25 give nan, -0.5 and -0.01 work?
///// changed sign of ft_Axxxc terms to have proper sign - then temp>0 melting, <0 freezing
///// gives freezing at - 0.05
///// add concentration field, 1st fixed values just to check up to which value where no c-dynamics necessary (roughly) pd is reproduced 
///// add deltaEpsilon for miscibility gap
///// add veg Law in operator, but for matching pd 1st exclude this effect as quantitative estimate for coexistence shift due to elastic non trivial 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex>
#include <iostream>

#include <fftw3.h>

#ifdef USE_MPI
#include <mpi.h>
#else
#include "mpiDummy.h"
#endif

using namespace std;

typedef complex<double> Complex;

#define PI (2.0*asin(1.0))

#define NTHREADS 1

// abort criteria
// #define SOFTTIMELIMIT (1) // use a negative time to switch off this criterion
#define SOFTTIMELIMIT (-1)
#define TIMESTEPS (5000)  // use a negative value to switch off this criterion
// #define TIMESTEPS (-1)

#define TRUE 1
#define FALSE 0

#define I (Complex(0, 1))
#define PI (2.0*asin(1.0))

#define EPSILON 0.1   // only important if USE_SQUARE_OPERATOR == TRUE
#define AS 1.0
#define F0 1.0    // -n0*kb*TM/2 * C'' * q0^(-1) us^2 EPSILON^(-1/2)

#define PHI 0.1   // rotation of the grain

#define X 64.0
#define Y 64.0
#define Z 2.0

#define NX 128  // number of grid points in x direction
#define NY 128  // number of grid points in y direction
#define NZ 4  // number of grid points in z direction

#define DX (X/NX)
#define DY (Y/NY)
#define DZ (Z/NZ)

#define DT 0.1  // timestep

#define SAVE 10
#define SAVE_IMAGE 100000    // save image after SAVE_IMAGE timesteps
#define SAVE_VTK 50

#define USE_SQUARE_OPERATOR TRUE

#define BURGERS (1.0 * 2*sqrt(2.0)*PI*sqrt(EPSILON))

// vtk output information
#define FINEGRID 1
#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0
#define ZMIN 0.0
#define ZMAX 1.0

// perturbed cylinder. Set amplitude to zero for a straight cylinder
#define PERTURBATION_AMPLITUDE 0.0
#define PERTURBATION_N 3

/*
 save and restore fftw wisdom in a file. This is mainly important if binary compatilibity
 of runs is an issue.
 */
#define USE_WISDOM TRUE

#define VOLUME_CONSTRAINT FALSE

#define TEMP_COUPLING TRUE

#define CONC 0.575
#define ALPHA 0.1


/* inhomogeneous temperature profile */
#define TSOLID -1.0
#define TLIQUID 1.0
#define TRADIUS (X/1.0)
#define TETA 1.0      // interface thickness

const double k011[] = {0.0, 1.0/sqrt(2.0), 1.0/sqrt(2.0)};
const double k101[] = {1.0/sqrt(2.0), 0.0, 1.0/sqrt(2.0)};
const double k110[] = {1.0/sqrt(2.0), 1.0/sqrt(2.0), 0.0};
const double k01m1[] = {0.0, 1.0/sqrt(2.0), -1.0/sqrt(2.0)};
const double k10m1[] = {1.0/sqrt(2.0), 0.0, -1.0/sqrt(2.0)};
const double k1m10[] = {1.0/sqrt(2.0), -1.0/sqrt(2.0), 0.0};

struct param_t
{
  int n;   // time index;
};

float delta_n_Of_r(int i, int j, int k, const double inVec1[],const double inVec2[],const double inVec3[],const double inVec4[],const double inVec5[],const double inVec6[], Complex ***A011, Complex ***A101, Complex ***A110, 
				   Complex ***A01m1, Complex ***A10m1, Complex ***A1m10);

/*************************************************************************/
Complex **fftw_cmatrix(const int nrh, const int nch)
/*
 allocate a complex matrix with subscript range m[0..nrh-1][0..nch-1]
 using the fftw allocation routine
 */
{
  int i, nrow=nrh, ncol=nch;
  Complex **m;
  
  /* allocate pointers to rows */
  m=(Complex **) fftw_malloc((size_t)(nrow*sizeof(Complex*)));
  if (!m) 
    {
      printf("allocation failure 1 in fftw_cmatrix()");
      exit(1);
    }
  
  /* allocate rows and set pointers to them */
  /*
   For fftw it is important that the actual data is stored in a
   CONTIGOUS block
   */
  m[0]=(Complex *) fftw_malloc((size_t)(nrow*ncol*sizeof(Complex)));
  if (!m[0])
    {
      printf("allocation failure 2 in fftw_cmatrix()");
      exit(1);
    }
  
  for (i=1; i<nrh; i++)
    m[i] = m[i-1] + ncol;
  
  /* return pointer to array of pointers to rows */
  return m;
}

/*************************************************************************/
void fftw_free_cmatrix(Complex **m)
/* free a complex matrix allocated by fftw_cmatrix() */
{
  fftw_free(m[0]);   // free the data block
  fftw_free(m);      // free the pointer vector
}

/*************************************************************************/
Complex ***fftw_ctensor(const int nrh, const int nch, const int ndh)
/* allocate a complex rank 3 tensor with range [0..nrh-1][0..nch-1][0..ndh-1] */
{
  int i, j, nrow=nrh, ncol=nch, ndep=ndh;
  Complex ***t;
  
  /* allocate pointers to pointers to rows */
  t=(Complex ***) malloc((size_t)(nrow*sizeof(Complex**)));
  if (!t)
	{
	  printf("allocation failure 1 in fftw_ctensor()");
	  exit(1);
	}
  
  /* allocate pointers to rows and set pointers to them */
  t[0]=(Complex **) malloc((size_t)((nrow*ncol)*sizeof(Complex*)));
  if (!t[0]) 
	{
	  printf("allocation failure 2 in fftw_ctensor()");
	}
  
  /* allocate rows and set pointers to them */
  t[0][0]=(Complex *) malloc((size_t)((nrow*ncol*ndep)*sizeof(Complex)));
  if (!t[0][0]) 
	{
	  printf("allocation failure 3 in fftw_ctensor()");
	  exit(1);
	}
  
  for(j=1; j<nch; j++)
	t[0][j]=t[0][j-1]+ndep;
  
  for(i=1; i<nrh; i++)
	{
	  t[i]=t[i-1]+ncol;
	  t[i][0]=t[i-1][0]+ncol*ndep;
	  for(j=1; j<nch; j++)
		t[i][j]=t[i][j-1]+ndep;
	}
  
  /* return pointer to array of pointers to rows */
  return t;
}

/*************************************************************************/
void fftw_free_ctensor(Complex ***t)
{
  free(t[0][0]);
  free(t[0]);
  free(t);
}

/*************************************************************************/
Complex dx_op(const int jx, const int jy, const int jz)
{
  int jx_corr;
  
  if (2*jx<NX)
    jx_corr = jx;
  else
    jx_corr = jx-NX;
  
  return 2.0*jx_corr*PI*I/X;
}

/*************************************************************************/
Complex dy_op(const int jx, const int jy, const int jz)
{
  int jy_corr;
  
  if (2*jy<NY)
    jy_corr = jy;
  else
    jy_corr = jy - NY;
  
  return 2.0*jy_corr*PI*I/Y;
}

/*************************************************************************/
Complex dz_op(const int jx, const int jy, const int jz)
{
  int jz_corr;
  
  if (2*jz<NZ)
    jz_corr = jz;
  else
    jz_corr = jz - NZ;
  
  return 2.0*jz_corr*PI*I/Z;
}

/*************************************************************************/
Complex dx2_op(const int jx, const int jy, const int jz)
{
  int jx_corr;
  
  if (2*jx<NX)
    jx_corr = jx;
  else
    jx_corr = jx-NX;
  
  return -4.0*jx_corr*jx_corr*PI*PI/(X*X);
}

/*************************************************************************/
Complex dy2_op(const int jx, const int jy, const int jz)
{
  int jy_corr;
  
  if (2*jy<NY)
    jy_corr = jy;
  else
    jy_corr = jy - NY;
  
  return -4.0*jy_corr*jy_corr*PI*PI/(Y*Y);
}

/*************************************************************************/
Complex dz2_op(const int jx, const int jy, const int jz)
{
  int jz_corr;
  
  if (2*jz<NZ)
    jz_corr = jz;
  else
    jz_corr = jz - NZ;
  
  return -4.0*jz_corr*jz_corr*PI*PI/(Z*Z);
}

/*************************************************************************/
/*
 this contribution is the same for all fields. xxx can be 110, 101, 011, 1m10,
 10m1, 01m1.
 */
Complex fp1_Axxxc(const Complex Axxx)
{
  return F0 * Axxx/12.0;
}

/*******************************************************************************/
Complex fp1(const Complex A011, const Complex A101, const Complex A110, 
            const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return F0/12.0 * (A011*A011c+A101*A101c+A110*A110c+A01m1*A01m1c+A10m1*A10m1c+A1m10*A1m10c);
}

/***********************************************************************************/
Complex fp2_A011c(const Complex A011, const Complex A101, const Complex A110, 
                  const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return F0/90.0 * A011*(A011*A011c+2.0*A101*A101c+2.0*A110*A110c+2.0*A01m1*A01m1c+2.0*A10m1*A10m1c+2.0*A1m10*A1m10c);
}

/***********************************************************************************/
Complex fp2_A101c(const Complex A011, const Complex A101, const Complex A110, 
                  const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return F0/90.0 * A101*(2.0*A011*A011c+A101*A101c+2.0*A110*A110c+2.0*A01m1*A01m1c+2.0*A10m1*A10m1c+2.0*A1m10*A1m10c);
}

/***********************************************************************************/
Complex fp2_A110c(const Complex A011, const Complex A101, const Complex A110, 
                  const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return F0/90.0 * A110*(2.0*A011*A011c+2.0*A101*A101c+A110*A110c+2.0*A01m1*A01m1c+2.0*A10m1*A10m1c+2.0*A1m10*A1m10c);
}

/***********************************************************************************/
Complex fp2_A01m1c(const Complex A011, const Complex A101, const Complex A110, 
                   const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return F0/90.0 * A01m1*(2.0*A011*A011c+2.0*A101*A101c+2.0*A110*A110c+A01m1*A01m1c+2.0*A10m1*A10m1c+2.0*A1m10*A1m10c);
}

/***********************************************************************************/
Complex fp2_A10m1c(const Complex A011, const Complex A101, const Complex A110, 
                   const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return F0/90.0 * A10m1*(2.0*A011*A011c+2.0*A101*A101c+2.0*A110*A110c+2.0*A01m1*A01m1c+A10m1*A10m1c+2.0*A1m10*A1m10c);
}

/***********************************************************************************/
Complex fp2_A1m10c(const Complex A011, const Complex A101, const Complex A110, 
                   const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return F0/90.0 * A1m10*(2.0*A011*A011c+2.0*A101*A101c+2.0*A110*A110c+2.0*A01m1*A01m1c+2.0*A10m1*A10m1c+A1m10*A1m10c);
}

/*********************************************************************************************/
/* should be real ... */
Complex fp2(const Complex A011, const Complex A101, const Complex A110, 
            const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return F0*(pow(A011*A011c+A101*A101c+A110*A110c+A01m1*A01m1c+A10m1*A10m1c+A1m10*A1m10c,2.0)
			 -1.0/2.0*A011*A011*A011c*A011c-1.0/2.0*A101*A101*A101c*A101c-1.0/2.0*A110*A110*A110c*A110c
			 -1.0/2.0*A01m1*A01m1*A01m1c*A01m1c-1.0/2.0*A10m1*A10m1*A10m1c*A10m1c
			 -1.0/2.0*A1m10*A1m10*A1m10c*A1m10c) / 90.0;
}

/*********************************************************************************************/
Complex fp3_A011c(const Complex A011, const Complex A101, const Complex A110, 
                  const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return 2.0*F0/90.0 * (A1m10c*A01m1c*A110+A01m1*A10m1c*A101);
}

/*********************************************************************************************/
Complex fp3_A101c(const Complex A011, const Complex A101, const Complex A110, 
                  const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return 2.0*F0/90.0 * (A110*A1m10*A10m1c+A01m1c*A10m1*A011);
}

/*********************************************************************************************/
Complex fp3_A110c(const Complex A011, const Complex A101, const Complex A110, 
                  const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return 2.0*F0/90.0 * (A1m10c*A101*A10m1+A1m10*A011*A01m1);
}

/*********************************************************************************************/
Complex fp3_A01m1c(const Complex A011, const Complex A101, const Complex A110, 
                   const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return 2.0*F0/90.0 * (A1m10c*A011c*A110+A10m1*A101c*A011);
}

/*********************************************************************************************/
Complex fp3_A10m1c(const Complex A011, const Complex A101, const Complex A110, 
                   const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return 2.0*F0/90.0 * (A110*A1m10*A101c+A01m1*A101*A011c);
}

/*********************************************************************************************/
Complex fp3_A1m10c(const Complex A011, const Complex A101, const Complex A110, 
                   const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return 2.0*F0/90.0 * (A110c*A101*A10m1+A011c*A01m1c*A110);
}

/***************************************************************************************/
/* should be real ... */
Complex fp3(const Complex A011, const Complex A101, const Complex A110, 
            const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return 2.0*F0/90.0 * (A110c*A1m10c*A101*A10m1+A110*A1m10*A101c*A10m1c+A1m10*A011*A01m1*A110c
						+A1m10c*A011c*A01m1c*A110+A01m1*A10m1c*A101*A011c+A01m1c*A10m1*A101c*A011);
}

/***************************************************************************************/
Complex fp4_A011c(const Complex A011, const Complex A101, const Complex A110, 
                  const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return -F0/8.0 * (A101*A1m10c+A110*A10m1c);
}

/***********************************************************************************/
Complex fp4_A101c(const Complex A011, const Complex A101, const Complex A110, 
                  const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return -F0/8.0 * (A011*A1m10+A01m1c*A110);
}

/***********************************************************************************/
Complex fp4_A110c(const Complex A011, const Complex A101, const Complex A110, 
                  const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return -F0/8.0 * (A011*A10m1+A01m1*A101);
}

/***********************************************************************************/
Complex fp4_A01m1c(const Complex A011, const Complex A101, const Complex A110, 
                   const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return -F0/8.0 * (A110*A101c+A10m1*A1m10c);
}

/***********************************************************************************/
Complex fp4_A10m1c(const Complex A011, const Complex A101, const Complex A110, 
                   const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return -F0/8.0 * (A011c*A110+A01m1*A1m10);
}

/***********************************************************************************/
Complex fp4_A1m10c(const Complex A011, const Complex A101, const Complex A110, 
                   const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return -F0/8.0 * (A011c*A101+A01m1c*A10m1);
}

/***********************************************************************************/
/* should be real ... */
Complex fp4(const Complex A011, const Complex A101, const Complex A110, 
            const Complex A01m1, const Complex A10m1, const Complex A1m10)
{
  Complex A011c, A101c, A110c, A01m1c, A10m1c, A1m10c;
  
  A011c = conj(A011);
  A101c = conj(A101);
  A110c = conj(A110);
  A01m1c = conj(A01m1);
  A10m1c = conj(A10m1);
  A1m10c = conj(A1m10);
  
  return -F0/8.0 * (A011c*A101*A1m10c+A011*A101c*A1m10+A011c*A110*A10m1c+A011*A110c*A10m1
					+A01m1c*A110*A101c+A01m1*A110c*A101+A01m1c*A10m1*A1m10c+A01m1*A10m1c*A1m10);
}

/***********************************************************************************/
/*
 Contains the box operator (or only the k*n term if USE_SQUARE_OPERATOR == FALSE).
 
 L = -Box^2 + 1.0/12.0
 
 where Box depends on k*n.
 */
Complex LinearOperatorSpectral(const double kx, const double ky, const double kz,
							   const int jx, const int jy, const int jz)
{
  Complex square;
  
#if (USE_SQUARE_OPERATOR == TRUE)
  square = kx*dx_op(jx, jy, jz) + ky*dy_op(jx, jy, jz) + kz*dz_op(jx, jy, jz)
  - 0.5*sqrt(EPSILON)*I*(dx2_op(jx, jy, jz) + dy2_op(jx, jy, jz) + dz2_op(jx, jy, jz));
#else
  square = kx*dx_op(jx, jy, jz) + ky*dy_op(jx, jy, jz) + kz*dz_op(jx, jy, jz);
#endif
  
  return F0 * (-square*square + 1.0/12.0);
}

/******************************************************************************************/
/* tilt function */
Complex h(const Complex x)
{
  return x*x*(3.0-2.0*x);
}

/******************************************************************************************/
/* tilt function derivative*/
Complex hprim(const Complex x)
{
  return 6.0*x*(1.0-x);
}

/******************************************************************************************/
/*
 A011, A101, A110, A01m1, A10m1, A1m10 contain the current fields, and they will be
 overwritten by the new value
 
 n is the timestep, which is only needed to calculate the time derivative in the first
 iteration
 */
void timestep_spectral(Complex ***A011, Complex ***A101, Complex ***A110,
                       Complex ***A01m1, Complex ***A10m1, Complex ***A1m10,
                       Complex ***A011fft, Complex ***A101fft, Complex ***A110fft,
                       Complex ***A01m1fft, Complex ***A10m1fft, Complex ***A1m10fft,
                       Complex ***A011inhom, Complex ***A101inhom, Complex ***A110inhom,
                       Complex ***A01m1inhom, Complex ***A10m1inhom, Complex ***A1m10inhom,
                       Complex ***A011inhomfft, Complex ***A101inhomfft, Complex ***A110inhomfft,
                       Complex ***A01m1inhomfft, Complex ***A10m1inhomfft, Complex ***A1m10inhomfft,
                       Complex ***A011inhomprevfft, Complex ***A101inhomprevfft, Complex ***A110inhomprevfft,
                       Complex ***A01m1inhomprevfft, Complex ***A10m1inhomprevfft, Complex ***A1m10inhomprevfft,
                       Complex ***A011newfft, Complex ***A101newfft, Complex ***A110newfft,
                       Complex ***A01m1newfft, Complex ***A10m1newfft, Complex ***A1m10newfft,
					   Complex ***temperature,
                       fftw_plan* p011, fftw_plan* p101, fftw_plan* p110,
                       fftw_plan* p01m1, fftw_plan* p10m1, fftw_plan* p1m10,
                       fftw_plan* p011inhom, fftw_plan* p101inhom, fftw_plan* p110inhom,
                       fftw_plan* p01m1inhom, fftw_plan* p10m1inhom, fftw_plan* p1m10inhom,
                       fftw_plan* p011new, fftw_plan* p101new, fftw_plan* p110new, 
                       fftw_plan* p01m1new, fftw_plan* p10m1new, fftw_plan* p1m10new,
                       const int n)
{
  Complex L011op, L101op, L110op, L01m1op, L10m1op, L1m10op;
  Complex ft_A011c, ft_A101c, ft_A110c, ft_A01m1c, ft_A10m1c, ft_A1m10c;
  int i, j, k;
#if (VOLUME_CONSTRAINT == TRUE)
  Complex numerator_nonlin=0, numerator_lin=0, volume=0, lagrange;
#endif
  
  /*
   Calculate Fourier transform of the modes
   
   Notice that the information of the fields in contained in the plans.
   The input fields are here A011, A101, A110, A01m1, A10m1, A1m10.
   The output fields are A011fft, A101fft, A110fft, A01m1fft, A10m1fft, A1m10fft (UNNORMALIZED)
   */
  fftw_execute(*p011);
  fftw_execute(*p101);
  fftw_execute(*p110);
  fftw_execute(*p01m1);
  fftw_execute(*p10m1);
  fftw_execute(*p1m10);
  
	double concentration(CONC);
	double deltaEpsilon = log(2.0);
	//std::cout << "conc, deltaEpsilon: " << concentration << " " << deltaEpsilon << std::endl;
	//exit(1);
	
  // calculate inhomogeneity
  for (i=0; i<NX; ++i)
    for (j=0; j<NY; ++j)
	  for (k=0; k<NZ; ++k)
		{
		  // thermal tilt
		  /*
		   I write the energy as
		   1/6 * \sum_i (1-h(Ai*Aic/AS^2)) * temperature
		   
		   The factor 1/6 is used to give temperature the interpretation of the energy raise of the liquid.
		   Then the derivative with respect to Aic is
		   -1/6 * h'(Ai*Aic/AS^2) * TEMP * Ai/AS^2
		   */
#if (TEMP_COUPLING == TRUE)
			
			ft_A011c =  1.0/6.0 * hprim(A011[i][j][k]*conj(A011[i][j][k])/(AS*AS)) * A011[i][j][k]/(AS*AS) * (temperature[i][j][k]+concentration*deltaEpsilon);
			ft_A101c =  1.0/6.0 * hprim(A101[i][j][k]*conj(A101[i][j][k])/(AS*AS)) * A101[i][j][k]/(AS*AS) * (temperature[i][j][k]+concentration*deltaEpsilon);
			ft_A110c =  1.0/6.0 * hprim(A110[i][j][k]*conj(A110[i][j][k])/(AS*AS)) * A110[i][j][k]/(AS*AS) * (temperature[i][j][k]+concentration*deltaEpsilon);
			ft_A01m1c =  1.0/6.0 * hprim(A01m1[i][j][k]*conj(A01m1[i][j][k])/(AS*AS)) * A01m1[i][j][k]/(AS*AS) * (temperature[i][j][k]+concentration*deltaEpsilon);
			ft_A10m1c =  1.0/6.0 * hprim(A10m1[i][j][k]*conj(A10m1[i][j][k])/(AS*AS)) * A10m1[i][j][k]/(AS*AS) * (temperature[i][j][k]+concentration*deltaEpsilon);
			ft_A1m10c =  1.0/6.0 * hprim(A1m10[i][j][k]*conj(A1m10[i][j][k])/(AS*AS)) * A1m10[i][j][k]/(AS*AS) * (temperature[i][j][k]+concentration*deltaEpsilon);
			
			// below original temp coupling which corresponds to neg values for melting and pos values for freezing
			/*
		   ft_A011c = - 1.0/6.0 * hprim(A011[i][j][k]*conj(A011[i][j][k])/(AS*AS)) * A011[i][j][k]/(AS*AS) * temperature[i][j][k];
		   ft_A101c = - 1.0/6.0 * hprim(A101[i][j][k]*conj(A101[i][j][k])/(AS*AS)) * A101[i][j][k]/(AS*AS) * temperature[i][j][k];
		   ft_A110c = - 1.0/6.0 * hprim(A110[i][j][k]*conj(A110[i][j][k])/(AS*AS)) * A110[i][j][k]/(AS*AS) * temperature[i][j][k];
		   ft_A01m1c = - 1.0/6.0 * hprim(A01m1[i][j][k]*conj(A01m1[i][j][k])/(AS*AS)) * A01m1[i][j][k]/(AS*AS) * temperature[i][j][k];
		   ft_A10m1c = - 1.0/6.0 * hprim(A10m1[i][j][k]*conj(A10m1[i][j][k])/(AS*AS)) * A10m1[i][j][k]/(AS*AS) * temperature[i][j][k];
		   ft_A1m10c = - 1.0/6.0 * hprim(A1m10[i][j][k]*conj(A1m10[i][j][k])/(AS*AS)) * A1m10[i][j][k]/(AS*AS) * temperature[i][j][k];
			 */
#else
		  ft_A011c = ft_A101c = ft_A110c = ft_A01m1c = ft_A10m1c = ft_A1m10c = 0.0;
#endif
		  
		  // I normalize already here, which is ok since the DFT is linear
		  A011inhom[i][j][k] = (fp2_A011c(A011[i][j][k], A101[i][j][k], A110[i][j][k], A01m1[i][j][k], A10m1[i][j][k], A1m10[i][j][k])
								+ fp3_A011c(A011[i][j][k], A101[i][j][k], A110[i][j][k], A01m1[i][j][k], A10m1[i][j][k], A1m10[i][j][k])
								+ fp4_A011c(A011[i][j][k], A101[i][j][k], A110[i][j][k], A01m1[i][j][k], A10m1[i][j][k], A1m10[i][j][k])
								+ ft_A011c) 
		  / (1.0*NX*NY*NZ);
		  
		  A101inhom[i][j][k] = (fp2_A101c(A011[i][j][k], A101[i][j][k], A110[i][j][k], A01m1[i][j][k], A10m1[i][j][k], A1m10[i][j][k])
								+ fp3_A101c(A011[i][j][k], A101[i][j][k], A110[i][j][k], A01m1[i][j][k], A10m1[i][j][k], A1m10[i][j][k])
								+ fp4_A101c(A011[i][j][k], A101[i][j][k], A110[i][j][k], A01m1[i][j][k], A10m1[i][j][k], A1m10[i][j][k])
								+ ft_A101c) 
		  / (1.0*NX*NY*NZ);
		  
		  A110inhom[i][j][k] = (fp2_A110c(A011[i][j][k], A101[i][j][k], A110[i][j][k], A01m1[i][j][k], A10m1[i][j][k], A1m10[i][j][k])
								+ fp3_A110c(A011[i][j][k], A101[i][j][k], A110[i][j][k], A01m1[i][j][k], A10m1[i][j][k], A1m10[i][j][k])
								+ fp4_A110c(A011[i][j][k], A101[i][j][k], A110[i][j][k], A01m1[i][j][k], A10m1[i][j][k], A1m10[i][j][k])
								+ ft_A110c) 
		  / (1.0*NX*NY*NZ);
		  
		  A01m1inhom[i][j][k] = (fp2_A01m1c(A011[i][j][k], A101[i][j][k], A110[i][j][k], A01m1[i][j][k], A10m1[i][j][k], A1m10[i][j][k])
								 + fp3_A01m1c(A011[i][j][k], A101[i][j][k], A110[i][j][k], A01m1[i][j][k], A10m1[i][j][k], A1m10[i][j][k])
								 + fp4_A01m1c(A011[i][j][k], A101[i][j][k], A110[i][j][k], A01m1[i][j][k], A10m1[i][j][k], A1m10[i][j][k])
								 + ft_A01m1c)
		  / (1.0*NX*NY*NZ);
		  
		  A10m1inhom[i][j][k] = (fp2_A10m1c(A011[i][j][k], A101[i][j][k], A110[i][j][k], A01m1[i][j][k], A10m1[i][j][k], A1m10[i][j][k])
								 + fp3_A10m1c(A011[i][j][k], A101[i][j][k], A110[i][j][k], A01m1[i][j][k], A10m1[i][j][k], A1m10[i][j][k])
								 + fp4_A10m1c(A011[i][j][k], A101[i][j][k], A110[i][j][k], A01m1[i][j][k], A10m1[i][j][k], A1m10[i][j][k])
								 + ft_A10m1c)
		  / (1.0*NX*NY*NZ);
		  
		  A1m10inhom[i][j][k] = (fp2_A1m10c(A011[i][j][k], A101[i][j][k], A110[i][j][k], A01m1[i][j][k], A10m1[i][j][k], A1m10[i][j][k])
								 + fp3_A1m10c(A011[i][j][k], A101[i][j][k], A110[i][j][k], A01m1[i][j][k], A10m1[i][j][k], A1m10[i][j][k])
								 + fp4_A1m10c(A011[i][j][k], A101[i][j][k], A110[i][j][k], A01m1[i][j][k], A10m1[i][j][k], A1m10[i][j][k])
								 + ft_A1m10c)
		  / (1.0*NX*NY*NZ);
		  
#if (VOLUME_CONSTRAINT == TRUE)
		  /* 
		   calculate Lagrange multiplier 
		   
		   lambda = Re(sum_j A_j \delta F/\delta A_j) / V
		   
		   The multiplication with DX*DY*DZ cancels
		   */
		  numerator_nonlin += A011[i][j][k]*conj(A011inhom[i][j][k]) + A101[i][j][k]*conj(A101inhom[i][j][k])
		  + A110[i][j][k]*conj(A110inhom[i][j][k]) + A01m1[i][j][k]*conj(A01m1inhom[i][j][k])
		  + A10m1[i][j][k]*conj(A10m1inhom[i][j][k]) + A1m10[i][j][k]*conj(A1m10inhom[i][j][k]);
		  
		  numerator_lin += A011fft[i][j][k]*conj(A011fft[i][j][k])*LinearOperatorSpectral(k011[0], k011[1], k011[2], i, j, k)
		  + A101fft[i][j][k]*conj(A101fft[i][j][k])*LinearOperatorSpectral(k101[0], k101[1], k101[2], i, j, k)
		  + A110fft[i][j][k]*conj(A110fft[i][j][k])*LinearOperatorSpectral(k110[0], k110[1], k110[2], i, j, k)
		  + A01m1fft[i][j][k]*conj(A01m1fft[i][j][k])*LinearOperatorSpectral(k01m1[0], k01m1[1], k01m1[2], i, j, k)
		  + A10m1fft[i][j][k]*conj(A10m1fft[i][j][k])*LinearOperatorSpectral(k10m1[0], k10m1[1], k10m1[2], i, j, k)
		  + A1m10fft[i][j][k]*conj(A1m10fft[i][j][k])*LinearOperatorSpectral(k1m10[0], k1m10[1], k1m10[2], i, j, k);
		  
		  volume += A011[i][j][k]*conj(A011[i][j][k]) + A101[i][j][k]*conj(A101[i][j][k])
		  + A110[i][j][k]*conj(A110[i][j][k]) + A01m1[i][j][k]*conj(A01m1[i][j][k])
		  + A10m1[i][j][k]*conj(A10m1[i][j][k]) + A1m10[i][j][k]*conj(A1m10[i][j][k]);
#endif
		}
 
#if (VOLUME_CONSTRAINT == TRUE)
  lagrange = (numerator_nonlin.real()*(1.0*NX*NY*NZ) + numerator_lin.real()/(1.0*NX*NY*NZ)) / volume;
  
 // printf("%g %g\n", lagrange.real(), lagrange.imag());
#endif
  
  /*
   calculate Fourier transform of the inhomogeneity
   
   Information encoded in the plans:
   Input fields: A011inhom, A101inhom, A110inhom, A01m1inhom, A10m1inhom, A1m10inhom
   Output fields: A011inhomfft, A101inhomfft, A110inhomfft, A01m1inhomfft, A10m1inhomfft, A1m10inhomfft
   */
  fftw_execute(*p011inhom);
  fftw_execute(*p101inhom);
  fftw_execute(*p110inhom);
  fftw_execute(*p01m1inhom);
  fftw_execute(*p10m1inhom);
  fftw_execute(*p1m10inhom);
  
  if (n==0) // to have a well defined value for the derivative in the first timestep
    for (i=0; i<NX; ++i)
      for (j=0; j<NY; ++j)
		for (k=0; k<NZ; ++k)
		  {
			A011inhomprevfft[i][j][k] = A011inhomfft[i][j][k];
			A101inhomprevfft[i][j][k] = A101inhomfft[i][j][k];
			A110inhomprevfft[i][j][k] = A110inhomfft[i][j][k];
			A01m1inhomprevfft[i][j][k] = A01m1inhomfft[i][j][k];
			A10m1inhomprevfft[i][j][k] = A10m1inhomfft[i][j][k];
			A1m10inhomprevfft[i][j][k] = A1m10inhomfft[i][j][k];
		  }
  
  for (i=0; i<NX; ++i)
    for (j=0; j<NY; ++j)
	  for (k=0; k<NZ; ++k)
		{
		  // linear operator, two dimensions!
		  // with Lagrange multiplier
		  L011op = -LinearOperatorSpectral(k011[0], k011[1], k011[2], i, j, k);		
		  L101op = -LinearOperatorSpectral(k101[0], k101[1], k101[2], i, j, k);
		  L110op = -LinearOperatorSpectral(k110[0], k110[1], k110[2], i, j, k);
		  L01m1op = -LinearOperatorSpectral(k01m1[0], k01m1[1], k01m1[2], i, j, k);
		  L10m1op = -LinearOperatorSpectral(k10m1[0], k10m1[1], k10m1[2], i, j, k);
		  L1m10op = -LinearOperatorSpectral(k1m10[0], k1m10[1], k1m10[2], i, j, k);
		  
#if (VOLUME_CONSTRAINT == TRUE)
		  L011op += lagrange;
		  L101op += lagrange;
		  L110op += lagrange;
		  L01m1op += lagrange;
		  L10m1op += lagrange;
		  L1m10op += lagrange;
#endif
		  
		  // contains normalization
		  A011newfft[i][j][k] = A011fft[i][j][k]/(1.0*NX*NY*NZ) * exp(L011op*DT) + (-A011inhomfft[i][j][k])*(exp(L011op*DT)-1.0)/L011op
		  + (-A011inhomfft[i][j][k]-(-A011inhomprevfft[i][j][k]))/DT * (exp(L011op*DT) - 1.0 - L011op*DT) / (L011op*L011op);
		  
		  A101newfft[i][j][k] = A101fft[i][j][k]/(1.0*NX*NY*NZ) * exp(L101op*DT) + (-A101inhomfft[i][j][k])*(exp(L101op*DT)-1.0)/L101op
		  + (-A101inhomfft[i][j][k]-(-A101inhomprevfft[i][j][k]))/DT * (exp(L101op*DT) - 1.0 - L101op*DT) / (L101op*L101op);       
		  
		  A110newfft[i][j][k] = A110fft[i][j][k]/(1.0*NX*NY*NZ) * exp(L110op*DT) + (-A110inhomfft[i][j][k])*(exp(L110op*DT)-1.0)/L110op
		  + (-A110inhomfft[i][j][k]-(-A110inhomprevfft[i][j][k]))/DT * (exp(L110op*DT) - 1.0 - L110op*DT) / (L110op*L110op);        
		  
		  A01m1newfft[i][j][k] = A01m1fft[i][j][k]/(1.0*NX*NY*NZ) * exp(L01m1op*DT) + (-A01m1inhomfft[i][j][k])*(exp(L01m1op*DT)-1.0)/L01m1op
		  + (-A01m1inhomfft[i][j][k]-(-A01m1inhomprevfft[i][j][k]))/DT * (exp(L01m1op*DT) - 1.0 - L01m1op*DT) / (L01m1op*L01m1op);
		  
		  A10m1newfft[i][j][k] = A10m1fft[i][j][k]/(1.0*NX*NY*NZ) * exp(L10m1op*DT) + (-A10m1inhomfft[i][j][k])*(exp(L10m1op*DT)-1.0)/L10m1op
		  + (-A10m1inhomfft[i][j][k]-(-A10m1inhomprevfft[i][j][k]))/DT * (exp(L10m1op*DT) - 1.0 - L10m1op*DT) / (L10m1op*L10m1op); 
		  
		  A1m10newfft[i][j][k] = A1m10fft[i][j][k]/(1.0*NX*NY*NZ) * exp(L1m10op*DT) + (-A1m10inhomfft[i][j][k])*(exp(L1m10op*DT)-1.0)/L1m10op
		  + (-A1m10inhomfft[i][j][k]-(-A1m10inhomprevfft[i][j][k]))/DT * (exp(L1m10op*DT) - 1.0 - L1m10op*DT) / (L1m10op*L1m10op);
		}
  
  /* 
   Backward transformation of the new guess.
   
   Information encoded in the plans:
   Input fields: A011newfft, A101newfft, A110newfft, A01m1newfft, A10m1newfft, A1m10newfft
   Output fields: A011, A101, A110, A01m1, A10m1, A1m10
   
   This means that the original data is overwritten!
   */
  fftw_execute(*p011new);
  fftw_execute(*p101new);
  fftw_execute(*p110new);
  fftw_execute(*p01m1new);
  fftw_execute(*p10m1new);
  fftw_execute(*p1m10new);
}

/*************************************************************************************/
/*
 The rotation is in the xy plane
 */
void init_pure_rotatedxy(Complex ***A011, Complex ***A101, Complex ***A110,
						 Complex ***A01m1, Complex ***A10m1, Complex ***A1m10,
						 const double phi)
{
  int i, j, k;
  double x, y, z, A[3][3], vx, vy, vz;
  
  A[0][0] = cos(phi);
  A[0][1] = sin(phi);
  A[0][2] = 0.0;
  A[1][0] = -sin(phi);
  A[1][1] = cos(phi);
  A[1][2] = 0.0;
  A[2][0] = 0.0;
  A[2][1] = 0.0;
  A[2][2] = 1.0;
  
  for (i=0; i<NX; ++i)
    for (j=0; j<NY; ++j)
	  for (k=0; k<NZ; ++k)
		{
		  /* rotated crystal */
		  x = i*DX;
		  y = j*DY;
		  z = k*DZ;
		  /* v_j = (A_ij - delta_ij) * x_i */
		  vx = (A[0][0] - 1.0) * x + A[0][1] * y + A[0][2] * z;
		  vy = A[1][0] * x + (A[1][1] - 1.0) * y + A[1][2] * z;
		  vz = A[2][0] * x + A[2][1] * y + (A[2][2] - 1.0) * z;
		  
		  A011[i][j][k]  = AS * exp(I * (k011[0]*vx + k011[1]*vy + k011[2]*vz)/sqrt(EPSILON));
		  A101[i][j][k]  = AS * exp(I * (k101[0]*vx + k101[1]*vy + k101[2]*vz)/sqrt(EPSILON));
		  A110[i][j][k]  = AS * exp(I * (k110[0]*vx + k110[1]*vy + k110[2]*vz)/sqrt(EPSILON));
		  
		  A01m1[i][j][k]  = AS * exp(I * (k01m1[0]*vx + k01m1[1]*vy + k01m1[2]*vz)/sqrt(EPSILON));
		  A10m1[i][j][k]  = AS * exp(I * (k10m1[0]*vx + k10m1[1]*vy + k10m1[2]*vz)/sqrt(EPSILON));
		  A1m10[i][j][k]  = AS * exp(I * (k1m10[0]*vx + k1m10[1]*vy + k1m10[2]*vz)/sqrt(EPSILON));
		}
}

/*************************************************************************************/
/*
 The rotation is in the xy plane
 */
void init_symmetric_tilt(Complex ***A011, Complex ***A101, Complex ***A110,
						 Complex ***A01m1, Complex ***A10m1, Complex ***A1m10,
						 const double phi)
{
  int i, j, k;
  double x, y, z, Al[3][3], Ar[3][3], vx, vy, vz;
  
  Al[0][0] = cos(phi);
  Al[0][1] = sin(phi);
  Al[0][2] = 0.0;
  Al[1][0] = -sin(phi);
  Al[1][1] = cos(phi);
  Al[1][2] = 0.0;
  Al[2][0] = 0.0;
  Al[2][1] = 0.0;
  Al[2][2] = 1.0;

  Ar[0][0] = cos(-phi);
  Ar[0][1] = sin(-phi);
  Ar[0][2] = 0.0;
  Ar[1][0] = -sin(-phi);
  Ar[1][1] = cos(-phi);
  Ar[1][2] = 0.0;
  Ar[2][0] = 0.0;
  Ar[2][1] = 0.0;
  Ar[2][2] = 1.0;
  
  for (i=0; i<NX; ++i)
    for (j=0; j<NY; ++j)
	  for (k=0; k<NZ; ++k)
	  {
		/* rotated crystal */
		x = i*DX;
		y = j*DY;
		z = k*DZ;
		
		if (x < X/2.0)
		{
		  /* v_j = (A_ij - delta_ij) * x_i */
		  vx = (Al[0][0] - 1.0) * x + Al[0][1] * y + Al[0][2] * z;
		  vy = Al[1][0] * x + (Al[1][1] - 1.0) * y + Al[1][2] * z;
		  vz = Al[2][0] * x + Al[2][1] * y + (Al[2][2] - 1.0) * z;
		}
		else
		{
		  /* v_j = (A_ij - delta_ij) * x_i */
		  vx = (Ar[0][0] - 1.0) * x + Ar[0][1] * y + Ar[0][2] * z;
		  vy = Ar[1][0] * x + (Ar[1][1] - 1.0) * y + Ar[1][2] * z;
		  vz = Ar[2][0] * x + Ar[2][1] * y + (Ar[2][2] - 1.0) * z;
		}

		A011[i][j][k]  = AS * exp(I * (k011[0]*vx + k011[1]*vy + k011[2]*vz)/sqrt(EPSILON));
		A101[i][j][k]  = AS * exp(I * (k101[0]*vx + k101[1]*vy + k101[2]*vz)/sqrt(EPSILON));
		A110[i][j][k]  = AS * exp(I * (k110[0]*vx + k110[1]*vy + k110[2]*vz)/sqrt(EPSILON));
		
		A01m1[i][j][k]  = AS * exp(I * (k01m1[0]*vx + k01m1[1]*vy + k01m1[2]*vz)/sqrt(EPSILON));
		A10m1[i][j][k]  = AS * exp(I * (k10m1[0]*vx + k10m1[1]*vy + k10m1[2]*vz)/sqrt(EPSILON));
		A1m10[i][j][k]  = AS * exp(I * (k1m10[0]*vx + k1m10[1]*vy + k1m10[2]*vz)/sqrt(EPSILON));
	  }
}

/*************************************************************************************/
/*
 The rotation is in the xy plane
 */
void init_solid_liquid(Complex ***A011, Complex ***A101, Complex ***A110,
					   Complex ***A01m1, Complex ***A10m1, Complex ***A1m10,
					   const double phi)
{
  int i, j, k;
  double x, y, z, A[3][3], vx, vy, vz;
  
  A[0][0] = cos(phi);
  A[0][1] = sin(phi);
  A[0][2] = 0.0;
  A[1][0] = -sin(phi);
  A[1][1] = cos(phi);
  A[1][2] = 0.0;
  A[2][0] = 0.0;
  A[2][1] = 0.0;
  A[2][2] = 1.0;
  
  for (i=0; i<NX; ++i)
    for (j=0; j<NY; ++j)
	  for (k=0; k<NZ; ++k)
		{
		  /* rotated crystal */
		  x = i*DX;
		  y = j*DY;
		  z = k*DZ;
		  /* v_j = (A_ij - delta_ij) * x_i */
		  vx = (A[0][0] - 1.0) * x + A[0][1] * y + A[0][2] * z;
		  vy = A[1][0] * x + (A[1][1] - 1.0) * y + A[1][2] * z;
		  vz = A[2][0] * x + A[2][1] * y + (A[2][2] - 1.0) * z;
		  
		  if (x < X/2)
			{
			  A011[i][j][k]  = AS * exp(I * (k011[0]*vx + k011[1]*vy + k011[2]*vz)/sqrt(EPSILON));
			  A101[i][j][k]  = AS * exp(I * (k101[0]*vx + k101[1]*vy + k101[2]*vz)/sqrt(EPSILON));
			  A110[i][j][k]  = AS * exp(I * (k110[0]*vx + k110[1]*vy + k110[2]*vz)/sqrt(EPSILON));
			  
			  A01m1[i][j][k]  = AS * exp(I * (k01m1[0]*vx + k01m1[1]*vy + k01m1[2]*vz)/sqrt(EPSILON));
			  A10m1[i][j][k]  = AS * exp(I * (k10m1[0]*vx + k10m1[1]*vy + k10m1[2]*vz)/sqrt(EPSILON));
			  A1m10[i][j][k]  = AS * exp(I * (k1m10[0]*vx + k1m10[1]*vy + k1m10[2]*vz)/sqrt(EPSILON));
			}
		  else
			{
			  A011[i][j][k] = 0;
			  A101[i][j][k] = 0;
			  A110[i][j][k] = 0;
			  
			  A01m1[i][j][k] = 0;
			  A10m1[i][j][k] = 0;
			  A1m10[i][j][k] = 0;
			}
		}
}

/*************************************************************************************/
/*
 The rotation is in the xy plane
 */
void init_wire(Complex ***A011, Complex ***A101, Complex ***A110,
			   Complex ***A01m1, Complex ***A10m1, Complex ***A1m10,
			   const double phi, const double radius)
{
  int i, j, k;
  double x, y, z, A[3][3], vx, vy, vz;
  
  A[0][0] = cos(phi);
  A[0][1] = sin(phi);
  A[0][2] = 0.0;
  A[1][0] = -sin(phi);
  A[1][1] = cos(phi);
  A[1][2] = 0.0;
  A[2][0] = 0.0;
  A[2][1] = 0.0;
  A[2][2] = 1.0;
  
  for (i=0; i<NX; ++i)
    for (j=0; j<NY; ++j)
	  for (k=0; k<NZ; ++k)
		{
		  /* rotated crystal */
		  x = i*DX;
		  y = j*DY;
		  z = k*DZ;
		  /* v_j = (A_ij - delta_ij) * x_i */
		  vx = (A[0][0] - 1.0) * x + A[0][1] * y + A[0][2] * z;
		  vy = A[1][0] * x + (A[1][1] - 1.0) * y + A[1][2] * z;
		  vz = A[2][0] * x + A[2][1] * y + (A[2][2] - 1.0) * z;
		  
		  if ((x-X/2.0)*(x-X/2.0)+(y-Y/2.0)*(y-Y/2.0) < radius*radius)
			{
			  A011[i][j][k]  = AS * exp(I * (k011[0]*vx + k011[1]*vy + k011[2]*vz)/sqrt(EPSILON));
			  A101[i][j][k]  = AS * exp(I * (k101[0]*vx + k101[1]*vy + k101[2]*vz)/sqrt(EPSILON));
			  A110[i][j][k]  = AS * exp(I * (k110[0]*vx + k110[1]*vy + k110[2]*vz)/sqrt(EPSILON));
			  
			  A01m1[i][j][k]  = AS * exp(I * (k01m1[0]*vx + k01m1[1]*vy + k01m1[2]*vz)/sqrt(EPSILON));
			  A10m1[i][j][k]  = AS * exp(I * (k10m1[0]*vx + k10m1[1]*vy + k10m1[2]*vz)/sqrt(EPSILON));
			  A1m10[i][j][k]  = AS * exp(I * (k1m10[0]*vx + k1m10[1]*vy + k1m10[2]*vz)/sqrt(EPSILON));
			}
		  else
			{
			  A011[i][j][k] = 0;
			  A101[i][j][k] = 0;
			  A110[i][j][k] = 0;
			  
			  A01m1[i][j][k] = 0;
			  A10m1[i][j][k] = 0;
			  A1m10[i][j][k] = 0;
			}
		}
}

/*************************************************************************************/
/*
 The rotation is in the xy plane
 */
void init_wire_screw(Complex ***A011, Complex ***A101, Complex ***A110,
					 Complex ***A01m1, Complex ***A10m1, Complex ***A1m10,
					 const double radius, const double b, const double x0,
					 const double y0)
{
  int i, j, k;
  double x, y, z, ux, uy, uz, alpha, beta, r, R;
    
  for (i=0; i<NX; ++i)
    for (j=0; j<NY; ++j)
	  for (k=0; k<NZ; ++k)
		{
		  /* rotated crystal */
		  x = i*DX;
		  y = j*DY;
		  z = k*DZ;
		  
		  // winding number is prefactor
		  alpha = 1 * 2.0*PI*z/Z;
		  beta = atan2(x-X/2.0, y-Y/2.0);
		  // notice that torsion is around the center of the wire, not the dislocation core 
		  r = sqrt((x-X/2.0)*(x-X/2.0) + (y-Y/2.0)*(y-Y/2.0))*sqrt(EPSILON);
		  
		  R = radius + PERTURBATION_AMPLITUDE * cos(2.0*PI * PERTURBATION_N * z/Z);
		  
		  if ((x-X/2.0)*(x-X/2.0)+(y-Y/2.0)*(y-Y/2.0) < R*R)
			{
			  ux = r*(cos(alpha+beta) - cos(beta));
			  uy = r*(sin(alpha+beta) - sin(beta));
			  uz = b/(2.0*PI)*atan2(x-x0, y-y0);
			  
			  A011[i][j][k]  = AS * exp(-I * (k011[0]*ux + k011[1]*uy + k011[2]*uz)/sqrt(EPSILON));
			  A101[i][j][k]  = AS * exp(-I * (k101[0]*ux + k101[1]*uy + k101[2]*uz)/sqrt(EPSILON));
			  A110[i][j][k]  = AS * exp(-I * (k110[0]*ux + k110[1]*uy + k110[2]*uz)/sqrt(EPSILON));
			  
			  A01m1[i][j][k]  = AS * exp(-I * (k01m1[0]*ux + k01m1[1]*uy + k01m1[2]*uz)/sqrt(EPSILON));
			  A10m1[i][j][k]  = AS * exp(-I * (k10m1[0]*ux + k10m1[1]*uy + k10m1[2]*uz)/sqrt(EPSILON));
			  A1m10[i][j][k]  = AS * exp(-I * (k1m10[0]*ux + k1m10[1]*uy + k1m10[2]*uz)/sqrt(EPSILON));
			}
		  else
			{
			  A011[i][j][k] = 0;
			  A101[i][j][k] = 0;
			  A110[i][j][k] = 0;
			  
			  A01m1[i][j][k] = 0;
			  A10m1[i][j][k] = 0;
			  A1m10[i][j][k] = 0;
			}
		}
}

/*************************************************************************************/
void save(char *filename, Complex ***A011, Complex ***A101, Complex ***A110, 
          Complex ***A01m1, Complex ***A10m1, Complex ***A1m10, int save_header)
{
  FILE *f;
  int i, j, k;
  double delta_n_local(1e5);
	
	
  f = fopen(filename, "w");
  
  printf("*** Saving amplitudes to %s ...", filename);
  fflush(stdout);

  if (save_header)
    {
      fprintf(f, "%g %g %g\n", DX, DY, DZ);
      fprintf(f, "%g\n", EPSILON);
    }
  
  for (i=0; i<NX; ++i)
    for (j=0; j<NY; ++j)
	  for (k=0; k<NZ; ++k)
		{
		  if (k == NZ/2)
		  {
			  delta_n_local =delta_n_Of_r(i, j, k, k011, k101, k110, k01m1, k10m1, k1m10,A011,A101, A110,A01m1,A10m1,A1m10);
		  }
		  else
		  {
			  delta_n_local = 1e4;  
		  }
		  fprintf(f, "%g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g\n", DX*i, DY*j, DZ*k,
				  real(A011[i][j][k]), imag(A011[i][j][k]),
				  real(A101[i][j][k]), imag(A101[i][j][k]),
				  real(A110[i][j][k]), imag(A110[i][j][k]), 
				  real(A01m1[i][j][k]), imag(A01m1[i][j][k]),
				  real(A10m1[i][j][k]), imag(A10m1[i][j][k]),
				  real(A1m10[i][j][k]), imag(A1m10[i][j][k]), delta_n_local);
		  fprintf(f, "\n");
		}
  
  fclose(f);
  
  printf(" done ***\n");
  fflush(stdout);
}
/*************************************************************************************/
float delta_n_Of_r(int i, int j, int k, const double inVec1[],const double inVec2[],const double inVec3[],const double inVec4[],const double inVec5[],const double inVec6[], Complex ***A011, Complex ***A101, Complex ***A110, 
				   Complex ***A01m1, Complex ***A10m1, Complex ***A1m10)
{
	float localRelativeDensityChange(1e5);	
	
	localRelativeDensityChange = real((1./6. ) * ( A011[i][j][k]*conj(A011[i][j][k])  +A101[i][j][k]*conj(A101[i][j][k])  +A110[i][j][k]*conj(A110[i][j][k]) +A01m1[i][j][k]*conj(A01m1[i][j][k]) + A10m1[i][j][k]*conj(A10m1[i][j][k]) +A1m10[i][j][k]*conj(A1m10[i][j][k])  ) );
	
	return localRelativeDensityChange;
}
/*************************************************************************************/
double psi_k(const double r[], const double k[], const double Ar, const double Ai)
{
  double kr;
  int i;
  
  kr = 0.0;
  for (i=0; i<3; ++i)
    kr += k[i]*r[i];
  
  return 2.0*Ar*cos(kr) - 2.0*Ai*sin(kr);
}

/*************************************************************************************/
void vtk_save(char *filename, Complex ***A011, Complex ***A101, Complex ***A110, 
			  Complex ***A01m1, Complex ***A10m1, Complex ***A1m10,
			  const int finegrid, const double xmin, const double xmax,
			  const double ymin, const double ymax, const double zmin, const double zmax)
{
  FILE *f;
  int i, j, k;
  double value, r[3];
  
  printf("*** Saving vtk image to %s ...", filename);
  fflush(stdout);

  f = fopen(filename, "w");
  
  fprintf(f, "# vtk DataFile Version 3.0\n");
  fprintf(f, "atoms\n");
  fprintf(f, "ASCII\n");
  fprintf(f, "DATASET STRUCTURED_POINTS\n");
  fprintf(f, "DIMENSIONS %d %d %d\n", int(NX*finegrid*(xmax-xmin)), int(NY*finegrid*(ymax-ymin)), int(NZ*finegrid*(zmax-zmin)));
  fprintf(f, "ASPECT_RATIO 1 1 1\n");
  fprintf(f, "ORIGIN 0 0 0\n");
  fprintf(f, "POINT_DATA %d\n", int(NX*finegrid*(xmax-xmin))*int(NY*finegrid*(ymax-ymin))*int(NZ*finegrid*(zmax-zmin)));
  fprintf(f, "SCALARS density double 1\n");
  fprintf(f, "LOOKUP_TABLE default\n");
  
  for (k=(int)(NZ*finegrid*zmin); k<(int)(NZ*finegrid*zmax); ++k)
	for (j=(int)(NY*finegrid*ymin); j<(int)(NY*finegrid*ymax); ++j)
	  for (i=(int)(NX*finegrid*xmin); i<(int)(NX*finegrid*xmax); ++i)
		{
		  r[0] = i*DX/sqrt(EPSILON)/finegrid;
		  r[1] = j*DY/sqrt(EPSILON)/finegrid;
		  r[2] = k*DZ/sqrt(EPSILON)/finegrid;
		  
		  value = psi_k(r, k011, real(A011[i/finegrid][j/finegrid][k/finegrid]), imag(A011[i/finegrid][j/finegrid][k/finegrid]))
		  + psi_k(r, k101, real(A101[i/finegrid][j/finegrid][k/finegrid]), imag(A101[i/finegrid][j/finegrid][k/finegrid]))
		  + psi_k(r, k110, real(A110[i/finegrid][j/finegrid][k/finegrid]), imag(A110[i/finegrid][j/finegrid][k/finegrid]))
		  + psi_k(r, k01m1, real(A01m1[i/finegrid][j/finegrid][k/finegrid]), imag(A01m1[i/finegrid][j/finegrid][k/finegrid]))
		  + psi_k(r, k10m1, real(A10m1[i/finegrid][j/finegrid][k/finegrid]), imag(A10m1[i/finegrid][j/finegrid][k/finegrid])) 
		  + psi_k(r, k1m10, real(A1m10[i/finegrid][j/finegrid][k/finegrid]), imag(A1m10[i/finegrid][j/finegrid][k/finegrid]));
		  
		  fprintf(f, "%g ", value);
		}
  
  fprintf(f, "\n");
  fclose(f);
  
  printf(" done ***\n");
  fflush(stdout);
}

/*************************************************************************************/
double gfunc(const double x)
{
  return x*x*(1-x)*(1-x);
}

/*************************************************************************************/
void vtk_save_interface(char *filename, Complex ***A011, Complex ***A101, Complex ***A110, 
						Complex ***A01m1, Complex ***A10m1, Complex ***A1m10,
						const double xmin, const double xmax,
						const double ymin, const double ymax, const double zmin, const double zmax)
{
  FILE *f;
  int i, j, k;
  double value;
  
  printf("*** Saving vtk interface image to %s ...", filename);
  fflush(stdout);
  
  f = fopen(filename, "w");
  
  fprintf(f, "# vtk DataFile Version 3.0\n");
  fprintf(f, "atoms\n");
  fprintf(f, "ASCII\n");
  fprintf(f, "DATASET STRUCTURED_POINTS\n");
  fprintf(f, "DIMENSIONS %d %d %d\n", int(NX*(xmax-xmin)), int(NY*(ymax-ymin)), int(NZ*(zmax-zmin)));
  fprintf(f, "ASPECT_RATIO 1 1 1\n");
  fprintf(f, "ORIGIN 0 0 0\n");
  fprintf(f, "POINT_DATA %d\n", int(NX*(xmax-xmin))*int(NY*(ymax-ymin))*int(NZ*(zmax-zmin)));
  fprintf(f, "SCALARS interface double density double 2\n");
  fprintf(f, "LOOKUP_TABLE default\n");
  
  for (k=(int)(NZ*zmin); k<(int)(NZ*zmax); ++k)
	for (j=(int)(NY*ymin); j<(int)(NY*ymax); ++j)
	  for (i=(int)(NX*xmin); i<(int)(NX*xmax); ++i)
		{
		  value = real(A011[i][j][k]*conj(A011[i][j][k]) + A101[i][j][k]*conj(A101[i][j][k])
					   + A110[i][j][k]*conj(A110[i][j][k]) + A01m1[i][j][k]*conj(A01m1[i][j][k])
					   + A10m1[i][j][k]*conj(A10m1[i][j][k]) + A1m10[i][j][k]*conj(A1m10[i][j][k]));
		  
		  fprintf(f, "%g %g", gfunc(value/6.0), value/6.0);
		}
  
  fprintf(f, "\n");
  fclose(f);
  
  printf(" done ***\n");
  fflush(stdout);
}

/*************************************************************************************/
void write_param(FILE *f, const struct param_t * const param)
{
  fwrite(param, sizeof(struct param_t), 1, f);
}

/*************************************************************************************/
void get_param(FILE *f, struct param_t *param)
{
  fread(param, sizeof(struct param_t), 1, f);
}

/*************************************************************************************/
void save_fields_exact_spectral(Complex ***A011, Complex ***A101, Complex ***A110,
                                Complex ***A01m1, Complex ***A10m1, Complex ***A1m10,
                                Complex ***A011inhomprevfft, Complex ***A101inhomprevfft,
                                Complex ***A110inhomprevfft,
                                Complex ***A01m1inhomprevfft, Complex ***A10m1inhomprevfft,
                                Complex ***A1m10inhomprevfft,
                                param_t *param, const char *filename)
{
  int fieldsize = NX * NY * NZ;
  FILE *f;
  
  printf("*** Saving binary data to %s ...", filename);
  fflush(stdout);
  
  f = fopen(filename, "w");
  
  write_param(f, param);
  
  fwrite(&(A011[0][0][0]), fieldsize, sizeof(Complex), f);
  fwrite(&(A101[0][0][0]), fieldsize, sizeof(Complex), f);
  fwrite(&(A110[0][0][0]), fieldsize, sizeof(Complex), f);
  fwrite(&(A01m1[0][0][0]), fieldsize, sizeof(Complex), f);
  fwrite(&(A10m1[0][0][0]), fieldsize, sizeof(Complex), f);
  fwrite(&(A1m10[0][0][0]), fieldsize, sizeof(Complex), f);
  fwrite(&(A011inhomprevfft[0][0][0]), fieldsize, sizeof(Complex), f);
  fwrite(&(A101inhomprevfft[0][0][0]), fieldsize, sizeof(Complex), f);
  fwrite(&(A110inhomprevfft[0][0][0]), fieldsize, sizeof(Complex), f);
  fwrite(&(A01m1inhomprevfft[0][0][0]), fieldsize, sizeof(Complex), f);
  fwrite(&(A10m1inhomprevfft[0][0][0]), fieldsize, sizeof(Complex), f);
  fwrite(&(A1m10inhomprevfft[0][0][0]), fieldsize, sizeof(Complex), f);
  
  fclose(f);
  
  printf(" done ***\n");
  fflush(stdout);
}

/*************************************************************************************/
void read_fields_exact_spectral(Complex ***A011, Complex ***A101, Complex ***A110,
                                Complex ***A01m1, Complex ***A10m1, Complex ***A1m10,
                                Complex ***A011inhomprevfft, Complex ***A101inhomprevfft,
                                Complex ***A110inhomprevfft,
                                Complex ***A01m1inhomprevfft, Complex ***A10m1inhomprevfft,
                                Complex ***A1m10inhomprevfft,
                                param_t *param, const char *filename)
{
  int fieldsize = NX * NY * NZ;
  FILE *f;
  
  f = fopen(filename, "r");
  
  get_param(f, param);
  
  fread(&(A011[0][0][0]), fieldsize, sizeof(Complex), f);  
  fread(&(A101[0][0][0]), fieldsize, sizeof(Complex), f);  
  fread(&(A110[0][0][0]), fieldsize, sizeof(Complex), f);  
  fread(&(A01m1[0][0][0]), fieldsize, sizeof(Complex), f);  
  fread(&(A10m1[0][0][0]), fieldsize, sizeof(Complex), f);  
  fread(&(A1m10[0][0][0]), fieldsize, sizeof(Complex), f);  
  fread(&(A011inhomprevfft[0][0][0]), fieldsize, sizeof(Complex), f);  
  fread(&(A101inhomprevfft[0][0][0]), fieldsize, sizeof(Complex), f);  
  fread(&(A110inhomprevfft[0][0][0]), fieldsize, sizeof(Complex), f);  
  fread(&(A01m1inhomprevfft[0][0][0]), fieldsize, sizeof(Complex), f);  
  fread(&(A10m1inhomprevfft[0][0][0]), fieldsize, sizeof(Complex), f);  
  fread(&(A1m10inhomprevfft[0][0][0]), fieldsize, sizeof(Complex), f);  
  
  fclose(f);
}

/*************************************************************************/
Complex SolidVolume(Complex ***A011, Complex ***A101, Complex ***A110,
					Complex ***A01m1, Complex ***A10m1, Complex ***A1m10)
{
  Complex vol = 0;
  int i, j, k;
  
  for (i=0; i<NX; ++i)
    for (j=0; j<NY; ++j)
	  for (k=0; k<NZ; ++k)
		vol += A011[i][j][k]*conj(A011[i][j][k]) + A101[i][j][k]*conj(A101[i][j][k])
		+ A110[i][j][k]*conj(A110[i][j][k]) + A01m1[i][j][k]*conj(A01m1[i][j][k])
		+ A10m1[i][j][k]*conj(A10m1[i][j][k]) + A1m10[i][j][k]*conj(A1m10[i][j][k]);
  
  return vol*DX*DY*DZ;
}

/*************************************************************************/
Complex IndicatorXY(Complex ***A011, Complex ***A101, Complex ***A110,
					Complex ***A01m1, Complex ***A10m1, Complex ***A1m10, int i, int j)
{
  int k;
  
  k = NZ/2;
  Complex ind=0;
  ind = A011[i][j][k]*conj(A011[i][j][k]) + A101[i][j][k]*conj(A101[i][j][k])
	+ A110[i][j][k]*conj(A110[i][j][k]) + A01m1[i][j][k]*conj(A01m1[i][j][k])
	+ A10m1[i][j][k]*conj(A10m1[i][j][k]) + A1m10[i][j][k]*conj(A1m10[i][j][k]);
  return ind;
}

/*************************************************************************/
Complex IndicatorXYminimum(Complex ***A011, Complex ***A101, Complex ***A110,
						   Complex ***A01m1, Complex ***A10m1, Complex ***A1m10,
						   const double radius)
{
  int i,j;
  double x,y, min, minX;
  i = 0;
  j = 0;
  
  min = IndicatorXY(A011, A101, A110, A01m1, A10m1, A1m10,67,67).real();
  
  for(i=0;i<NX;i++)
    for(j=0;j<NY;j++)
	  {
		if ((i-NX/2.0)*(i-NX/2.0)+(j-NY/2.0)*(j-NY/2.0) < radius*radius)
		  {
			if (min > IndicatorXY(A011, A101, A110, A01m1, A10m1, A1m10,i,j).real())
			  {
				min = IndicatorXY(A011, A101, A110, A01m1, A10m1, A1m10,i,j).real();
				minX = i;
			  }
		  }
	  }
  return minX;
}

/*************************************************************************/
Complex massX(Complex ***A011, Complex ***A101, Complex ***A110,
					Complex ***A01m1, Complex ***A10m1, Complex ***A1m10)
{
  int i,j,k;
  Complex factorX = 0;
  k = NZ/2;
  double NXhalf = NX/2;
    for (i=0; i<NX; ++i)
    for (j=0; j<NY; ++j)
	  factorX += ((6.0-(A011[i][j][k]*conj(A011[i][j][k]) + A101[i][j][k]*conj(A101[i][j][k])
		+ A110[i][j][k]*conj(A110[i][j][k]) + A01m1[i][j][k]*conj(A01m1[i][j][k])
		+ A10m1[i][j][k]*conj(A10m1[i][j][k]) + A1m10[i][j][k]*conj(A1m10[i][j][k])))*0.16667)*
		  ((6.0-(A011[i][j][k]*conj(A011[i][j][k]) + A101[i][j][k]*conj(A101[i][j][k])
		+ A110[i][j][k]*conj(A110[i][j][k]) + A01m1[i][j][k]*conj(A01m1[i][j][k])
		+ A10m1[i][j][k]*conj(A10m1[i][j][k]) + A1m10[i][j][k]*conj(A1m10[i][j][k])))*0.16667)*(i-NXhalf);
  return factorX*DX*DY*DZ;
}

/*************************************************************************/
Complex massY(Complex ***A011, Complex ***A101, Complex ***A110,
					Complex ***A01m1, Complex ***A10m1, Complex ***A1m10)
{
  int i,j,k;
  Complex factorY = 0;
  k = NZ/2;
  double NYhalf = NY/2;
    for (i=0; i<NX; ++i)
    for (j=0; j<NY; ++j)
	  factorY += ((6.0-(A011[i][j][k]*conj(A011[i][j][k]) + A101[i][j][k]*conj(A101[i][j][k])
		+ A110[i][j][k]*conj(A110[i][j][k]) + A01m1[i][j][k]*conj(A01m1[i][j][k])
		+ A10m1[i][j][k]*conj(A10m1[i][j][k]) + A1m10[i][j][k]*conj(A1m10[i][j][k])))*0.16667)*
		  ((6.0-(A011[i][j][k]*conj(A011[i][j][k]) + A101[i][j][k]*conj(A101[i][j][k])
		+ A110[i][j][k]*conj(A110[i][j][k]) + A01m1[i][j][k]*conj(A01m1[i][j][k])
		+ A10m1[i][j][k]*conj(A10m1[i][j][k]) + A1m10[i][j][k]*conj(A1m10[i][j][k])))*0.16667)*(j-NYhalf);
  return factorY*DX*DY*DZ;
}

/*************************************************************************/
Complex norm(Complex ***A011, Complex ***A101, Complex ***A110,
					Complex ***A01m1, Complex ***A10m1, Complex ***A1m10)
{
   int i,j,k;
   Complex norm1 = 0;
   k = NZ/2;
     for (i=0; i<NX; ++i)
     for (j=0; j<NY; ++j)
	  norm1 += ((6.0-(A011[i][j][k]*conj(A011[i][j][k]) + A101[i][j][k]*conj(A101[i][j][k])
		+ A110[i][j][k]*conj(A110[i][j][k]) + A01m1[i][j][k]*conj(A01m1[i][j][k])
		+ A10m1[i][j][k]*conj(A10m1[i][j][k]) + A1m10[i][j][k]*conj(A1m10[i][j][k])))*0.16667);
  return norm1*DX*DY*DZ;
}


/*************************************************************************/
void Init_temperature(Complex ***temperature)
{
  int i, j, k;
  double x,y,z,r;

  for (i=0; i<NX; ++i)
    for (j=0; j<NY; ++j)
	  for (k=0; k<NZ; ++k)
		{
		  x = i*DX;
		  y = j*DY;
		  z = k*DZ;
		  r = sqrt((x-X/2.0)*(x-X/2.0) + (y-Y/2.0)*(y-Y/2.0));
		  //temperature[i][j][k] = -TSOLID - (TLIQUID-TSOLID)*0.5*(1+tanh((r-TRADIUS)/TETA));
			temperature[i][j][k] = -0.4;
	  }
}

/*************************************************************************/
int main(int argc, char **argv)
{
  Complex ***A011, ***A101, ***A110, ***A01m1, ***A10m1, ***A1m10,
  ***A011fft, ***A101fft, ***A110fft, ***A01m1fft, ***A10m1fft, ***A1m10fft,
  ***A011inhom, ***A101inhom, ***A110inhom, ***A01m1inhom, ***A10m1inhom, ***A1m10inhom,
  ***A011inhomfft, ***A101inhomfft, ***A110inhomfft,
  ***A01m1inhomfft, ***A10m1inhomfft, ***A1m10inhomfft, 
  ***A011newfft, ***A101newfft, ***A110newfft, ***A01m1newfft, ***A10m1newfft, ***A1m10newfft,
  ***A011inhomprevfft, ***A101inhomprevfft, ***A110inhomprevfft,
  ***A01m1inhomprevfft, ***A10m1inhomprevfft, ***A1m10inhomprevfft, 
  ***FreeEnergyInhom, ***FreeEnergyInhomfft, ***temperature;
  
  fftw_plan p011, p101, p110, p01m1, p10m1, p1m10, p011inhom, p101inhom, p110inhom, 
  p01m1inhom, p10m1inhom, p1m10inhom, p011new, p101new, p110new, p01m1new, p10m1new, p1m10new;
  
  int abort, rank, size;
  FILE *f, *g, *h, *folderMassX, *folderMassY, *ix, *iy;
  double starttime, curtime;
  param_t param;
  char filename[255], filename2[255];
  int i, j, k,n;
  Complex solvol;
    
  /* Start MPI */
  MPI_Init(&argc, &argv);
  starttime = MPI_Wtime();
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
#ifdef MULTITHREADED
  fftw_init_threads();
  fftw_plan_with_nthreads(NTHREADS);
#endif
  
  printf("\n*** Parameters ***\n");
  printf("X = %g\n", X);
  printf("Y = %g\n", Y);
  printf("Z = %g\n", Z);
  printf("DX = %g\n", DX);
  printf("DY = %g\n", DY);
  printf("DZ = %g\n", DZ);
  printf("\n");
  
  A011   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A101   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A110   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A01m1   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A10m1   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A1m10   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  
  A011fft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A101fft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A110fft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A01m1fft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A10m1fft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A1m10fft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  
  A011inhom   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A101inhom   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A110inhom   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A01m1inhom   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A10m1inhom   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A1m10inhom   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  
  A011inhomfft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A101inhomfft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A110inhomfft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A01m1inhomfft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A10m1inhomfft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A1m10inhomfft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  
  A011inhomprevfft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A101inhomprevfft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A110inhomprevfft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A01m1inhomprevfft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A10m1inhomprevfft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A1m10inhomprevfft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  
  A011newfft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A101newfft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A110newfft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A01m1newfft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A10m1newfft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  A1m10newfft   = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  
  // for free energy calculation
  FreeEnergyInhom    = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  FreeEnergyInhomfft = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  
  temperature = fftw_ctensor(NX, NY, NZ);  // [0..NX-1][0..NY-1][0..NZ-1]
  
#if (USE_WISDOM == TRUE)
  f = fopen("wisdom.dat", "r");
  if (f == NULL)
    {
      printf("** No fftw wisdom file found **\n");
      fflush(stdout);
    }
  else
    {
      printf("** Restoring fftw wisdom ... ");
      fflush(stdout);
      fftw_import_wisdom_from_file(f);
      fclose(f);
      printf("done **\n");
      fflush(stdout);
    }
#endif
  
  /*
   I need the following plans:
   
   FORWARD:
   p011:      A011 -> A011fft
   p101:      A101 -> A101fft
   p110:      A110 -> A110fft
   p01m1:      A01m1 -> A01m1fft
   p10m1:      A10m1 -> A10m1fft
   p1m10:      A1m10 -> A1m10fft
   
   p011inhom: A011inhom -> A011inhomfft
   p101inhom: A101inhom -> A101inhomfft
   p110inhom: A110inhom -> A110inhomfft
   p01m1inhom: A01m1inhom -> A01m1inhomfft
   p10m1inhom: A10m1inhom -> A10m1inhomfft
   p1m10inhom: A1m10inhom -> A1m10inhomfft
   
   pFreeEnergyInhom: FreeEnergyInhom -> FreeEnergyInhomfft
   
   BACKWARD:
   p011new:   A011newfft -> A011
   p101new:   A101newfft -> A101
   p110new:   A110newfft -> A110
   p01m1new:   A01m1newfft -> A01m1
   p10m1new:   A10m1newfft -> A10m1
   p1m10new:   A1m10newfft -> A1m10
   */
  
  printf("** Setting up fftw plans ... ");
  fflush(stdout);
  /* 
   notice that we always have *A1 and so on, because we need a pointer to the data block,
   not to the vector of pointers.
   */
  p011 = fftw_plan_dft_3d(NX, NY, NZ, reinterpret_cast<fftw_complex*>(**A011), 
                          reinterpret_cast<fftw_complex*>(**A011fft), 
                          FFTW_FORWARD, FFTW_PATIENT);
  p101 = fftw_plan_dft_3d(NX, NY, NZ, reinterpret_cast<fftw_complex*>(**A101), 
                          reinterpret_cast<fftw_complex*>(**A101fft), 
                          FFTW_FORWARD, FFTW_PATIENT);
  p110 = fftw_plan_dft_3d(NX, NY, NZ, reinterpret_cast<fftw_complex*>(**A110), 
                          reinterpret_cast<fftw_complex*>(**A110fft), 
                          FFTW_FORWARD, FFTW_PATIENT);
  p01m1 = fftw_plan_dft_3d(NX, NY, NZ, reinterpret_cast<fftw_complex*>(**A01m1), 
                           reinterpret_cast<fftw_complex*>(**A01m1fft), 
                           FFTW_FORWARD, FFTW_PATIENT);
  p10m1 = fftw_plan_dft_3d(NX, NY, NZ, reinterpret_cast<fftw_complex*>(**A10m1), 
                           reinterpret_cast<fftw_complex*>(**A10m1fft), 
                           FFTW_FORWARD, FFTW_PATIENT);
  p1m10 = fftw_plan_dft_3d(NX, NY, NZ, reinterpret_cast<fftw_complex*>(**A1m10), 
                           reinterpret_cast<fftw_complex*>(**A1m10fft), 
                           FFTW_FORWARD, FFTW_PATIENT);
  
  p011inhom = fftw_plan_dft_3d(NX, NY, NZ, reinterpret_cast<fftw_complex*>(**A011inhom), 
                               reinterpret_cast<fftw_complex*>(**A011inhomfft), 
                               FFTW_FORWARD, FFTW_PATIENT);
  p101inhom = fftw_plan_dft_3d(NX, NY, NZ, reinterpret_cast<fftw_complex*>(**A101inhom), 
                               reinterpret_cast<fftw_complex*>(**A101inhomfft), 
                               FFTW_FORWARD, FFTW_PATIENT);
  p110inhom = fftw_plan_dft_3d(NX, NY, NZ, reinterpret_cast<fftw_complex*>(**A110inhom), 
                               reinterpret_cast<fftw_complex*>(**A110inhomfft), 
                               FFTW_FORWARD, FFTW_PATIENT);
  p01m1inhom = fftw_plan_dft_3d(NX, NY, NZ, reinterpret_cast<fftw_complex*>(**A01m1inhom), 
                                reinterpret_cast<fftw_complex*>(**A01m1inhomfft), 
                                FFTW_FORWARD, FFTW_PATIENT);
  p10m1inhom = fftw_plan_dft_3d(NX, NY, NZ, reinterpret_cast<fftw_complex*>(**A10m1inhom), 
                                reinterpret_cast<fftw_complex*>(**A10m1inhomfft), 
                                FFTW_FORWARD, FFTW_PATIENT);
  p1m10inhom = fftw_plan_dft_3d(NX, NY, NZ, reinterpret_cast<fftw_complex*>(**A1m10inhom), 
                                reinterpret_cast<fftw_complex*>(**A1m10inhomfft), 
                                FFTW_FORWARD, FFTW_PATIENT);
  
  p011new = fftw_plan_dft_3d(NX, NY, NZ, reinterpret_cast<fftw_complex*>(**A011newfft), 
                             reinterpret_cast<fftw_complex*>(**A011), 
                             FFTW_BACKWARD, FFTW_PATIENT);
  p101new = fftw_plan_dft_3d(NX, NY, NZ, reinterpret_cast<fftw_complex*>(**A101newfft), 
                             reinterpret_cast<fftw_complex*>(**A101), 
                             FFTW_BACKWARD, FFTW_PATIENT);
  p110new = fftw_plan_dft_3d(NX, NY, NZ, reinterpret_cast<fftw_complex*>(**A110newfft), 
                             reinterpret_cast<fftw_complex*>(**A110), 
                             FFTW_BACKWARD, FFTW_PATIENT);
  p01m1new = fftw_plan_dft_3d(NX, NY, NZ, reinterpret_cast<fftw_complex*>(**A01m1newfft), 
                              reinterpret_cast<fftw_complex*>(**A01m1), 
                              FFTW_BACKWARD, FFTW_PATIENT);
  p10m1new = fftw_plan_dft_3d(NX, NY, NZ, reinterpret_cast<fftw_complex*>(**A10m1newfft), 
                              reinterpret_cast<fftw_complex*>(**A10m1), 
                              FFTW_BACKWARD, FFTW_PATIENT);
  p1m10new = fftw_plan_dft_3d(NX, NY, NZ, reinterpret_cast<fftw_complex*>(**A1m10newfft), 
                              reinterpret_cast<fftw_complex*>(**A1m10), 
                              FFTW_BACKWARD, FFTW_PATIENT);
  
  printf("done **\n");
  fflush(stdout);
  
#if (USE_WISDOM == TRUE)
  printf("** Exporting fftw wisdom ... ");
  fflush(stdout);
  f = fopen("wisdom.dat", "w");
  fftw_export_wisdom_to_file(f);
  fclose(f);
  printf("done **\n");
  fflush(stdout);
#endif
  
  
  // continuation run or new run?
  f = fopen("fields_exact.dat", "r");
  
  if (f == NULL)
	{
	  printf("*** Starting new run ***\n");
	  
	  // inital conditions
	  // init_pure_rotatedxy(A011, A101, A110, A01m1, A10m1, A1m10, PHI);
	  //init_symmetric_tilt(A011, A101, A110, A01m1, A10m1, A1m10, PHI);
	   init_solid_liquid(A011, A101, A110, A01m1, A10m1, A1m10, PHI);
	  // init_wire(A011, A101, A110, A01m1, A10m1, A1m10, PHI, X/4.0);
	  // init_wire_screw(A011, A101, A110, A01m1, A10m1, A1m10, X/4.0, BURGERS, X*0.5, Y*0.5);
	  
	  param.n = 0;    // initialize loop counter
	}
  else
	{
	  fclose(f);
	  
	  printf("*** Continuing previous run ***\n");
	  
	  read_fields_exact_spectral(A011, A101, A110, A01m1, A10m1, A1m10,
								 A011inhomprevfft, A101inhomprevfft, A110inhomprevfft,
								 A01m1inhomprevfft, A10m1inhomprevfft, A1m10inhomprevfft,
								 &param, "fields_exact.dat");
	  
	}
  
  Init_temperature(temperature);
  
  g = fopen("SolidVolume.dat", "a");
  folderMassX = fopen("massX.dat", "w");
  folderMassY = fopen("massY.dat", "w");
  h = fopen("minimumX.dat", "w");
  
  abort = FALSE;
  
  // ****************** time loop *****************************
  do
	{  
	  timestep_spectral(A011, A101, A110, A01m1, A10m1, A1m10,
						A011fft, A101fft, A110fft, A01m1fft, A10m1fft, A1m10fft,
						A011inhom, A101inhom, A110inhom, A01m1inhom, A10m1inhom, A1m10inhom,
						A011inhomfft, A101inhomfft, A110inhomfft,
						A01m1inhomfft, A10m1inhomfft, A1m10inhomfft,
						A011inhomprevfft, A101inhomprevfft, A110inhomprevfft,
						A01m1inhomprevfft, A10m1inhomprevfft, A1m10inhomprevfft,
						A011newfft, A101newfft, A110newfft,
						A01m1newfft, A10m1newfft, A1m10newfft,
						temperature,
						&p011, &p101, &p110, &p01m1, &p10m1, &p1m10,
						&p011inhom, &p101inhom, &p110inhom, &p01m1inhom, &p10m1inhom, &p1m10inhom,
						&p011new, &p101new, &p110new, &p01m1new, &p10m1new, &p1m10new,
						param.n);
	  
	  
	  for (i=0; i<NX; ++i)
		for (j=0; j<NY; ++j)
		  for (k=0; k<NZ; ++k)
			{
			  A011inhomprevfft[i][j][k] = A011inhomfft[i][j][k];
			  A101inhomprevfft[i][j][k] = A101inhomfft[i][j][k];
			  A110inhomprevfft[i][j][k] = A110inhomfft[i][j][k];
			  A01m1inhomprevfft[i][j][k] = A01m1inhomfft[i][j][k];
			  A10m1inhomprevfft[i][j][k] = A10m1inhomfft[i][j][k];
			  A1m10inhomprevfft[i][j][k] = A1m10inhomfft[i][j][k];
			}
	  
	  if (param.n % SAVE == 0)
		{	  
		  solvol = SolidVolume(A011, A101, A110, A01m1, A10m1, A1m10);
		
		  ix = fopen("indicatorx.dat", "w");
		  for(n=0; n<NY;n++)
			{
			  fprintf(ix, "%g %g\n", n*DX, IndicatorXY(A011, A101, A110, A01m1, A10m1, A1m10,n,NY/2.0).real());
			}
		  fclose(ix);
		  
		  iy = fopen("indicatory.dat", "w");
		  for(n=0; n<NY;n++)
			{
			  fprintf(iy, "%g %g\n", n*DX, IndicatorXY(A011, A101, A110, A01m1, A10m1, A1m10,NX/2,n).real());
			}
		  fclose(iy);
		  
		  fprintf(g, "%g %g\n", param.n*DT, solvol.real());
		  fflush(g);
		  
		  printf("t = %g SolidVolume = %g\n", param.n*DT, solvol.real());
		  fflush(stdout);
		  
		  //***************************Calculate the minimum of the indicator function*********************************		  
		  fprintf(h, "%g %g\n", param.n*DT, IndicatorXYminimum(A011, A101, A110, A01m1, A10m1, A1m10, NX/4.0).real());
		  fflush(h);
		  
		  //************************************************************
		  fprintf(folderMassX, "%g %g\n", param.n*DT, massX(A011, A101, A110, A01m1, A10m1, A1m10).real()/norm(A011, A101, A110, A01m1, A10m1, A1m10).real());
		  fflush(folderMassX);
		  fprintf(folderMassY, "%g %g\n", param.n*DT, massY(A011, A101, A110, A01m1, A10m1, A1m10).real()/norm(A011, A101, A110, A01m1, A10m1, A1m10).real());
		  fflush(folderMassY);
		  
		}
	  
	  if (param.n % SAVE_VTK == 0)
		{	
		  sprintf(filename2, "vtkatoms.%d.vtk", param.n / SAVE_VTK);
		  vtk_save(filename2, A011, A101, A110, A01m1, A10m1, A1m10, FINEGRID, XMIN, XMAX,
				   YMIN, YMAX, ZMIN, ZMAX);
		  
		  sprintf(filename2, "vtkinterface.%d.vtk", param.n / SAVE_VTK);
		  vtk_save_interface(filename2, A011, A101, A110, A01m1, A10m1, A1m10, XMIN, XMAX,
							 YMIN, YMAX, ZMIN, ZMAX);
		}
	  
	  if (param.n % SAVE_IMAGE == 0)
		{
		  sprintf(filename, "atoms.%d.dat", param.n / SAVE_IMAGE);
		  save(filename, A011, A101, A110, A01m1, A10m1, A1m10, true);
		}
	  
	  curtime = MPI_Wtime();
	  if ((SOFTTIMELIMIT > 0) && (curtime-starttime > SOFTTIMELIMIT))
		{
		  printf("Softtimelimit exceeded!\n");
		  abort = TRUE;
		}
	  
	  if ((TIMESTEPS > 0) && (param.n > TIMESTEPS))
		{
		  printf("Maximum number of iterations reached!\n");
		  abort = TRUE;
		}
	  
	  
	  // increase timer
	  ++param.n; 
	}
  while (!abort);    // ****************** end of time loop *****************************
  
  fclose(g);
  fclose(h);
  fclose(folderMassX);
  fclose(folderMassY);
  
  sprintf(filename, "fields.dat");
  // save(filename, A011, A101, A110, A01m1, A10m1, A1m10, false);

  sprintf(filename, "atoms.dat");
  // save(filename, A011, A101, A110, A01m1, A10m1, A1m10, true);
  
  sprintf(filename, "fields_exact.dat");
  save_fields_exact_spectral(A011, A101, A110, A01m1, A10m1, A1m10,
							 A011inhomprevfft, A101inhomprevfft, A110inhomprevfft,
							 A01m1inhomprevfft, A10m1inhomprevfft, A1m10inhomprevfft,
							 &param, filename);
  
  /* generate vtk output */
  /*
  vtk_save("vtkatoms.vtk", A011, A101, A110, A01m1, A10m1, A1m10, FINEGRID, XMIN, XMAX,
		   YMIN, YMAX, ZMIN, ZMAX);
  */
  
  // free memory
  fftw_destroy_plan(p011);
  fftw_destroy_plan(p101);
  fftw_destroy_plan(p110);
  fftw_destroy_plan(p01m1);
  fftw_destroy_plan(p10m1);
  fftw_destroy_plan(p1m10);
  
  fftw_destroy_plan(p011inhom);
  fftw_destroy_plan(p101inhom);
  fftw_destroy_plan(p110inhom);
  fftw_destroy_plan(p01m1inhom);
  fftw_destroy_plan(p10m1inhom);
  fftw_destroy_plan(p1m10inhom);
  
  fftw_destroy_plan(p011new);
  fftw_destroy_plan(p101new);
  fftw_destroy_plan(p110new);
  fftw_destroy_plan(p01m1new);
  fftw_destroy_plan(p10m1new);
  fftw_destroy_plan(p1m10new);
  
  fftw_free_ctensor(A011);
  fftw_free_ctensor(A101);
  fftw_free_ctensor(A110);
  fftw_free_ctensor(A01m1);
  fftw_free_ctensor(A10m1);
  fftw_free_ctensor(A1m10);
  
  fftw_free_ctensor(A011fft);
  fftw_free_ctensor(A101fft);
  fftw_free_ctensor(A110fft);
  fftw_free_ctensor(A01m1fft);
  fftw_free_ctensor(A10m1fft);
  fftw_free_ctensor(A1m10fft);
  
  fftw_free_ctensor(A011inhom);
  fftw_free_ctensor(A101inhom);
  fftw_free_ctensor(A110inhom);
  fftw_free_ctensor(A01m1inhom);
  fftw_free_ctensor(A10m1inhom);
  fftw_free_ctensor(A1m10inhom);
  
  fftw_free_ctensor(A011inhomfft);
  fftw_free_ctensor(A101inhomfft);
  fftw_free_ctensor(A110inhomfft);
  fftw_free_ctensor(A01m1inhomfft);
  fftw_free_ctensor(A10m1inhomfft);
  fftw_free_ctensor(A1m10inhomfft);
  
  fftw_free_ctensor(A011inhomprevfft);
  fftw_free_ctensor(A101inhomprevfft);
  fftw_free_ctensor(A110inhomprevfft);
  fftw_free_ctensor(A01m1inhomprevfft);
  fftw_free_ctensor(A10m1inhomprevfft);
  fftw_free_ctensor(A1m10inhomprevfft);
  
  fftw_free_ctensor(A011newfft);
  fftw_free_ctensor(A101newfft);
  fftw_free_ctensor(A110newfft);
  fftw_free_ctensor(A01m1newfft);
  fftw_free_ctensor(A10m1newfft);
  fftw_free_ctensor(A1m10newfft);
  
  fftw_free_ctensor(FreeEnergyInhom);
  fftw_free_ctensor(FreeEnergyInhomfft);
  
  fftw_free_ctensor(temperature);
  
  printf("******** end of run ********\n");
  fflush(stdout);
  
  
  /* Terminate MPI */
  MPI_Finalize();
  
  return 0;
}

