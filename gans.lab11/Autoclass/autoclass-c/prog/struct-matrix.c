#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "autoclass.h"
#include "globals.h"

/* SUPRESS CODECENTER WARNING MESSSAGES */

/* empty body for 'while' statement */
/*SUPPRESS 570*/
/* formal parameter '<---->' was not used */
/*SUPPRESS 761*/
/* automatic variable '<---->' was not used */
/*SUPPRESS 762*/
/* automatic variable '<---->' was set but not used */
/*SUPPRESS 765*/


fptr *compute_factor( fptr *a, int n)

/* does LU factorization of nxn matrix a in place.*/
{

   int i,j,k;

   for (k=0;k<n;k++)
      for(i=k+1;i<n;i++)
      {
         a[i][k] /= a[k][k];
         for(j=k+1;j<n;j++)
            a[i][j] -= a[i][k] * a[k][j];
       }
   return(a);
}


/* This takes the factorization of some A generated by factor and a
   column-vector b, and replaces b by the solution x to Ax=b.  It returns b. */

float *solve( fptr *a, float *x,int n)
{
   int i,j;


   for(i=0;i<n;i++)
      for (j=0;j<i;j++)
         x[i] -= a[i][j] * x[j];

   for(i=n-1;i>=0;i--)
   {
     for(j=i+1;j<n;j++)
        x[i] -= a[i][j] * x[j];
     x[i] /= a[i][i];
   }
   return(x);
}
