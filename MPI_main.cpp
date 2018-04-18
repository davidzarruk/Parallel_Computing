//
//  Created by David Zarruk Valencia on June, 2016.
//  Copyright (c) 2016 David Zarruk Valencia. All rights reserved.
//

#include <iostream>
#include <nlopt.hpp>
#include <math.h>
#include <mpi.h>
using namespace std;


//======================================
//         Grids
//======================================

// Function to construct the grid for capital (x)
void gridx(const int nx, const float xmin, const float xmax, float* xgrid){

  const float size = nx;
  const float xstep = (xmax - xmin) /(size - 1);
  float it = 0;

  for(int i = 0; i < nx; i++){
    xgrid[i] = xmin + it*xstep;
    it++;
  }
}


// Function to construct the grid for productivity (e) using Tauchen (1986)
void gride(const int ne, const float ssigma_eps, const float llambda_eps, const float m, float* egrid){

  // This grid is made with Tauchen (1986)
  const float size = ne;
  const float ssigma_y = sqrt(pow(ssigma_eps, 2) / (1 - pow(llambda_eps, 2)));
  const float estep = 2*ssigma_y*m / (size-1);
  float it = 0;

  for(int i = 0; i < ne; i++){
    egrid[i] = (-m*sqrt(pow(ssigma_eps, 2) / (1 - pow(llambda_eps, 2))) + it*estep);
    it++;
  }
}

// Function to compute CDF of Normal distribution
float normCDF(const float value){
  return 0.5 * erfc(-value * M_SQRT1_2);
}



// Function to construct the transition probability matrix for productivity (P) using Tauchen (1986)
void eprob(const int ne, const float ssigma_eps, const float llambda_eps, const float m, const float* egrid, float* P){

  // This grid is made with Tauchen (1986)
  // P is: first ne elements are transition from e_0 to e_i,
  //       second ne elementrs are from e_1 to e_i, ...
  const float w = egrid[1] - egrid[0];

  for(int j = 0; j < ne; j++){
    for(int k = 0; k < ne; k++){
      if(k == 0){
        P[j*ne + k] = normCDF((egrid[k] - llambda_eps*egrid[j] + (w/2))/ssigma_eps);
      } else if(k == ne-1){
        P[j*ne + k] = 1 - normCDF((egrid[k] - llambda_eps*egrid[j] - (w/2))/ssigma_eps);
      } else{
        P[j*ne + k] = normCDF((egrid[k] - llambda_eps*egrid[j] + (w/2))/ssigma_eps) - normCDF((egrid[k] - llambda_eps*egrid[j] - (w/2))/ssigma_eps);
      }
    }
  }
}


// Data structure of state and exogenous variables
struct modelState{
  int ie;
  int ix;
  int ne;
  int nx;
  int T;
  int age;
  float *P;
  float *xgrid;
  float *egrid;
  float ssigma;
  float bbeta;
  float *V;
  float w;
  float r;
};

// Function that computes value function, given vector of state variables
float value(modelState currentState){

  int ie         = currentState.ie;
  int ix         = currentState.ix;
  int ne         = currentState.ne;
  int nx         = currentState.nx;
  int T          = currentState.T;
  int age        = currentState.age;
  float *P       = currentState.P;
  float *xgrid   = currentState.xgrid;
  float *egrid   = currentState.egrid;
  float ssigma   = currentState.ssigma;
  float bbeta    = currentState.bbeta;
  float *V       = currentState.V;
  float w        = currentState.w;
  float r        = currentState.r;

  float expected;
  float utility;
  float cons;
  float VV = pow(-10.0,5.0);

  for(int ixp = 0; ixp < nx; ixp++){

    expected = 0.0;
    if(age < T-1){
      for(int iep = 0; iep < ne; iep++){
        expected = expected + P[ie*ne + iep]*V[(age+1)*nx*ne + ixp*ne + iep];
      }
    }

    cons  = (1 + r)*xgrid[ix] + egrid[ie]*w - xgrid[ixp];

    utility = pow(cons, 1-ssigma) / (1-ssigma) + bbeta*expected;

    if(cons <= 0){
      utility = pow(-10.0, 5.0);
    }

    if(utility >= VV){
      VV = utility;
    }
  }

  return VV;
}


//======================================
//         MAIN  MAIN  MAIN
//======================================


int main()
{ 

  //--------------------------------//
  //               MPI              //
  //--------------------------------//

  // Thread id and number of threads
  int tid,nthreads;

  // Initialize MPI
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &tid);
  MPI_Comm_size(MPI_COMM_WORLD, &nthreads);


  //--------------------------------//
  //         Initialization         //
  //--------------------------------//

  // Grid for x
  const int nx              = 1500; 
  const float xmin          = 0.1; 
  const float xmax          = 4.0;

  // Grid for e
  const int ne              = 15; 
  const float ssigma_eps    = 0.02058; 
  const float llambda_eps   = 0.99; 
  const float m             = 1.5; 

  // Utility function
  const float ssigma        = 2; 
  const float bbeta         = 0.97;
  const int T               = 10;

  // Prices
  const float r             = 0.07;
  const float w             = 5;

  // Initialize the grid for X
  float xgrid[nx];

  // Initialize the grid for E and the probability matrix
  float egrid[ne];  
  float P[ne*ne];

  // Total Value function
  size_t sizeVal     = T*ne*nx*sizeof(float);
  float *Val;
  Val     = (float *)malloc(sizeVal);

  // Value function at every age t+1
  size_t sizeVal1     = ne*nx*sizeof(float);
  float *Val1;
  Val1     = (float *)malloc(sizeVal1);


  // Loop limits of parallelization  
  int loop_min    = (int)((tid + 0)  *  ceil((float)(nx*ne)/nthreads));
  int loop_max    = (int)((tid + 1)  *  ceil((float)(nx*ne)/nthreads));
  if(ne*nx < loop_max){
    loop_max = ne*nx;
  }
  int leng = (loop_max - loop_min);

  // Value function at every age t, computed by every processor
  size_t sizeValp     = leng*sizeof(float);
  float *Valp;
  Valp     = (float *)malloc(sizeValp);

  
  // Initialize iterators
  int ix;
  int ie;
  int iter;


  // Parameters for the gathering (MPI_Gatherv) of value function after each iteration  
  int displs[nthreads],rcounts[nthreads];
  for(int i = 0; i < nthreads; i++){
    displs[i] = i*leng;
    rcounts[i] = leng;
  }


  //--------------------------------//
  //         Grid creation          //
  //--------------------------------//

  gridx(nx, xmin, xmax, xgrid);
  gride(ne, ssigma_eps, llambda_eps, m, egrid);
  eprob(ne, ssigma_eps, llambda_eps, m, egrid, P);

    // Exponential of the grid e
  for(int i=0; i<ne; i++){
    egrid[i] = exp(egrid[i]);
  }


  //--------------------------------//
  //    Life-cycle computation      //
  //--------------------------------//
  
  if(tid == 0){
    cout << " " << endl;
    cout << "Life cycle computation: " << endl;
    cout << " " << endl;
  }

  // Variables for computation time
  double t0  = MPI_Wtime();
  double t   = t0;


  // Compute value function bakwards
  for(int age=T-1; age>=0; age--){

    // Synchronize
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(Val, (T*ne*nx), MPI_FLOAT, 0, MPI_COMM_WORLD);

    iter = 0;

    for(int ind = loop_min; ind < loop_max; ind++){

      ix = floor(ind/ne);
      ie = ind % ne;

      modelState currentState = {ie, ix, ne, nx, T, age, P, xgrid, egrid, ssigma, bbeta, Val, w, r};

      Valp[iter] = value(currentState);
      iter = iter + 1;
    }

    MPI_Gatherv(Valp, leng, MPI_FLOAT, Val1, rcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(tid == 0){
      for(int ind = 0; ind < (nx*ne); ind++){

        ix = floor(ind/ne);
        ie = ind % ne;

        Val[age*nx*ne + ix*ne + ie] = Val1[ix*ne + ie];
      }

      t = MPI_Wtime() - t0;
      cout << "Age: " << age << ". Time: " << 1000000*((float)t)/CLOCKS_PER_SEC << " seconds." << endl;
    }
  }

  MPI_Finalize();
  
  if(tid == 0){
    std::cout << " " << std::endl;
    t = MPI_Wtime() - t0;
    std::cout << "TOTAL ELAPSED TIME: " << 1000000*((float)t)/CLOCKS_PER_SEC << " seconds. " << std::endl;
  }


  // //--------------------------------//
  // //           Some checks          //
  // //--------------------------------//

  if(tid == 0){
    std::cout << " " << std::endl;
    std::cout << " - - - - - - - - - - - - - - - - - - - - - " << std::endl;
    std::cout << " " << std::endl;
    std::cout << "The first entries of the value function: " << std::endl;
    std::cout << " " << std::endl;
  
    for(int i = 0; i<3; i++){
      std::cout << Val[i] << std::endl;
    }
  }

  return 0;
}
