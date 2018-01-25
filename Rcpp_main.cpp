
#include <iostream>
#include <nlopt.hpp>
#include <omp.h>
using namespace std;


//======================================
//         Grids
//======================================

void gridx(const int nx, const float xmin, const float xmax, float* xgrid){
  
  const float size = nx;
  const float xstep = (xmax - xmin) /(size - 1);
  float it = 0;
  
  for(int i = 0; i < nx; i++){
    xgrid[i] = xmin + it*xstep;
    it++;
  }
}

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

float normCDF(const float value){
  return 0.5 * erfc(-value * M_SQRT1_2);
}

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

//======================================
//         MAIN  MAIN  MAIN
//======================================

// [[Rcpp::export]]
vector<double> value(int nx, float xmin, float xmax, 
                     int ne, float ssigma_eps, float llambda_eps, float m, 
                     float ssigma, float eeta, float ppsi, float rrho, 
                     float llambda, float bbeta, int T, float r, float w){ 

  // I create the grid for X
  float xgrid[nx];

  // I create the grid for E and the probability matrix
  float egrid[ne];  
  float P[ne*ne];

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

  vector<double> V;
  V.resize(T*nx*ne);

  //--------------------------------//
  //    Life-cycle computation      //
  //--------------------------------//

  float expected;
  float utility;
  float cons;
  float VV = pow(-10.0,5.0);
  
  cout << " " << endl;
  cout << "Life cycle computation: " << endl;
  cout << " " << endl;

  double t0  = omp_get_wtime();
  double t   = t0;

  for(int age=T-1; age>=0; age--){

    #pragma omp parallel for shared(V, age, P, xgrid, egrid, t, t0) private(expected, cons, utility, VV)

    for(int ix = 0; ix<nx; ix++){
      for(int ie = 0; ie<ne; ie++){
	    VV = pow(-10, 5);
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

        V[age*nx*ne + ix*ne + ie] = VV;
      }
    }

    t = omp_get_wtime() - t0;
    cout << "Age: " << age << ". Time: " << 1000000*((float)t)/CLOCKS_PER_SEC << " seconds." << endl;
  }
  
  cout << " " << endl;
  t = omp_get_wtime() - t0;
  cout << "TOTAL ELAPSED TIME: " << 1000000*((float)t)/CLOCKS_PER_SEC << " seconds. " << endl;

  return V;
}
