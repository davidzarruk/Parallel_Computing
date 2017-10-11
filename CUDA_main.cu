// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppEigen)]]

#include <iostream>
#include <omp.h>
using namespace std;

//======================================
//         Grids
//======================================

void gridx(const int nx, const double xmin, const double xmax, double* xgrid){

  const double size = nx;
  const double xstep = (xmax - xmin) /(size - 1);
  double it = 0;

  for(int i = 0; i < nx; i++){
    xgrid[i] = xmin + it*xstep;
    it++;
  }
}


void gride(const int ne, const double ssigma_eps, const double llambda_eps, const double m, double* egrid){

  // This grid is made with Tauchen (1986)
  const double size = ne;
  const double ssigma_y = sqrt(pow(ssigma_eps, 2) / (1 - pow(llambda_eps, 2)));
  const double estep = 2*ssigma_y*m / (size-1);
  double it = 0;

  for(int i = 0; i < ne; i++){
    egrid[i] = (-m*sqrt(pow(ssigma_eps, 2) / (1 - pow(llambda_eps, 2))) + it*estep);
    it++;
  }
}

double normCDF(const double value){
  return 0.5 * erfc(-value * M_SQRT1_2);
}



void eprob(const int ne, const double ssigma_eps, const double llambda_eps, const double m, const double* egrid, double* P){

  // This grid is made with Tauchen (1986)
  // P is: first ne elements are transition from e_0 to e_i,
  //       second ne elementrs are from e_1 to e_i, ...
  const double w = egrid[1] - egrid[0];

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
//         Parameter structure
//======================================

class parameters{
 public:
  int nx;
  double xmin;
  double xmax;
  int ne;
  double ssigma_eps;
  double llambda_eps;
  double m;

  double ssigma;
  double eeta;
  double ppsi;
  double rrho;
  double llambda;
  double bbeta;
  int T;
  double r;
  double w;

  void load(const char*);
};



//======================================
//         MAIN  MAIN  MAIN
//======================================

__global__ void Vmaximization(const parameters params, const double* xgrid, const double* egrid, const double* P, const int age, double* V){

  // Recover the parameters
  const int nx              = params.nx;
  const int ne              = params.ne;
  const double ssigma        = params.ssigma;
  const double bbeta         = params.bbeta;
  const int T               = params.T;
  const double r             = params.r;
  const double w             = params.w;

  // Recover state variables from indices
  const int ix  = blockIdx.x * blockDim.x + threadIdx.x;
  const int ie  = threadIdx.y;

  double expected;
  double utility;
  double cons;
  double VV = pow(-10.0,5.0);

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

    utility = 0.0;
  }

  V[age*nx*ne + ix*ne + ie] = VV;
}



int main()
{
  // Grids
  const int nx              = 300; 
  const double xmin          = 0.1;
  const double xmax          = 4.0;
  const int ne              = 15;
  const double ssigma_eps    = 0.02058;
  const double llambda_eps   = 0.99;
  const double m             = 1.5;

  // Parameters
  const double ssigma        = 2;
  const double eeta          = 0.36;
  const double ppsi          = 0.89;
  const double rrho          = 0.5;
  const double llambda       = 1;
  const double bbeta         = 0.97;
  const int T             	= 10;

  // Prices
  const double r             = 0.07;
  const double w             = 5;

  parameters params = {nx, xmin, xmax, ne, ssigma_eps, llambda_eps, m, ssigma, eeta, ppsi, rrho, llambda, bbeta, T, r, w};

  // Pointers to variables in the DEVICE memory
  double *V, *X, *E, *P;
  size_t sizeX = nx*sizeof(double);
  size_t sizeE = ne*sizeof(double);
  size_t sizeP = ne*ne*sizeof(double);
  size_t sizeV = T*ne*nx*sizeof(double);

  cudaMalloc((void**)&X, sizeX);
  cudaMalloc((void**)&E, sizeE);
  cudaMalloc((void**)&P, sizeP);
  cudaMalloc((void**)&V, sizeV);

  // Parameters for CUDA: cada block tiene ne columnas, y una fila que representa un valor de x
  //                      Hay nx blocks
  //                      Cada layer es una edad >= hay 80 layers

  const int block_size = 30;
  dim3 dimBlock(block_size, ne);
  dim3 dimGrid(nx/block_size, 1);


  // Variables in the host have "h" prefix
  // I create the grid for X
  double hxgrid[nx];
  gridx(nx, xmin, xmax, hxgrid);

  // I create the grid for E and the probability matrix
  double hegrid[ne];
  double hP[ne*ne];
  gride(ne, ssigma_eps, llambda_eps, m, hegrid);
  eprob(ne, ssigma_eps, llambda_eps, m, hegrid, hP);

    // Exponential of the grid e
  for(int i=0; i<ne; i++){
    hegrid[i] = exp(hegrid[i]);
  }

  double *hV;
  hV = (double *)malloc(sizeV);

  // Copy matrices from host (CPU) to device (GPU) memory
  cudaMemcpy(X, hxgrid, sizeX, cudaMemcpyHostToDevice);
  cudaMemcpy(E, hegrid, sizeE, cudaMemcpyHostToDevice);
  cudaMemcpy(P, hP, sizeP, cudaMemcpyHostToDevice);
  cudaMemcpy(V, hV, sizeV, cudaMemcpyHostToDevice);

  std::cout << " " << std::endl;
  std::cout << "Life cycle computation: " << std::endl;
  std::cout << " " << std::endl;

  // Time the GPU startup overhead
  clock_t t;
  clock_t t0;
  t0 	= clock();
  t 	= t0;

  for(int age=T-1; age>=0; age--){
    Vmaximization<<<dimGrid,dimBlock>>>(params, X, E, P, age, V);
    cudaDeviceSynchronize();

  	t = clock() - t0;
  	std::cout << "Age: " << age << ". Time: " << ((double)t)/CLOCKS_PER_SEC << " seconds." << std::endl;

  }

  std::cout << " " << std::endl;
  t = clock() - t0;
  std::cout << "TOTAL ELAPSED TIME: " << ((double)t)/CLOCKS_PER_SEC << " seconds. " << std::endl;

  cudaMemcpy(hV, V, sizeV, cudaMemcpyDeviceToHost);

  // Free variables in device memory
  cudaFree(V);
  cudaFree(X);
  cudaFree(E);
  cudaFree(P);

  std::cout << " " << std::endl;
  std::cout << " - - - - - - - - - - - - - - - - - - - - - " << std::endl;
  std::cout << " " << std::endl;
  std::cout << "The first entries of the value function: " << std::endl;
  std::cout << " " << std::endl;

  for(int i = 0; i<3; i++){
    std::cout << hV[i] << std::endl;
  }

  std::cout << " " << std::endl;

  return 0;
}
