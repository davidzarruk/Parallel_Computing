
#--------------------------------#
#         House-keeping          #
#--------------------------------#

.libPaths( c( .libPaths(), "~/R/x86_64-pc-linux-gnu-library/3.4/") )

library("Rcpp")

setwd("/home/david/Dropbox/Documents/Doctorado/Computation course/Codigos/Github Repository/Parallel_Computing_2/")

Sys.setenv("PKG_CXXFLAGS"=" -fopenmp")

# Number of workers
sourceCpp("Rcpp_main.cpp")


#--------------------------------#
#         Initialization         #
#--------------------------------#

# Grid for x
nx            = 1500; 
xmin          = 0.1; 
xmax          = 4.0; 

# Grid for e: parameters for Tauchen
ne            = 15; 
ssigma_eps    = 0.02058; 
llambda_eps   = 0.99; 
m             = 1.5; 

# Utility function
ssigma        = 2; 
bbeta         = 0.97;
T             = 10;

# Prices
r             = 0.07;
w             = 5;


#--------------------------------#
#        Value function          #
#--------------------------------#

V = value(nx, xmin, xmax, 
      ne, ssigma_eps, llambda_eps, m, 
      ssigma, bbeta, T, r, w);


# I recover the Policy Functions
Value   = array(0,dim=c(T, nx, ne));

for (age in 1:T){
  for (ix in 1:nx){
    for(ie in 1:ne){
      Value[age, ix, ie]   = V[(age-1)*nx*ne + (ix-1)*ne + ie];
    }
  }
}

#---------------------#
#     Some checks     #
#---------------------#

print(" ")
print(" - - - - - - - - - - - - - - - - - - - - - ")
print(" ")
print("The first entries of the value function: ")
print(" ")

for(i in 1:3){
  print(Value[1, 1, i])
}

print(" ")
