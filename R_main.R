
#--------------------------------#
#         House-keeping          #
#--------------------------------#

library("parallel")
args<-commandArgs(TRUE)

#--------------------------------#
#         Initialization         #
#--------------------------------#

# Number of workers
no_cores <- as.integer(args)
cl <- makeCluster(no_cores)

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

# Initialize grids
xgrid = matrix(0, 1, nx)
egrid = matrix(0, 1, ne)
P     = matrix(0, ne, ne)
V     = array(0, dim=c(T, nx, ne))


#--------------------------------#
#         Grid creation          #
#--------------------------------#

# Grid for capital (x)
size = nx;
xstep = (xmax - xmin) /(size - 1);
it = 0;
for(i in 1:nx){
  xgrid[i] = xmin + it*xstep;
  it = it+1;
}

# Grid for productivity (e) with Tauchen (1986)
size = ne;
ssigma_y = sqrt((ssigma_eps^2) / (1 - (llambda_eps^2)));
estep = 2*ssigma_y*m / (size-1);
it = 0;
for(i in 1:ne){
  egrid[i] = (-m*sqrt((ssigma_eps^2) / (1 - (llambda_eps^2))) + it*estep);
  it = it+1;
}

# Transition probability matrix (P) Tauchen (1986)
mm = egrid[2] - egrid[1];
for(j in 1:ne){
  for(k in 1:ne){
    if(k == 1){
      P[j, k] = pnorm((egrid[k] - llambda_eps*egrid[j] + (mm/2))/ssigma_eps);
    } else if(k == ne){
      P[j, k] = 1 - pnorm((egrid[k] - llambda_eps*egrid[j] - (mm/2))/ssigma_eps);
    } else{
      P[j, k] = pnorm((egrid[k] - llambda_eps*egrid[j] + (mm/2))/ssigma_eps) - pnorm((egrid[k] - llambda_eps*egrid[j] - (mm/2))/ssigma_eps);
    }
  }
}

# Exponential of the grid e
for(i in 1:ne){
  egrid[i] = exp(egrid[i]);
}

#--------------------------------#
#        Value function          #
#--------------------------------#

# Function that computes value function, given vector of state variables
value = function(x){

  age    = x$age
  ind    = x$ind
  ne     = x$ne
  nx     = x$nx
  T      = x$T
  P      = x$P
  xgrid  = x$xgrid
  egrid  = x$egrid
  ssigma = x$ssigma
  bbeta  = x$bbeta
  V      = x$V
  w      = x$w
  r      = x$r

  ix = as.integer(floor((ind-0.05)/ne))+1;
  ie = as.integer(floor((ind-0.05) %% ne)+1);
  
  VV = -10.0^3;
  for(ixp in 1:nx){
    
    expected = 0.0;
    if(age < T){
      for(iep in 1:ne){
        expected = expected + P[ie, iep]*V[age+1, ixp, iep];
      }
    }
    
    cons  = (1 + r)*xgrid[ix] + egrid[ie]*w - xgrid[ixp];
    
    utility = (cons^(1-ssigma))/(1-ssigma) + bbeta*expected;
    
    if(cons <= 0){
      utility = -10.0^(5);
    }
    
    if(utility >= VV){
      VV = utility;
    }
  }
  
  return(VV);
}


#--------------------------------#
#     Life-cycle computation     #
#--------------------------------#

print(" ")
print("Life cycle computation: ")
print(" ")

start = proc.time()[3];

for(age in T:1){
  
  states = lapply(1:(ne*nx), function(x) list(age=age,ind=x,ne=ne,nx=nx,T=T,P=P,
                                            xgrid=xgrid,egrid=egrid,ssigma=ssigma,bbeta=bbeta,V=V,w=w,r=r))
  s = parLapply(cl, states, value)

  for(ind in 1:(nx*ne)){
    ix = as.integer(floor((ind-0.05)/ne))+1;
    ie = as.integer(floor((ind-0.05) %% ne)+1);
    
    V[age, ix, ie] = s[[ind]][1]
  }  
  
  finish = proc.time()[3] - start;
  print(paste0("Age: ", age, ". Time: ", round(finish, 3), " seconds."))
}

print(" ")
finish = proc.time()[3] - start;
print(paste("TOTAL ELAPSED TIME: ", finish, " seconds. "))


#---------------------#
#     Some checks     #
#---------------------#

print(" ")
print(" - - - - - - - - - - - - - - - - - - - - - ")
print(" ")
print("The first entries of the value function: ")
print(" ")

for(i in 1:3){
  print(V[1, 1, i])
}

print(" ")
