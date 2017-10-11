#--------------------------------#
#         House-keeping          #
#--------------------------------#

workspace()
using Distributions

#--------------------------------#
#         Initialization         #
#--------------------------------#

# Number of workers
# addprocs(8)

# Grid for x
@everywhere nx  = 300;
xmin            = 0.1;
xmax            = 4.0;

# Grid for e: parameters for Tauchen
@everywhere ne  = 15;
ssigma_eps      = 0.02058;
llambda_eps     = 0.99;
m               = 1.5;

# Utility function
@everywhere ssigma   = 2;
@everywhere eeta     = 0.36;
@everywhere ppsi     = 0.89;
@everywhere rrho     = 0.5;
@everywhere llambda  = 1;
@everywhere bbeta    = 0.97;
@everywhere T        = 10;

# Prices
@everywhere r  = 0.07;
@everywhere w  = 5;

# Initialize grids
@everywhere xgrid = zeros(nx)
@everywhere egrid = zeros(ne)
@everywhere P     = zeros(ne, ne)
@everywhere V     = zeros(T, nx, ne)

# Initialize value function as a shared array
tempV = SharedArray{Float64}(ne*nx, init = tempV -> tempV[Base.localindexes(tempV)] = myid())

#--------------------------------#
#         Grid creation          #
#--------------------------------#

# Grid for x
size = nx;
xstep = (xmax - xmin) /(size - 1);
it = 0;
for i = 1:nx
  xgrid[i] = xmin + it*xstep;
  it = it+1;
end

# Grid for e with Tauchen (1986)
size = ne;
ssigma_y = sqrt((ssigma_eps^2) / (1 - (llambda_eps^2)));
estep = 2*ssigma_y*m / (size-1);
it = 0;
for i = 1:ne
  egrid[i] = (-m*sqrt((ssigma_eps^2) / (1 - (llambda_eps^2))) + it*estep);
  it = it+1;
end

# Transition probability matrix Tauchen (1986)
mm = egrid[2] - egrid[1];
for j = 1:ne
  for k = 1:ne
    if(k == 1)
      P[j, k] = cdf(Normal(), (egrid[k] - llambda_eps*egrid[j] + (mm/2))/ssigma_eps);
    elseif(k == ne)
      P[j, k] = 1 - cdf(Normal(), (egrid[k] - llambda_eps*egrid[j] - (mm/2))/ssigma_eps);
    else
      P[j, k] = cdf(Normal(), (egrid[k] - llambda_eps*egrid[j] + (mm/2))/ssigma_eps) - cdf(Normal(), (egrid[k] - llambda_eps*egrid[j] - (mm/2))/ssigma_eps);
    end
  end
end

# Exponential of the grid e
for i = 1:ne
  egrid[i] = exp(egrid[i]);
end

#--------------------------------#
#     Life-cycle computation     #
#--------------------------------#

print(" \n")
print("Life cycle computation: \n")
print(" \n")

start = Dates.unix2datetime(time())

for age = T:-1:1

  @sync @parallel for ind = 1:(ne*nx)

    ix      = convert(Int, ceil(ind/ne));
    ie      = convert(Int, floor(mod(ind-0.05, ne))+1);

    VV = -10^3;

    for ixp = 1:nx

      expected = 0.0;
      if(age < T)
        for iep = 1:ne
          expected = expected + P[ie, iep]*V[age+1, ixp, iep];
        end
      end

      cons  = (1 + r)*xgrid[ix] + egrid[ie]*w - xgrid[ixp];

      utility = (cons^(1-ssigma))/(1-ssigma) + bbeta*expected;

      if(cons <= 0)
        utility = -10^(5);
      end

      if(utility >= VV)
        VV = utility;
      end

    end

    tempV[ind] = VV;
  end

  for ind = 1:(ne*nx)

    ix      = convert(Int, ceil(ind/ne));
    ie      = convert(Int, floor(mod(ind-0.05, ne))+1);

    V[age, ix, ie] = tempV[ind]
  end

  finish = convert(Int, Dates.value(Dates.unix2datetime(time())- start))/1000;
  print("Age: ", age, ". Time: ", finish, " seconds. \n")
end

print("\n")
finish = convert(Int, Dates.value(Dates.unix2datetime(time())- start))/1000;
print("TOTAL ELAPSED TIME: ", finish, " seconds. \n")

#---------------------#
#     Some checks     #
#---------------------#

print(" \n")
print(" - - - - - - - - - - - - - - - - - - - - - \n")
print(" \n")
print("The first entries of the value function: \n")
print(" \n")

# I print the first entries of the value function, to check
for i = 1:3
  print(round(V[1, 1, i], 5), "\n")
end
