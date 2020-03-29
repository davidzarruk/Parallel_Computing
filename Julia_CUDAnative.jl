using Distributions
using Dates
using CUDAnative, CuArrays, CUDAdrv
using Crayons

println(Crayon(foreground = :yellow), "\nYour CUDA-device: " * CUDAdrv.name(CuDevice(0)), Crayon(foreground = :white))

struct params
    ne::Int64
    nx::Int64
    T::Int64
    ssigma::Float64
    bbeta::Float64
    w::Float64
    r::Float64
end

# Function that computes value function, given vector of state variables
function value(params::params, age::Int64, xgrid, egrid, P, V)
    
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    ie = threadIdx().y
    
    ne      = params.ne
    nx      = params.nx
    T       = params.T
    ssigma  = params.ssigma
    bbeta   = params.bbeta
    w       = params.w
    r       = params.r

    VV      = -10.0^3;
    ixpopt  = 0;

    # @inbounds disables bounds checking
    @inbounds for ixp = 1:nx
        expected = 0.0;
        if(age < T)
            for iep = 1:ne
                expected = expected + P[ie, iep]*V[age + 1, ixp, iep];
            end
        end

        cons  = (1 + r)*xgrid[ix] + egrid[ie]*w - xgrid[ixp];

        # The ^ operator seems to broken on the GPU. One can always use the CUDAnative internals instead.
        utility = (CUDAnative.pow.(cons,1-ssigma))/(1-ssigma) + bbeta*expected;

        if(cons <= 0)
            utility = -10.0^(5);
        end

        if(utility >= VV)
            VV = utility;
            ixpopt = ixp;
        end

        utility = 0.0;
    end

    V[age, ix, ie] = VV

    return nothing
end

function main()    

    # Grid for x
    nx = 1500;
    xmin = 0.1;
    xmax = 4.0;

    # Grid for e: parameters for Tauchen
    ne = 15;
    ssigma_eps = 0.02058;
    llambda_eps = 0.99;
    m = 1.5;

    # Utility function
    ssigma = 2;
    bbeta = 0.97;
    T = 10;

    # Prices
    r = 0.07;
    w = 5;

    # The following arrays are CUDA-arrays. 

    # Initialize the grid for X
    xgrid = CuArray{Float64,1}(zeros(nx))

    # Initialize the grid for E and the transition probability matrix
    egrid = CuArray{Float64,1}(zeros(ne))
    P = CuArray{Float64,2}(zeros(ne, ne))

    # Initialize value function V
    V = CuArray{Float64,3}(zeros(T, nx, ne))

    #--------------------------------#
    #         Grid creation          #
    #--------------------------------#

    # The following operations on the CUDA-arrays trigger the warning "Performing scalar operations on GPU arrays: This is very slow..."
    # One could as well create regular arrays first and transfer them to the GPU later. However, this doesn't change the runtime significantly.

    # Grid for capital (x)
    size = nx;
    xstep = (xmax - xmin) /(size - 1);
    for i = 1:nx
    xgrid[i] = xmin + (i-1)*xstep;
    end

    # Grid for productivity (e) with Tauchen (1986)
    size = ne;
    ssigma_y = sqrt((ssigma_eps^2) / (1 - (llambda_eps^2)));
    estep = 2*ssigma_y*m / (size-1);
    for i = 1:ne
    egrid[i] = (-m*sqrt((ssigma_eps^2) / (1 - (llambda_eps^2))) + (i-1)*estep);
    end

    # Transition probability matrix (P) Tauchen (1986)
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
    currentState = params(ne,nx,T,ssigma,bbeta,w,r)
    @inbounds for age = T:-1:1
        # I need to sync for correct time measurement. Caution: Just @sync refers to Base.@sync
        CuArrays.@sync @cuda blocks=50 threads=(30, 15) value(currentState, age, xgrid, egrid, P, V)
        finish = convert(Int, Dates.value(Dates.unix2datetime(time())- start))/1000;
        # One should keep in mind, that printing in some environments is pretty slow in Julia. 
        # Without printing, on my machine Julia runs roughly as fast as C++-CUDA (ca. 1-2% slower)
        print("Age: ", age, ". Time: ", finish, " seconds. \n")
    end
    # transfer V back to the CPU to get a fair comparison with C++-CUDA (one can print the results without transfering them back manually, however I don't know what happens under the hood)
    results = Array(V)
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
        print(round(results[1, 1, i], digits=5), "\n")
    end
end

println(Crayon(foreground = :red), "\nWarmup call -- slow!", Crayon(foreground = :white))
main()
println(Crayon(foreground = :green), "\nProper call -- correct time measurement.", Crayon(foreground = :white))
main()