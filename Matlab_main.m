function []=Matlab_main(cores)
%--------------------------------%
%         House-keeping          %
%--------------------------------%

%clc

%--------------------------------%
%         Initialization         %
%--------------------------------%

% Number of workers
parpool(str2num(cores));

% Grid for x
nx            = 300; 
xmin          = 0.1; 
xmax          = 4.0;

% Grid for e: parameters for Tauchen
ne            = 15; 
ssigma_eps    = 0.02058; 
llambda_eps   = 0.99; 
m             = 1.5; 

% Utility function
ssigma        = 2; 
eeta          = 0.36; 
ppsi          = 0.89; 
rrho          = 0.5; 
llambda       = 1; 
bbeta         = 0.97;
T             = 10;

% Prices
r             = 0.07;
w             = 5;

% Initialize grid
xgrid = zeros(1, nx);
egrid = zeros(1, ne);
P     = zeros(ne, ne);
V     = zeros(T, nx, ne);

%--------------------------------%
%         Grid creation          %
%--------------------------------%

% Grid for x
size = nx;
xstep = (xmax - xmin) /(size - 1);
it = 0;
for i = 1:nx
  xgrid(i) = xmin + it*xstep;
  it = it+1;
end

% Grid for e with Tauchen (1986)
size = ne;
ssigma_y = sqrt((ssigma_eps^2) / (1 - (llambda_eps^2)));
estep = 2*ssigma_y*m / (size-1);
it = 0;
for i = 1:ne
  egrid(i) = (-m*sqrt((ssigma_eps^2) / (1 - (llambda_eps^2))) + it*estep);
  it = it+1;
end

% Transition probability matrix Tauchen (1986)
mm = egrid(2) - egrid(1);
for j = 1:ne
  for k = 1:ne
    if k == 1
      P(j, k) = normcdf((egrid(k) - llambda_eps*egrid(j) + (mm/2))/ssigma_eps);
    elseif k == ne 
      P(j, k) = 1 - normcdf((egrid(k) - llambda_eps*egrid(j) - (mm/2))/ssigma_eps);
    else
      P(j, k) = normcdf((egrid(k) - llambda_eps*egrid(j) + (mm/2))/ssigma_eps) - normcdf((egrid(k) - llambda_eps*egrid(j) - (mm/2))/ssigma_eps);
    end
  end
end

% Exponential of the grid e
for i = 1:ne
  egrid(i) = exp(egrid(i));
end

%--------------------------------%
%     Life-cycle computation     %
%--------------------------------%

disp(' ')
disp('Life cycle computation: ')
disp(' ')

tic;

tempV = zeros(nx*ne);

for age = T:-1:1
    parfor ind = 1:(ne*nx)
            
        ix = int64(floor((ind-0.05)/ne))+1;
        ie = int64(floor(mod(ind-0.05, ne))+1);
        
        VV = -10^3;
        for ixp = 1:1:nx
        
            expected = 0.0;
            if age < T
                for iep = 1:1:ne
                    expected = expected + P(ie, iep)*V(age+1, ixp, iep);
                end
            end
        
            cons  = (1 + r)*xgrid(ix) + egrid(ie)*w - xgrid(ixp);
        
            utility = (cons^(1-ssigma))/(1-ssigma) + bbeta*expected;
        
            if cons <= 0
                utility = -10^(5);
            end
            if utility >= VV
                VV = utility;
            end
        
            utility = 0.0;
        end
      
        tempV(ind) = VV;
     
    end
  
    for ind = 1:nx*ne
        ix = int64(floor((ind-0.05)/ne))+1;
        ie = int64(floor(mod(ind-0.05, ne))+1);

        V(age, ix, ie) = tempV(ind);
    end
  
	finish = toc;
	disp(['Age: ', num2str(age), '. Time: ', num2str(round(finish, 3)), ' seconds.'])
end

disp(' ')
finish = toc;
disp(['TOTAL ELAPSED TIME: ', num2str(finish), ' seconds. '])


%---------------------%
%     Some checks     %
%---------------------%

disp(' ')
disp(' - - - - - - - - - - - - - - - - - - - - - ')
disp(' ')
disp('The first entries of the value function: ')
disp(' ')

for i = 1:3
	disp(num2str(V(1, 1, i)));
end

disp(' ')
quit()

end