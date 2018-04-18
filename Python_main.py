
#--------------------------------#
#         House-keeping          #
#--------------------------------#

import numpy
import math
import time
from scipy.stats import norm
from joblib import Parallel, delayed
import multiprocessing
import sys

#--------------------------------#
#         Initialization         #
#--------------------------------#

# Number of workers
num_cores = int(sys.argv[1]);

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

# Initialize the grid for X
xgrid = numpy.zeros(nx)

# Initialize the grid for E and the transition probability matrix
egrid = numpy.zeros(ne)
P     = numpy.zeros((ne, ne))

# Initialize value function V
V     = numpy.zeros((T, nx, ne))


#--------------------------------#
#         Grid creation          #
#--------------------------------#

# Function to construct the grid for capital (x)
size = nx;
xstep = (xmax - xmin) /(size - 1);
it = 0;
for i in range(0,nx):
	xgrid[i] = xmin + it*xstep;
	it = it+1;


# Function to construct the grid for productivity (e) using Tauchen (1986)
size = ne;
ssigma_y = math.sqrt(math.pow(ssigma_eps, 2) / (1 - math.pow(llambda_eps,2)));
estep = 2*ssigma_y*m / (size-1);
it = 0;
for i in range(0,ne):
	egrid[i] = (-m*math.sqrt(math.pow(ssigma_eps, 2) / (1 - math.pow(llambda_eps,2))) + it*estep);
	it = it+1;


# Function to construct the transition probability matrix for productivity (P) using Tauchen (1986)
mm = egrid[1] - egrid[0];
for j in range(0,ne):
	for k in range(0,ne):
		if (k == 0):
			P[j, k] = norm.cdf((egrid[k] - llambda_eps*egrid[j] + (mm/2))/ssigma_eps);
		elif (k == ne-1):
			P[j, k] = 1 - norm.cdf((egrid[k] - llambda_eps*egrid[j] - (mm/2))/ssigma_eps);
		else:
			P[j, k] = norm.cdf((egrid[k] - llambda_eps*egrid[j] + (mm/2))/ssigma_eps) - norm.cdf((egrid[k] - llambda_eps*egrid[j] - (mm/2))/ssigma_eps);


# Exponential of the grid e
for i in range(0,ne):
	egrid[i] = math.exp(egrid[i]);



#--------------------------------#
#     Structure and function     #
#--------------------------------#

# Value function
VV = math.pow(-10, 5);


# Data structure of state and exogenous variables
class modelState(object):
	def __init__(self,ind,ne,nx,T,age,P,xgrid,egrid,ssigma,bbeta,w,r):
		self.ind		= ind
		self.ne			= ne
		self.nx			= nx
		self.T			= T
		self.age		= age
		self.P			= P
		self.xgrid		= xgrid
		self.egrid		= egrid
		self.ssigma		= ssigma
		self.bbeta		= bbeta
		self.w			= w
		self.r			= r

# Function that returns value for a given state
# ind: a unique state that corresponds to a pair (ie,ix)
def value_func(states):

	ind = states.ind
	ne = states.ne
	nx = states.nx
	T = states.T
	age = states.age
	P = states.P
	xgrid = states.xgrid
	egrid = states.egrid
	ssigma = states.ssigma
	bbeta = states.bbeta
	w = states.w
	r = states.r

	ix = int(math.floor(ind/ne));
	ie = int(math.floor(ind%ne));

	VV = math.pow(-10, 3)
	for ixp in range(0,nx):
		expected = 0.0;
		if(age < T-1):
			for iep in range(0,ne):
				expected = expected + P[ie, iep]*V[age+1, ixp, iep]

		cons  = (1 + r)*xgrid[ix] + egrid[ie]*w - xgrid[ixp];

		utility = math.pow(cons, (1-ssigma))/(1-ssigma) + bbeta*expected;

		if(cons <= 0):
			utility = math.pow(-10,5);
		
		if(utility >= VV):
			VV = utility;

		utility = 0.0;

	return[VV];



#--------------------------------#
#     Life-cycle computation     #
#--------------------------------#

print(" ")
print("Life cycle computation: ")
print(" ")


start = time.time()

for age in reversed(range(0,T)):

	# This function computes `value_func` in parallel for all the states
	results = Parallel(n_jobs=num_cores)(delayed(value_func)(modelState(ind,ne,nx,T,age,P,xgrid,egrid,ssigma,bbeta,w,r)) for ind in range(0,nx*ne))

	# I write the results on the value matrix: V
	for ind in range(0,nx*ne):
		
		ix = int(math.floor(ind/ne));
		ie = int(math.floor(ind%ne));

		V[age, ix, ie] = results[ind][0];

	finish = time.time() - start
	print "Age: ", age+1, ". Time: ", round(finish, 4), " seconds."

finish = time.time() - start
print "TOTAL ELAPSED TIME: ", round(finish, 4), " seconds. \n"


#---------------------#
#     Some checks     #
#---------------------#

print " - - - - - - - - - - - - - - - - - - - - - \n"
print "The first entries of the value function: \n"

for i in range(0,3):
	print(round(V[0, 0, i], 5))

print " \n"

