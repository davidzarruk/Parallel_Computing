
# Home directory
HOMET=/home/davidza/

# Ask the number of cores to be used in the parallelization:
CORES := $(shell bash -c 'read -p "Enter the number of cores for parallelization: " pwd; echo $$pwd') 


# Execute the code for each language
cpp:
	g++ Cpp_main_OpenMP.cpp -fopenmp -o Cpp_main
	export OMP_NUM_THREADS=$(CORES); ./Cpp_main;
	rm Cpp_main

cpp1:
	g++ Cpp_main_OpenMP.cpp -fopenmp -o Cpp_main -O1 
	export OMP_NUM_THREADS=$(CORES); ./Cpp_main;
	rm Cpp_main

cpp2:
	g++ Cpp_main_OpenMP.cpp -fopenmp -o Cpp_main -O2 
	export OMP_NUM_THREADS=$(CORES); ./Cpp_main;
	rm Cpp_main

cpp3:
	g++ Cpp_main_OpenMP.cpp -fopenmp -o Cpp_main -O3 
	export OMP_NUM_THREADS=$(CORES); ./Cpp_main;
	rm Cpp_main

pgi2:
	pgc++ Cpp_main_OpenMP.cpp -o Cpp_main -mp -Minfo=accel -O
	export OMP_NUM_THREADS=$(CORES); ./Cpp_main;
	rm Cpp_main

pgi:
	pgc++ Cpp_main_OpenACC.cpp -o Cpp_main -fast -ta=multicore -acc
	export OMP_NUM_THREADS=$(CORES); ./Cpp_main;
	rm Cpp_main

acc:
	PGI=/opt/pgi;
	PATH=/opt/pgi/linux86-64/17.10/bin:$PATH;
	MANPATH=$MANPATH:/opt/pgi/linux86-64/17.10/man;
	LM_LICENSE_FILE=$LM_LICENSE_FILE:/opt/pgi/license.dat; 
	pgc++ Cpp_main_OpenACC.cpp -o Cpp_main_acc.exe -fast -acc -ta=nvidia 
	./Cpp_main_acc.exe;
	rm Cpp_main_acc.exe

acc_profile:
	pgc++ Cpp_main_OpenACC.cpp -o Cpp_main_acc.exe -fast -acc -ta=nvidia -Minfo=accel,ccff
	./Cpp_main_acc.exe;
	pgprof ./Cpp_main_acc.exe
	rm Cpp_main_acc.exe

omp:
	g++ Cpp_main_OpenACC.cpp -o Cpp_main_omp.exe -fopenmp 
	./Cpp_main_omp.exe;
	rm Cpp_main_omp.exe
	
omp_pgi:
	pgc++ Cpp_main_OpenACC.cpp -o Cpp_main_omp_pgi.exe -fast -mp -ta=multicore
	./Cpp_main_omp_pgi.exe;
	rm Cpp_main_omp_pgi.exe

julia_parallel:
	julia -p$(CORES) Julia_main_parallel.jl

julia_pmap:
	julia -p$(CORES) Julia_main_pmap.jl

Rcpp:
	export OMP_NUM_THREADS=$(CORES); Rscript Rcpp_main.R;

R:
	Rscript R_main.R $(CORES)

python:
	python Python_main.py $(CORES)

matlab:
	matlab -nodesktop -nodisplay -r "Matlab_main $(CORES)"

MPI:
	mpic++ -g MPI_main.cpp -o main
	mpirun -np $(CORES) -hostfile MPI_host_file ./main
	rm main

CUDA:
	export PATH=/usr/local/cuda/bin/:$PATH
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
	nvcc CUDA_main.cu -o main
	./main
	rm main

