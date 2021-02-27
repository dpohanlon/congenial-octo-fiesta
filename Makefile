all:
	g++ -D_GLIBCXX_USE_CXX11_ABI=0 -O3 -std=c++17 -Wall -L/cvmfs/sft.cern.ch/lcg/releases/LCG_96python3/Boost/1.70.0/x86_64-centos7-gcc8-opt/lib/ gwMCMC.cpp -lboost_timer -lboost_program_options -lpoplin -lpopops -lpoputil -lpoplar -lpoprand -lpoprithms -o gwMCMC
