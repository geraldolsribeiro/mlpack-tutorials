all: mlp-mlpack

mlp-mlpack: mlp-mlpack.cpp
	g++ mlp-mlpack.cpp -o mlp-mlpack -fopenmp -std=c++11 -larmadillo -lmlpack -lboost_serialization

README.md: mlp-mlpack.cpp
	intmain_docmd c++ $^ $@

plot:
	gnuplot plot.gp

