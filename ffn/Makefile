all: ffn

ffn: ffn.cpp
	g++ $^ -o $@ -fopenmp -std=c++11 -larmadillo -lmlpack -lboost_serialization

README.md: ffn.cpp
	intmain_docmd c++ $^ $@

plot:
	gnuplot plot.gp

