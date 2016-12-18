#ifndef NNETS_HPP
#define NNETS_HPP
#include <armadillo>

class Layer
{
	public:
		arma::colvec nodes;
 		arma::Mat<float> weight;	
		void (*fp)();
		void ReLU();
		void Sigmoid();
};

#endif
