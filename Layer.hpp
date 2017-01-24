#ifndef LAYER_HPP
#define LAYER_HPP
#include <armadillo>
#include <algorithm>
#include <cmath>
#include <boost/random/random_device.hpp>
#include <boost/random/normal_distribution.hpp>
#include "Activation.hpp"
std::random_device rd;
std::default_random_engine g(rd());
std::normal_distribution<> distribution(0, 0.1);



/*
  weight initial by Gaussian Distribution with zero mean, 0.01 variances	 
*/
class Baselayer
{
};

template <class T> //class T is the activation type
class Layer:public Baselayer
{
	// armadillo initial is not good....
	public:
		Layer(int inputSize, int outputSize): weight(outputSize, inputSize)
		{ 
			
			weight.for_each([] (arma::mat::elem_type& val) 
					{
						val = distribution(g);	
					});
			
		}
		arma::mat Compute(arma::mat& x)
		{
			arma::mat qq = weight * x;
			y = T()(qq);
			dy = T().derivative(y);
			return y;
		}
		arma::mat y;
		arma::mat dy;

		arma::mat weight; // OutSize x InputSize matrix

};
#endif
