#ifndef LAYER_HPP
#define LAYER_HPP
#include <armadillo>
#include <algorithm>
#include <cmath>
#include <boost/random/random_device.hpp>
#include <boost/random/normal_distribution.hpp>

std::random_device rd;
std::default_random_engine g(rd());
std::normal_distribution<> distribution(0, 0.1);



/*
  weight initial by Gaussian Distribution with zero mean, 0.01 variances	 
*/
class Baselayer
{
	public:
		Baselayer()
		{
			std::cout << " Baselayer" << std::endl;
		}
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
			//y = arma::normalise(y);
			dy = T().derivative(y);
			return y;
		}
		void check ()
		{
			std::cout << T()(weight);
			std::cout << T().derivative(weight); 
		}
		void printw()
		{
			std::cout << weight << std::endl;
		}

		arma::mat y;
		arma::mat dy;
		arma::mat weight; // OutSize x InputSize matrix

};
class Sigmoid
{
	public:
		arma::mat operator () (arma::mat x)
		{
			return x.for_each([] (arma::mat::elem_type& val)
					{ 
						val = 1.0 / (1.0 + std::exp(-val) ); 
					} ); 		
		}
		static arma::mat derivative(arma::mat x)
		{
			return x % (1.0 - x);
		}
};
class Hypertan
{
	public:
		arma::mat operator ()(arma::mat x)
		{
			return x.for_each([] (arma::mat::elem_type& val)
					{
						val = tanh(val);
					} ); 
		}

		static arma::mat derivative(arma::mat x)
		{

			return  x.for_each([] (arma::mat::elem_type& val)
					{
						
						val = 1.0 - val*val;
					});
		}	

};
class ReLu
{
	public:
		arma::mat operator ()(arma::mat x)
		{
			return  x.for_each([] (arma::mat::elem_type& val)
					{
						val = std::max(0.0, val);
					} ); 
		}	
		static arma::mat derivative(arma::mat x)
		{

			return  x.for_each([] (arma::mat::elem_type& val)
					{
						val = (val > 0.0 ? 1.0 : 0.0);
					});
		}
};

#endif
