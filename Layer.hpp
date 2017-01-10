#ifndef LAYER_HPP
#define LAYER_HPP
#include <armadillo>
#include <algorithm>
#include <boost/random/random_device.hpp>
#include <boost/random/normal_distribution.hpp>


std::random_device rd;
std::default_random_engine g(rd());
std::normal_distribution<> distribution(5.0, 0.1);
/*
  weight initial by Gaussian Distribution with zero mean, 0.01 variances
	 
*/
template < class T> //class T is the activation type
class Layer
{
	public:
		Layer(int inputSize, int outputSize)
		{ 
			//arma::mat tmp(OutSize, InputSize);							   
			weight = arma::randu<arma::mat>(outputSize, inputSize);
			/* 
			weight = tmp.for_each([] (arma::mat::elem_type& val) 
					{
						val = distribution(g);	
					});
			weight = tmp;
			*/
		}
		arma::mat Compute(arma::mat x)
		{
			y = T()(weight * x);
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
			auto y = x.for_each([] (arma::mat::elem_type& val)
					{ 
						val = 1.0 / (1.0 + std::exp(-val) ); 
					} ); 		
			return y;
		}
		static arma::mat derivative(arma::mat x)
		{
			auto y = x.for_each([] (arma::mat::elem_type& val)
					{ 
						val = val * (1.0 - val);
					} ); 		
			return y;
		}
};
class ReLu
{
	public:
		arma::mat operator ()(arma::mat x)
		{
			auto y = x.for_each([] (arma::mat::elem_type& val)
					{
						val = std::max(0.0, val);
					} ); 
			return y;
		}	
		static arma::mat derivative(arma::mat x)
		{

			auto y = x.for_each([] (arma::mat::elem_type& val)
					{
						val = (val > 0.0 ? 1.0 : 0.0);
					});
			return y;
		}
};


#endif
