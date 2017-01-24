#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP
#include <armadillo>
#include <algorithm>
#include <cmath>

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
