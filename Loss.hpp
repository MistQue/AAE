#ifndef LOSS_HPP
#define LOSS_HPP
#include <armadillo>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <boost/iterator/zip_iterator.hpp>
#include "Layer.hpp" 

class LeastSquare
{
	public:
		void operator() (const arma::mat& target, const arma::mat& output,\
				        const int& batchSize, arma::mat& delta, double& error)
		{
			delta = output - target;	
			error = arma::norm(delta, 2) * 0.5 * (1./ double(batchSize));

		}		
};
class Crossentropy
{
	public:
		void operator() (const arma::mat& target, const arma::mat& output,\
				        const int& batchSize, arma::mat& delta, double& error)
		{
			arma::mat logOutput = output;
			logOutput.for_each([](arma::mat::elem_type& val)
			{
					val = log(val);
			});
			delta = output - target;
			error = arma::accu( target % logOutput + (1 - target) % \
												(1 - logOutput));
			error /= -double(batchSize);
		}
};
#endif
