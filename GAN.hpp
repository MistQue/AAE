#ifndef AUTOENCODE_HPP
#define AUTOENCODE_HPP
#include <armadillo>
#include <vector>
#include <iostream>
#include <boost/any.hpp> 
#include <boost/range/adaptor/reversed.hpp>
#include "Layer.hpp" 

template <class T>
class GAN
{
	public:
		GAN()
		{	
			batchSize = 4;
			Layer<T> hl1(2, 2);
			Layer<T> hl2(2, 2);
			Layer<T> hl3(2, 1);
			layers.push_back(hl1);
			layers.push_back(hl2);
			layers.push_back(hl3);
		}
		void Feedforward(arma::mat x)
		{
			for(auto& i:layers)
				x = i.Compute(x);
			output = x;
		}
		void Backpropagation(arma::mat& batch, arma::mat& target)
		{
			// Error function is least error 

			arma::mat delta = (output - target);	
			double error = arma::norm(delta, 2) * (1./double(batchSize));
			std::cout << "error:" << error << std::endl;
			double scalar = alpha * (1./double(batchSize));
			int size = layers.size() - 1;

			delta = delta % layers[size].dy;
			arma::mat dw = delta * layers[size - 1].y.t();
			layers[size].weight -= scalar * dw;

			for(int i = size - 1; i > 0; i--)
			{
				delta = (layers[i + 1].weight.t() * delta ) % layers[i].dy;
				dw = delta * layers[i - 1].y.t();
				layers[i].weight -= scalar * dw; 
			}

			delta = ( layers[1].weight.t() * delta) % layers[0].dy;
			dw = delta * batch.t();
			layers[0].weight -= scalar * dw; 
		}
		void Train(const arma::mat& data, const int& epoch)
		{
			arma::mat target = {0, 1, 1, 0};
			for(int i = 1; i <= epoch; i++)
			{
				std::cout << i << " epoch, ";
				for(int j = 0; j < 1; j++)
				{
					auto batch = data;	
					Feedforward(batch);	
					Backpropagation(batch, target);
					std::cout << output << std::endl;	
				}
			}
		
		}
	private:
		int batchSize;
		boost::any layerList;
		arma::mat output;
		std::vector <Layer <T> > layers;
		double alpha = 10; // learning rate 
};
#endif
