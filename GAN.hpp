#ifndef AUTOENCODE_HPP
#define AUTOENCODE_HPP
#include <armadillo>
#include <vector>
#include <string>
#include <iostream>
#include "Layer.hpp" 
#include "Loss.hpp"
/*************************
 
first template parameter is type of loss function  
second template parameter is activation of hidden layer
thrid template parameter is activation of output layer 
*************************/
template <class T1, class T2>  
class GAN
{
	public:
		GAN(std::vector<int>& v)
		{	
			// 784-500-10-500-784			
			for(std::vector<int>::iterator it = v.begin(); \
				it != v.end() - 1; it++)
			{
				Layer<T2> l(*it, *(it + 1));
				layerList.push_back(l);
			}
		}
		void Feedforward(arma::mat x)
		{
			for(auto& i:layerList)
				x = i.Compute(x);
			output = x;
		}
		void Backpropagation(const arma::mat& batch, 
							const arma::mat& target, double& error)
		{
			arma::mat delta;
			
			T1()(target, output, batchSize, delta, error);
			
			double scalar = alpha * (1./double(batchSize));

			int size = layerList.size() - 1;

			delta = delta % layerList[size].dy;
			arma::mat dw = delta * layerList[size - 1].y.t();
			layerList[size].weight -= scalar * dw;

			for(int i = size - 1; i > 0; i--)
			{
				delta = (layerList[i + 1].weight.t() * delta ) % \
						layerList[i].dy;
				dw = delta * layerList[i - 1].y.t();
				layerList[i].weight -= scalar * dw; 
			}

			delta = (layerList[1].weight.t() * delta) % layerList[0].dy;
			dw = delta * batch.t();
			layerList[0].weight -= scalar * dw; 
		}
		void Train(const arma::mat& data, const int& epoch, \
				   const int& batchSize, const double& alpha)
		{
			double error = .0;
			this->batchSize = batchSize;
			this->alpha = alpha;
			int n_cols = data.n_cols;
			for(int i = 1; i <= epoch; i++)
			{
				std::cout << i << " - epoch -";
				int col = 0;
				const arma::mat& shuffleData = arma::shuffle(data, 1);
				while(col < n_cols)
				{
					int range = col + batchSize - 1;
					range = (range > n_cols ? n_cols - 1 : range); 

					const auto& batchData = shuffleData.cols(col, range);	
					const auto& batchTarget = shuffleData.cols(col, range);

					Feedforward(batchData);	
					Backpropagation(batchData, batchTarget, error);
					col += batchSize;
				}

				std::cout << "error : " << error << std::endl;
//				std::cout << output << std::endl;
			}
			std::cout << "Finish" << std::endl;	
		}
		void SaveWeight()
		{
			int l = 1;
			std::string fileName = "weight";
			for(auto& i:layerList)
			{
				i.weight.save(fileName + std::to_string(l));
				l++;	
			}
		}
		void LoadWeight(const std::vector<arma::mat>& weight)
		{
			std::cout << "Loading Weight" << std::endl;
			for(int i = 0; i < 6; i++ )
				layerList[i].weight = weight[i];
			std::cout << "Loading Finish" << std:: endl;
		}
		arma::mat Test(const arma::mat& data)
		{
			std::cout << "Test" << std::endl;
			Feedforward(data);
			std::cout << "Test Finish" << std::endl;
			return output;
		}
	private:
		int epoch;
		int batchSize;
		double alpha;
		arma::mat output;
		std::vector < Layer<T2> > layerList;
};
#endif
