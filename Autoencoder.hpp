#ifndef AUTOENCODE_HPP
#define AUTOENCODE_HPP
#include <armadillo>
#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include "Layer.hpp" 
#include "Loss.hpp"
#include "Optimizer.hpp"
/*************************
 
first template parameter is type of loss function  
second template parameter is activation of hidden layer
thrid template parameter is activation of output layer 
fourth template parameter is optimizer 
*************************/
template <class T1, class T2, class T3, class T4>  
class Autoencoder
{
	public:
		Autoencoder(std::vector<int>& v)
		{	
			// 784-500-10-500-784			
			std::vector<int>::iterator it;
			for(it = v.begin(); it != v.end() - 2; it++)
			{
				layerList.emplace_back(new Layer<T2>(*it, *(it+1)));
			}
			layerList.emplace_back(new Layer<T3>(*it, *(it+1)));
		}
		void Feedforward(arma::mat x)
		{
			for(auto& i:layerList)
				x = static_cast<Layer<T2>*>(i.get())->Compute(x);
			output = x;
		}
		void Backpropagation(const arma::mat& batch, \
							const arma::mat& target, double& error)
		{
			arma::mat delta;
			double scalar = alpha * (1./double(batchSize));

			// loss function	
			T1()(target, output, batchSize, delta, error);

			// Optimizer
			T4().template Optimize<T2, T3>(layerList, delta, batch, scalar);
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
			int size = layerList.size() - 1;
			std::string fileName = "weight/weight";
			int i = 0;
			for(;i < size; i++)
			{
				static_cast<Layer<T2>*>(layerList[i].get())->weight.save\
										(fileName + std::to_string(i + 1));
			}
			static_cast<Layer<T3>*>(layerList[i].get())->weight.save\
									(fileName + std::to_string(i + 1));
		}
		void LoadWeight(const std::vector<arma::mat>& weight)
		{
			std::cout << "Loading Weight" << std::endl;
			int size = layerList.size() - 1;
			int i = 0;
			for(; i < size; i++ )
			{
				static_cast<Layer<T2>*>(layerList[i].get())->weight = weight[i];
			}
			static_cast<Layer<T3>*>(layerList[i].get())->weight  = weight[i];
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
		std::vector<std::unique_ptr<Baselayer>> layerList;
};
#endif
