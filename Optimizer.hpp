#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP
#include <armadillo>
#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include "Layer.hpp"

class SGD
{
	public:
		template<class T1, class T2>
		void Optimize(std::vector<std::unique_ptr<Baselayer> >& layerList,\
						arma::mat& delta,const arma::mat& batch, double& scalar) 
		{
			int size = layerList.size() - 1;

			delta = delta % static_cast<Layer<T2>*>(layerList[size].get())->dy;
			arma::mat dw = delta * static_cast<Layer<T2>*> \
						 	(layerList[size - 1].get())->y.t();
			static_cast<Layer<T2>*>(layerList[size].get())->weight -= scalar *\
																	  dw;

			for(int i = size - 1; i > 0; i--)
			{
				delta = (static_cast<Layer<T1>*>\
						(layerList[i + 1].get())->weight.t() * delta ) % \
						static_cast<Layer<T1>*>(layerList[i].get())->dy;
				dw = delta * static_cast<Layer<T1>*>\
					(layerList[i - 1].get())->y.t();
				static_cast<Layer<T1>*>(layerList[i].get())->weight -= scalar * dw; 
			}

			delta = (static_cast<Layer<T1>*>\
					(layerList[1].get())->weight.t() * delta) % \
				static_cast<Layer<T1>*>(layerList[0].get())->dy;
			dw = delta * batch.t();
			static_cast<Layer<T1>*>(layerList[0].get())->weight -= scalar * dw; 
		}		
};
class adam
{
	public:
		template<class T1, class T2>
		void Optimize(std::vector<std::unique_ptr<Baselayer> >& layerList,\
						arma::mat& delta,const arma::mat& batch, double& scalar) 
		{
		}		
};
#endif
