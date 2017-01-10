#include <iostream>
#include <armadillo>
#include <vector>
#include <string>
#include "mnist.hpp"
#include "GAN.hpp"
#include "Trasfer.hpp"
int main()
{

	std::string train_file = "mnist/train-images-idx3-ubyte";

	arma::mat data = read_Mnist(train_file);
	std::cout << size(data) << std::endl;
	arma::mat test = data.col(0);
	test.reshape(28, 28);
	std::cout << size(test) << std::endl;
	//int n = data.size();
	GAN<Sigmoid> g; 	
	//arma::mat c = arma::expmat(m);
	//std::cout << "is "<< c << std::endl;
	auto i = to_cvmat(test);	



	return 0;
}
