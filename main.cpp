#include <iostream>
#include <armadillo>
#include <vector>
#include <string>
#include "mnist.hpp"
#include "Trasfer.hpp"
int main()
{

	std::string train_file = "mnist/train-images-idx3-ubyte";
	std::vector<arma::mat> vec;

	read_Mnist(train_file, vec);
	std::cout << "training data size:" << vec.size() << std::endl;
	auto i = to_cvmat(vec[0]);	


	return 0;
}
