#include <iostream>
#include <armadillo>
#include "Layer.hpp"
#include "GAN.hpp"

int main()
{		

	GAN<ReLu> g;
	//Layer<ReLu> l(3, 5);
	std::vector <arma::mat> v;
	
	arma::mat x = {{.0, .0, 1., 1.}, {.0, 1., .0, 1.}};
	arma::mat t = {1.2, 3.1};
	v.push_back(x);
	v.push_back(t);
	g.Train(x, 5000);
    //l.Input = arma::ones<arma::mat>(5);
	//std::cout << "l " << l.Input << std::endl;
	//l.printw();
	//l.check();	
	//arma::mat v = arma::ones<arma::vec>(3);
	//arma::mat m(3, 3);
	//std::cout << arma::size(m) << std::endl;
	//m = arma::ones<arma::mat>(5, 5);
	//std::cout << arma::size(m) << std::endl;
	//std::cout << "v is " << v << std::endl;
	//std::cout << "m is " << m << std::endl;

	//std::cout << "m*v " << m*v << std::endl;
	
	return 0;


}
