#include <iostream>
#include <armadillo>
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"
#include "mnist.hpp"
#include "GAN.hpp"
#include "Trasfer.hpp"

int main()
{

	int epoch = 10;
	int batchSize = 100;
	double learnRate = 0.02;
	std::vector<int> v = {784, 500, 20, 500, 784};
	std::vector<arma::mat> weight;
	std::string train_file = "mnist/train-images-idx3-ubyte";
	arma::mat data = read_Mnist(train_file) / 255.0;
	std::cout << "Training data size: " << size(data) << std::endl;
	GAN<LeastSquare, Sigmoid, Sigmoid, SGD> g(v);
	g.Train(data, epoch, batchSize, learnRate);
	g.SaveWeight();
	
	// test
	
	
	GAN<LeastSquare, Sigmoid, Sigmoid, SGD> testG(v);
	for(int i= 1; i < 5; i++)
	{

		arma::mat tmp;
		tmp.load("weight" + std::to_string(i));
		weight.push_back(tmp);
	}

	testG.LoadWeight(weight);

	arma::mat testData = data.col(1000) * 255;
	arma::mat testResult = testG.Test(testData) * 255;

	testData.reshape(28, 28);
	testResult.reshape(28, 28);

	auto cvData = to_cvmat(testData);
	auto cvResult = to_cvmat(testResult);
	cv::imwrite("testData.png", cvData);
	cv::imwrite("testResult.png", cvResult);

	return 0;
}
