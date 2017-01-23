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

	int epoch = 1000;
	int batchSize = 100;
	double learnRate = 0.02;
	std::vector<int> v = {784, 500, 20, 500, 784};
	std::vector<arma::mat> weight;
	std::string train_file = "mnist/train-images-idx3-ubyte";
	arma::mat data = read_Mnist(train_file) / 255.0;
	std::cout << "Training data size: " << size(data) << std::endl;
	GAN<LeastSquare, Sigmoid> g(v);
	g.Train(data, epoch, batchSize, learnRate);
	g.SaveWeight();
	
	// test
	
	/*
	GAN<Sigmoid> testG(epoch, batchSize, learnRate);
	for(int i= 1; i < 7; i++)
	{

		arma::mat tmp;
		tmp.load("weight" + std::to_string(i));
		weight.push_back(tmp);
	}
	testG.LoadWeight(weight);
	*/
	arma::mat testData = data.col(1) * 255;
	arma::mat testMat = g.Test(testData) * 255;

	testData.reshape(28, 28);
	testMat.reshape(28, 28);

	auto cvMat1 = to_cvmat(testData);
	auto cvMat2 = to_cvmat(testMat);
	cv::imwrite("test1.png", cvMat1);
	cv::imwrite("test2.png", cvMat2);

	return 0;
}
