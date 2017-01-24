#include <iostream>
#include <armadillo>
#include <vector>
#include <string>
#include <random>
#include "opencv2/opencv.hpp"
#include "mnist.hpp"
#include "Autoencoder.hpp"
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
	Autoencoder<LeastSquare, Sigmoid, Sigmoid, SGD> AE(v);
	AE.Train(data, epoch, batchSize, learnRate);
	AE.SaveWeight();
	
	// test
	
	
	Autoencoder<LeastSquare, Sigmoid, Sigmoid, SGD> testAE(v);
	int size = v.size();
	for(int i= 1; i < size; i++)
	{

		arma::mat tmp;
		tmp.load("weight/weight" + std::to_string(i));
		weight.push_back(tmp);
	}

	testAE.LoadWeight(weight);

	std::mt19937 rng;
    rng.seed(std::random_device()());
	std::uniform_int_distribution<std::mt19937::result_type> \
									dist(0, data.n_cols - 1);
	for(int i = 1; i <= 100; i++)
	{
		arma::mat testData = data.col(dist(rng)) * 255;
		arma::mat testResult = testAE.Test(testData) * 255;

		testData.reshape(28, 28);
		testResult.reshape(28, 28);

		auto cvData = to_cvmat(testData);
		auto cvResult = to_cvmat(testResult);
		std::string path = "img/";
		cv::imwrite(path + std::to_string(i) + "Data.png", cvData);
		cv::imwrite(path + std::to_string(i) + "Result.png", cvResult);
	}
	return 0;
}
