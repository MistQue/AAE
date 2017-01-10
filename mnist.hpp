#ifndef MNIST_H
#define MNIST_H

#include <math.h>
#include <armadillo>
#include <vector>
#include <string>
int ReverseInt (int i)
{
	int ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return( ch1 << 24) + (ch2 << 16) + (ch3 << 8) + ch4;
}

arma::mat read_Mnist(std::string filename)
{
	std::ifstream file (filename, std::ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		int n_elements = 0;
		file.read((char*)& magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)& number_of_images,sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)& n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)& n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		n_elements = n_rows * n_cols;
		arma::mat data(n_elements, number_of_images);
		for(int i = 0; i < number_of_images; ++i)
		{
			/*
			arma::mat tp(n_rows, n_cols);
			for(int r = 0; r < n_rows; ++r)
			{
				for(int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*) &temp, sizeof(temp));
					tp(r, c) = (double) temp;
				}
			}
			vec.push_back(tp);	
			*/
			for(int e = 0; e < n_elements; e++)
			{
				unsigned char temp = 0;
				file.read((char*)& temp, sizeof(temp));
				data(e, i) = (double) temp;
			}
			
		}
		return data;
	}
	else
	{
		std::cout << "Can not open the file" << std::endl;
	}
}

void read_Mnist_Label(std::string filename, arma::colvec& vec)
{
	std::ifstream file (filename, std::ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		//int n_rows = 0;
		//int n_cols = 0;
		file.read((char*) &magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*) &number_of_images,sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		for(int i = 0; i < number_of_images; ++i)
		{
			unsigned char temp = 0;
			file.read((char*) &temp, sizeof(temp));
			vec(i)= (double)temp;
		}
	}
}

#endif
