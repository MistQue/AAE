#ifndef TRANSFER_HPP
#define TRANSFER_HPP

#include <armadillo>
#include <opencv2/core/core.hpp>

static void Cv_mat_to_arma_mat(const cv::Mat& cv_mat_in, arma::mat& arma_mat_out)
{//convert unsigned int cv::Mat to arma::Mat<double>
	for(int r=0;r<cv_mat_in.rows;r++){
		for(int c=0;c<cv_mat_in.cols;c++){
			arma_mat_out(r,c)=cv_mat_in.data[r*cv_mat_in.cols+c]/255.0;
		}
	}
};
	template<typename T>
static void Arma_mat_to_cv_mat(const arma::Mat<T>& arma_mat_in,cv::Mat_<T>& cv_mat_out)
{
	cv::transpose(cv::Mat_<T>(static_cast<int>(arma_mat_in.n_cols),
					static_cast<int>(arma_mat_in.n_rows),
					const_cast<T*>(arma_mat_in.memptr())),
					cv_mat_out);
}

template <typename T>
cv::Mat_<T> to_cvmat(const arma::Mat<T> &src)
{
  return cv::Mat_<double>{int(src.n_cols), int(src.n_rows), const_cast<T*>(src.memptr())};
}

#endif 
