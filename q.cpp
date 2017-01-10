#include <iostream>
#include <armadillo>
#include <boost/range/adaptor/reversed.hpp>
using namespace std;

int main()
{
	arma::mat q = {{1, 2}, {3, 4}};
	arma::mat p = {{2, 1}, {3, 4}};
	vector<int> v = {1, 2, 3, 4};
	for(auto& i : boost::adaptors::reverse(v))
		cout << i << endl;
	//cout << p << endl;
	//cout << a << endl;
	return 0;

}
