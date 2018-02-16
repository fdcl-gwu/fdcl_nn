#ifndef _C_SOFTMAX_H
#define _C_SOFTMAX_H

#include <iostream>
#include <vector>
#include <math.h> // pow
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class c_softmax_layer
{
public:
	int N;
	VectorXd x,y;
	MatrixXd df_dx;
	c_softmax_layer(){};
	c_softmax_layer(int N);
	~c_softmax_layer(){};
	
	void init(int N);
	VectorXd f(VectorXd x);
	MatrixXd df(VectorXd x);
	void df_check();
	
private:
	double sum;
};

#endif