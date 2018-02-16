#ifndef _C_MLP_H
#define _C_MLP_H

#include <iostream>
#include <vector>
#include <math.h> // pow
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

#define LOSS_FUNC_QUAD 0
#define LOSS_FUNC_CROSS_ENTROPY 1


class c_mlp_layer
{
public:
	int N_in, N_out;
	MatrixXd W, dJ_dW;
	VectorXd b, x, y, z, dJ_db, e;

	c_mlp_layer(){};
	c_mlp_layer(int N_in, int N_out);
	~c_mlp_layer(){};
	void init(int N_in, int N_out);
	VectorXd f(VectorXd x);
	VectorXd df(VectorXd x);
	void dJ(VectorXd e);
	
private:
	VectorXd dy;
	double s(double x){ return 1./(1.+exp(-x)); };
	double ds(double x){ return s(x)*(1.-s(x)); };	
};


class c_mlp
{
public:
	c_mlp(){};
	~c_mlp(){};
	int N_layer;
	std::vector<c_mlp_layer> layer;
	std::vector<int> N;
//	VectorXd x,y;
	int LOSS_FUNC_TYPE;
	
	void init(std::vector<int> N, int LOSS_FUNC_TYPE);
	VectorXd f(VectorXd);
	void dJ(VectorXd, VectorXd);
	double J(VectorXd, VectorXd);
	void dJ_check();
};



#endif