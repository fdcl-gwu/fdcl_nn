#ifndef _FDCL_LAYER_H
#define _FDCL_LAYER_H

#include <iostream>
#include <vector>
#include <math.h> // pow
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

#define LOSS_FUNC_QUAD 0
#define LOSS_FUNC_CROSS_ENTROPY 1
#define LAYER_PC 0
#define LAYER_SF 1

class fdcl_layer
{
public:
	int N_in, N_out;
	VectorXd x, y, e;
	std::vector<MatrixXd> theta;
	std::vector<MatrixXd> dJ_dtheta;	
	fdcl_layer(){};
	~fdcl_layer(){};
	void init_io(int N_in, int N_out);
	virtual VectorXd f(VectorXd)=0;
	virtual void compute_dJ_dtheta(VectorXd)=0;
	virtual VectorXd back_prop(VectorXd)=0;
	virtual double act_func(double)=0;
	virtual double d_act_func(double)=0;
};

#endif