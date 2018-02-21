#ifndef _FDCL_NN_H
#define _FDCL_NN_H

#include "fdcl_layer.h"
#include "fdcl_mlp_layer.h"
#include "fdcl_softmax_layer.h"


class fdcl_nn
{
public:
	std::vector<std::unique_ptr<fdcl_layer>> layer;
	std::vector<int> N, layer_type;
	int N_layer, N_in, N_out, N_data;
	static int LOSS_FUNC_TYPE;
	std::vector<VectorXd> X_data,Y_data;
	
	fdcl_nn(){};
	fdcl_nn(std::vector<int>, std::vector<int>);
	void init(std::vector<int>, std::vector<int>);
	VectorXd f(VectorXd);
	double J(VectorXd, VectorXd);
	double J(std::vector<VectorXd> X, std::vector<VectorXd> Y);
	void compute_dJ_dtheta(VectorXd, VectorXd);
	void dJ_check();
	VectorXd back_prop(VectorXd, VectorXd);
	void init_data(int);
	void grad_descent(int);
	void grad_descent();
	
};

#endif
