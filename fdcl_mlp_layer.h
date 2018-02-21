#ifndef _FDCL_MLP_LAYER_H
#define _FDCL_MLP_LAYER_H

#include "fdcl_layer.h"

class fdcl_mlp_layer : public fdcl_layer
{
public:
	VectorXd z;
	fdcl_mlp_layer(){};
	fdcl_mlp_layer(int, int);
	~fdcl_mlp_layer(){};

	VectorXd f(VectorXd);
	void compute_dJ_dtheta(VectorXd);
	VectorXd back_prop(VectorXd);
	double act_func(double x){return 1./(1.+exp(-x));};
	double d_act_func(double x){return act_func(x)*(1.-act_func(x)); };		
};

#endif