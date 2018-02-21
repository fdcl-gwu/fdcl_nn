#ifndef _FDCL_SOFTMAX_LAYER_H
#define _FDCL_SOFTMAX_LAYER_H

#include "fdcl_layer.h"

class fdcl_softmax_layer : public fdcl_layer
{
public:
	VectorXd z;
	fdcl_softmax_layer(){};
	fdcl_softmax_layer(int, int);
	~fdcl_softmax_layer(){};

	VectorXd f(VectorXd);
	void compute_dJ_dtheta(VectorXd);
	VectorXd back_prop(VectorXd);
	double act_func(double x){return 0.;};
	double d_act_func(double x){return 0.; };		
};

#endif