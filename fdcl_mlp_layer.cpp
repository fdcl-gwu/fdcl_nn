#include "fdcl_mlp_layer.h"

fdcl_mlp_layer::fdcl_mlp_layer(int N_in, int N_out)
{
	init_io(N_in,N_out);
	
	z.resize(N_out);
	theta.resize(2);
	dJ_dtheta.resize(2);
	
	theta[0].resize(N_out,N_in); // W
	theta[1].resize(N_out,1); // b
	
	dJ_dtheta[0].resize(N_out,N_in);
	dJ_dtheta[1].resize(N_out,1);
	
	theta[0].setRandom();
	theta[1].setRandom();
}

VectorXd fdcl_mlp_layer::f(VectorXd x)
{
	this->x=x;
	
	z=theta[0]*x+theta[1];
	for (int i=0; i<N_out ; i++)
		y(i)=act_func(z(i));
	return y;	
}

void fdcl_mlp_layer::compute_dJ_dtheta(VectorXd e)
{
	int i,j;
	this->e=e;
	
	for(i=0; i<N_out; i++)
	{
		dJ_dtheta[1](i)=d_act_func(z(i))*e(i);
		for (j=0;j<N_in;j++)
		{
			dJ_dtheta[0](i,j)=dJ_dtheta[1](i)*x(j);
		}
	}	
}

VectorXd fdcl_mlp_layer::back_prop(VectorXd e)
{
	VectorXd e_prior, ds_e;
	e_prior.resize(N_in);
	ds_e.resize(N_out);
	
	for(int i=0;i<N_out;i++)
		ds_e(i)=d_act_func(z(i))*e(i);
	
	e_prior=theta[0].transpose()*ds_e;
	
	return e_prior;
}
