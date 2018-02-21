#include "fdcl_softmax_layer.h"

fdcl_softmax_layer::fdcl_softmax_layer(int N_in, int N_out)
{
	init_io(N_in,N_out);
	
	z.resize(N_out);
	theta.resize(1);
	dJ_dtheta.resize(1);
	
	theta[0].resize(N_out,N_in); // W
	
	dJ_dtheta[0].resize(N_out,N_in);
	
	theta[0].setRandom();
}

VectorXd fdcl_softmax_layer::f(VectorXd x)
{
	int i;
	double sum=0., z_max;

	this->x=x;
	
	z=theta[0]*x;
	z_max=z.maxCoeff();
	
	for(i=0; i<y.size(); i++)
		y(i)=exp(z(i)-z_max);
	
	for(i=0; i<y.size(); i++)
		sum+=y(i);
	
	for(i=0; i<y.size(); i++)
		y(i)/=sum;
	
	return y;
}

void fdcl_softmax_layer::compute_dJ_dtheta(VectorXd e)
{
	int i,j;
	MatrixXd df_dx;
	this->e=e;
	
	df_dx.resize(N_out,N_out);
	
	for(i=0;i<N_out;i++)
	{
		df_dx(i,i)=y(i)*(1.0-y(i));
		for(j=i+1;j<N_out;j++)
		{
			df_dx(i,j)=-y(i)*y(j);
			df_dx(j,i)=df_dx(i,j);
		}
	}

	dJ_dtheta[0]=df_dx*e*x.transpose();	
}

VectorXd fdcl_softmax_layer::back_prop(VectorXd e)
{
	int i,j;
	VectorXd e_prior;
	MatrixXd df_dx;
	e_prior.resize(N_in);
	df_dx.resize(N_out,N_out);

	for(i=0;i<N_out;i++)
	{
		df_dx(i,i)=y(i)*(1.0-y(i));
		for(j=i+1;j<N_out;j++)
		{
			df_dx(i,j)=-y(i)*y(j);
			df_dx(j,i)=df_dx(i,j);
		}
	}

	e_prior=theta[0].transpose()*df_dx*e;
	return e_prior;	
}
