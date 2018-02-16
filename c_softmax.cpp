#include "c_softmax.h"

c_softmax_layer::c_softmax_layer(int N)
{
	init(N);
}

void c_softmax_layer::init(int N)
{
	this->N = N;

	x.resize(N);
	y.resize(N);
	df_dx.resize(N,N);

	x.setZero();
	y.setZero();
	df_dx.setZero();
}

VectorXd c_softmax_layer::f(VectorXd x)
{
	int i;
	sum=0.;
	
	for(i=0; i<x.size(); i++)
		y(i)=exp(x(i));
	
	for(i=0; i<x.size(); i++)
		sum+=y(i);
	
	for(i=0; i<x.size(); i++)
		y(i)/=sum;
	
	return y;
}

MatrixXd c_softmax_layer::df(VectorXd x)
{
	int i,j;
	y=f(x);
	
	for(i=0;i<N;i++)
	{
		df_dx(i,i)=y(i)*(1.0-y(i));
		for(j=i+1;j<N;j++)
		{
			df_dx(i,j)=-y(i)*y(j);
			df_dx(j,i)=df_dx(i,j);
		}
	}
	
	return df_dx;
}

void c_softmax_layer::df_check()
{
	VectorXd dx, del_f, del_f_approx;
	double eps;
	
	del_f.resize(N);
	del_f_approx.resize(N);
	dx.resize(N);
	x.setRandom();
	dx.setRandom();
	eps=1.e-3;
	
	del_f=f(x+eps*dx)-f(x);
	
	del_f_approx=eps*df(x)*dx;
	
	cout << "c_softmax_layer::df_check" << endl;
	cout << "del_f_exact  = " << del_f.transpose() << endl;
	cout << "del_f_approx = " << del_f_approx.transpose() << endl << endl;
	
}