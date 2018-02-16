#include "c_mlp.h"

c_mlp_layer::c_mlp_layer(int N_in, int N_out)
{
	init(N_in,N_out);
}

void c_mlp_layer::init(int N_in, int N_out)
{
	this->N_in=N_in;
	this->N_out=N_out;
	
	W.resize(N_out,N_in);
	dJ_dW.resize(N_out,N_in);
	dJ_db.resize(N_out);
	b.resize(N_out);
	y.resize(N_out);
	dy.resize(N_out);
	z.resize(N_in);
	e.resize(N_out);
	
	W.setRandom();
	b.setRandom();
	y.setZero();
	dy.setZero();
	z.setZero();
	dJ_dW.setZero();
	dJ_db.setZero();
	e.setZero();		
}

VectorXd c_mlp_layer::f(VectorXd x)
{	
	this->x=x;
	z=W*x+b;
	for (int i=0; i<N_out ; i++)
		y(i)=s(z(i));
	return y;
}

VectorXd c_mlp_layer::df(VectorXd x)
{	
	this->x=x;
	z=W*x+b;
	for (int i=0; i<N_out ; i++)
		dy(i)=ds(z(i));
	return dy;
}

void c_mlp_layer::dJ(VectorXd e)
{
	int i,j;
	this->e=e;
	
	for(i=0; i<N_out; i++)
	{
		dJ_db(i)=ds(z(i))*e(i);
		for (j=0;j<N_in;j++)
		{
			dJ_dW(i,j)=dJ_db(i)*x(j);
		}
	}
}


void c_mlp::init(std::vector<int> N, int LOSS_FUNC_TYPE)
{
	this->N.resize(N.size());
	this->LOSS_FUNC_TYPE = LOSS_FUNC_TYPE;
	
	for (int i=0;i<N.size(); i++)
		this->N[i]=N[i];

	N_layer = N.size()-1;
	layer.resize(N_layer);

//	x.resize(N[0]);
//	y.resize(N[N_layer]);	
	
	for(int i=0;i<N_layer;i++)
		layer[i].init(N[i],N[i+1]);
	
};

VectorXd c_mlp::f(VectorXd x)
{
	layer[0].y=layer[0].f(x);
	
	for(int i=0;i<N_layer-1;i++)
		layer[i+1].y=layer[i+1].f(layer[i].y);
	
	return layer[N_layer-1].y;	
}

double c_mlp::J(VectorXd x, VectorXd y)
{
	double J;
	VectorXd fx;
	
	fx=f(x);
	
	switch (LOSS_FUNC_TYPE)
	{
		case LOSS_FUNC_QUAD :		
			J=1./2.*pow((y-fx).norm(),2);			
		break;
		
		case LOSS_FUNC_CROSS_ENTROPY :
			J=0.;
			for(int i=0; i< y.size(); i++)	
				J+=-y(i)*log(fx(i))-(1.-y(i))*log(1-fx(i));
		break;
		
	}
	return J;
}

void c_mlp::dJ(VectorXd x, VectorXd e)
{
	f(x);	
	layer[N_layer-1].e=e;
	layer[N_layer-1].dJ(layer[N_layer-1].e);
	
	for(int i=N_layer-2; i>=0; i--)
	{
		layer[i].e=layer[i+1].W.transpose()*(layer[i+1].df(layer[i+1].x).cwiseProduct(layer[i+1].e));
		layer[i].dJ(layer[i].e);
	}
			
}

void c_mlp::dJ_check()
{
	double J, del_J, del_J_approx, eps;
	MatrixXd dW;
	VectorXd db, x, y;
	
	x.resize(layer[0].N_in);
	y.resize(layer[N_layer-1].N_out);
	dW.resize(layer[0].N_out,layer[0].N_in);
	db.resize(layer[0].N_out);

	x.setRandom();
	y.setRandom();
	dW.setRandom();
	db.setRandom();
	eps=1.e-3;

	dW=dW*eps;
	db=db*eps;
	
	J=this->J(x,y);
	this->dJ(x,f(x)-y);

	layer[0].W+=dW;
	del_J=this->J(x,y)-J;
	
	del_J_approx=(dW.transpose()*layer[0].dJ_dW).trace();

	cout << "c_mlp::dJ_check" << endl;
	cout << "dW" << endl;
	cout << "del_J_exact  = " << del_J << endl;
	cout << "del_J_approx = " << del_J_approx << endl;
	
	layer[0].W-=dW;
	layer[0].b+=db;
	del_J=this->J(x,y)-J;	
	del_J_approx=db.transpose()*layer[0].dJ_db;
	
	cout << "db" << endl;
	cout << "del_J_exact  = " << del_J << endl;
	cout << "del_J_approx = " << del_J_approx << endl << endl;

	layer[0].b-=db;		
}

