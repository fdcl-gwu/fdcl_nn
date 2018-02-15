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
	
private:
	double sum;
};

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
	this->x = x;
	sum=0.;
	
	for(i=0; i<x.size(); i++)
		sum+=exp(x(i));
	
	for(i=0; i<x.size(); i++)
		y(i)=exp(x(i))/sum;
	
	return y;
}

MatrixXd c_softmax_layer::df(VectorXd x)
{
	int i,j;
	y=f(x);
	
	for(i=0;i<N;i++)
	{
		df_dx(i,i)=y(i);
		for(j=i;j<N;j++)
		{
			df_dx(i,j)-=y(i)*y(j);
		}
	}
	for (i=0;i<N;i++)
		for(j=i+1;j<N;j++)
			df_dx(j,i)=df_dx(i,j);
	
	return df_dx;
}
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

class c_mlp
{
public:
	c_mlp(){};
	~c_mlp(){};
	int N_layer;
	std::vector<c_mlp_layer> layer;
	std::vector<int> N;
//	VectorXd x,y;
	
	void init(std::vector<int> N);
	VectorXd f(VectorXd);
	void dJ(VectorXd, VectorXd);
	double J(VectorXd, VectorXd);
	void dJ_check();
};

double c_mlp::J(VectorXd x, VectorXd y)
{
	double J;
	
	J=1./2.*pow((y-f(x)).norm(),2);
	return J;
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
	this->dJ(x,y);

	layer[0].W+=dW;
	del_J=this->J(x,y)-J;
	
	del_J_approx=(dW.transpose()*layer[0].dJ_dW).trace();

	cout << "dW" << endl;
	cout << "del_J_exact  = " << del_J << endl;
	cout << "del_J_approx = " << del_J_approx << endl;
	
	layer[0].W-=dW;
	layer[0].b+=db;
	del_J=this->J(x,y)-J;	
	del_J_approx=db.transpose()*layer[0].dJ_db;
	
	cout << "db" << endl;
	cout << "del_J_exact  = " << del_J << endl;
	cout << "del_J_approx = " << del_J_approx << endl;

	layer[0].b-=db;		
}

void c_mlp::dJ(VectorXd x, VectorXd y)
{
	
	layer[N_layer-1].e=f(x)-y;
	layer[N_layer-1].dJ(layer[N_layer-1].e);
	
	for(int i=N_layer-2; i>=0; i--)
	{
		layer[i].e=layer[i+1].W.transpose()*(layer[i+1].df(layer[i+1].x).cwiseProduct(layer[i+1].e));
		layer[i].dJ(layer[i].e);
	}
			
}
void c_mlp::init(std::vector<int> N)
{
	this->N.resize(N.size());
	
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

class c_mlp_classifier
{
public:
	c_mlp mlp;
	c_softmax_layer sf;
	std::vector<int> N;
	VectorXd x,y;
	int N_data, N_in, N_out;
	MatrixXd X,Y;
	
	c_mlp_classifier(){};
	~c_mlp_classifier(){};
	void init(std::vector<int> N);
	VectorXd f(VectorXd);
};

void c_mlp_classifier::init(std::vector<int> N)
{
	this->N.resize(N.size());

	for (int i=0;i<N.size(); i++)
		this->N[i]=N[i];
	
	N_in=N[0];
	N_out=N[N.size()-1];
	
	mlp.init(N);
	sf.init(N_out);
	
	x.resize(N_in);
	y.resize(N_out);
	
	X.resize(N_in,N_data);
	Y.resize(N_out,N_data);	
}

VectorXd c_mlp_classifier::f(VectorXd x)
{
	this->x=x;
	
	y=sf.f(mlp.f(x));
	return y;
}

int main()
{
	std::vector<int> N;
	VectorXd x,y;
	
	c_mlp_classifier mlp_clf;
	
	N = {2, 2, 4};
	mlp_clf.init(N);
	x.resize(N[0]);
	x.setRandom();
	
	cout << mlp_clf.f(x) << endl;
}