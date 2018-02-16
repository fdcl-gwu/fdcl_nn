#include <iostream>
#include <vector>
#include <math.h> // pow
#include <Eigen/Dense>

#include "c_softmax.h"
#include "c_mlp.h"

using namespace std;
using namespace Eigen;

class c_mlp_classifier
{
public:
	c_mlp mlp;
	c_softmax_layer sf;
	std::vector<int> N;
	VectorXd y;
	int N_data, N_in, N_out, LOSS_FUNC_TYPE;
	std::vector<VectorXd> X_data,Y_data;
	
	c_mlp_classifier(){};
	~c_mlp_classifier(){};
	void init(std::vector<int> N, int LOSS_FUNC_TYPE);
	VectorXd f(VectorXd);
	double J(VectorXd, VectorXd);
	double J(std::vector<VectorXd>, std::vector<VectorXd>);

	void dJ(VectorXd, VectorXd);
	void dJ_check();
	void init_data(int);
	void grad_descent(int);
	void grad_descent();

};

void c_mlp_classifier::init_data(int N_data)
{
	this->N_data=N_data;
	
	X_data.resize(N_data);
	Y_data.resize(N_data);
	
	for (int i=0; i<N_data; i++)
	{
		X_data[i].resize(N_in);
		Y_data[i].resize(N_out);		
	}
}

void c_mlp_classifier::grad_descent(int N_epoch)
{
	for(int i=0; i < N_epoch; i++)
	{
		grad_descent();
		if (i % 100 == 0)
			cout << "epoch = " << i << "  J = " << J(X_data,Y_data) << endl;
	}
}

void c_mlp_classifier::grad_descent()
{
	int i, j;
	double eta=0.9; // learning rate
	double lambda=0.0; // quadratic penalty on weight
	
	for(i=0; i< N_data; i++)
	{
		dJ(X_data[i], Y_data[i]);
		for (j=0; j<mlp.N_layer; j++)
		{
			mlp.layer[j].W-=eta*(mlp.layer[j].dJ_dW+lambda*mlp.layer[j].W);
			mlp.layer[j].b-=eta*(mlp.layer[j].dJ_db+lambda*mlp.layer[j].b);
		}
	}
}

void c_mlp_classifier::init(std::vector<int> N, int LOSS_FUNC_TYPE)
{
	this->N.resize(N.size());
	this->LOSS_FUNC_TYPE=LOSS_FUNC_TYPE;

	for (int i=0;i<N.size(); i++)
		this->N[i]=N[i];
	
	N_in=N[0];
	N_out=N[N.size()-1];
	
	mlp.init(N,LOSS_FUNC_TYPE);
	sf.init(N_out);
	
//	x.resize(N_in);
	y.resize(N_out);
	
	
}

VectorXd c_mlp_classifier::f(VectorXd x)
{
	y=sf.f(mlp.f(x));
	return y;
}

double c_mlp_classifier::J(VectorXd x, VectorXd y)
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

double c_mlp_classifier::J(std::vector<VectorXd> X, std::vector<VectorXd> Y)
{
	double J=0.;
	
	for(int i=0;i<X.size(); i++)
		J+=this->J(X[i],Y[i]);
	
	return J;
}

void c_mlp_classifier::dJ(VectorXd x, VectorXd y)
{
	VectorXd e, fx, mlp_fx;
	e.resize(y.size());
	
	mlp_fx=mlp.f(x);
	fx=f(x);
	
	switch (LOSS_FUNC_TYPE)
	{
		case LOSS_FUNC_QUAD:		
		e=sf.df(mlp_fx)*(fx-y);
		break;
		
		case LOSS_FUNC_CROSS_ENTROPY:
		for (int i=0; i<y.size(); i++)
		{
			e(i)=-1./fx(i)*y(i)+1./(1.-fx(i))*(1.-y(i));
		}
		e=sf.df(mlp_fx)*e;
		break;
	}
	
	mlp.dJ(x,e);
}

void c_mlp_classifier::dJ_check()
{
	double J, del_J, del_J_approx, eps;
	MatrixXd dW;
	VectorXd db, x, y;
	
	x.resize(N_in);
	y.resize(N_out);
	dW.resize(mlp.layer[0].N_out,mlp.layer[0].N_in);
	db.resize(mlp.layer[0].N_out);

	x.setRandom();
	y.setRandom();
	dW.setRandom();
	db.setRandom();
	eps=1.e-3;

	dW=dW*eps;
	db=db*eps;
	
	J=this->J(x,y);
	this->dJ(x,y);
	
	mlp.layer[0].W+=dW;
	del_J=this->J(x,y)-J;
	
	del_J_approx=(dW.transpose()*mlp.layer[0].dJ_dW).trace();

	cout << "c_mlp_classifier::dJ_check" << endl;
	cout << "dW" << endl;
	cout << "del_J_exact  = " << del_J << endl;
	cout << "del_J_approx = " << del_J_approx << endl;
	
	mlp.layer[0].W-=dW;
	mlp.layer[0].b+=db;
	del_J=this->J(x,y)-J;	
	del_J_approx=db.transpose()*mlp.layer[0].dJ_db;
	
	cout << "db" << endl;
	cout << "del_J_exact  = " << del_J << endl;
	cout << "del_J_approx = " << del_J_approx << endl;

	mlp.layer[0].b-=db;		
	
}
 

int main()
{
	std::vector<int> N;
	VectorXd x,y;
	
	c_mlp_classifier mlp_clf;
	
	N = {25, 50, 5};
	mlp_clf.init(N,LOSS_FUNC_CROSS_ENTROPY);
		
/*	x.resize(N[0]);
	x.setRandom();
	y.resize(N[2]);
	y.setRandom();
	
	mlp_clf.sf.df_check();	
	mlp_clf.mlp.dJ_check();
	mlp_clf.dJ_check();
	mlp_clf.dJ_check();
*/


		
	mlp_clf.init_data(5);

	mlp_clf.X_data[0] <<	
		0, 1, 1, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 1, 0, 0,
		0, 1, 1, 1, 0;

	mlp_clf.X_data[1] <<	
		1, 1, 1, 1, 0,
		0, 0, 0, 0, 1,
		0, 1, 1, 1, 0,
		1, 0, 0, 0, 0,
		1, 1, 1, 1, 1;

	mlp_clf.X_data[2] <<	
		1, 1, 1, 1, 0,
		0, 0, 0, 0, 1,
		0, 1, 1, 1, 0,
		0, 0, 0, 0, 1,
		1, 1, 1, 1, 0;

	mlp_clf.X_data[3] <<	
		0, 0, 0, 1, 0,
		0, 0, 1, 1, 0,
		0, 1, 0, 1, 0,
		1, 1, 1, 1, 1,
		0, 0, 0, 1, 0;		
		
	mlp_clf.X_data[4] <<	
		1, 1, 1, 1, 1,
		1, 0, 0, 0, 0,
		1, 1, 1, 1, 0,
		0, 0, 0, 0, 1,
		1, 1, 1, 1, 0;		
	
	mlp_clf.Y_data[0] << 1, 0, 0, 0, 0;
	mlp_clf.Y_data[1] << 0, 1, 0, 0, 0;
	mlp_clf.Y_data[2] << 0, 0, 1, 0, 0;
	mlp_clf.Y_data[3] << 0, 0, 0, 1, 0;
	mlp_clf.Y_data[4] << 0, 0, 0, 0, 1;


	mlp_clf.grad_descent(10000);
	
	cout << endl;
	for (int i=0; i<5; i++)
		cout << mlp_clf.f(mlp_clf.X_data[i]) << endl << endl;
	
/*	for (int i=0; i<mlp_clf.mlp.N_layer; i++)
	{
		cout << "layer " << i << endl;
		cout << mlp_clf.mlp.layer[i].W << endl;
		cout << mlp_clf.mlp.layer[i].b << endl;
		
	}
	*/
}