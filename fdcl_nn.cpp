#include "fdcl_nn.h"

int fdcl_nn::LOSS_FUNC_TYPE=1;

fdcl_nn::fdcl_nn(std::vector<int> N, std::vector<int> layer_type)
{
	init(N,layer_type);
}
void fdcl_nn::init(std::vector<int> N, std::vector<int> layer_type)
{
	this->N=N;
	this->layer_type=layer_type;

	N_layer=N.size()-1;
	N_in=N[0];
	N_out=N[N_layer];
	
	for (int i=0;i<N_layer;i++)
	{
		switch(layer_type[i])
		{
			case LAYER_PC:
				layer.emplace_back(new fdcl_mlp_layer(N[i],N[i+1]));
			break;

			case LAYER_SF:
				layer.emplace_back(new fdcl_softmax_layer(N[i],N[i+1]));
			break;

		}		
	}
}

VectorXd fdcl_nn::f(VectorXd x)
{
	layer[0]->y=layer[0]->f(x);
	for(int i=0;i<N_layer-1;i++)
	{
		layer[i+1]->y=layer[i+1]->f(layer[i]->y);
	}
	
	return layer[N_layer-1]->y;
}

double fdcl_nn::J(VectorXd x, VectorXd y)
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
			for(int i=0; i< N_out; i++)	
				J+=-y(i)*log(fx(i))-(1.-y(i))*log(1-fx(i));
		break;
		
	}
	return J;
}

double fdcl_nn::J(std::vector<VectorXd> X, std::vector<VectorXd> Y)
{
	double J=0.;
	
	for(int i=0;i<X.size(); i++)
		J+=this->J(X[i],Y[i]);
	
	return J;
}

void fdcl_nn::compute_dJ_dtheta(VectorXd x, VectorXd y)
{
	layer[N_layer-1]->e=this->back_prop(x,y);
	layer[N_layer-1]->compute_dJ_dtheta(layer[N_layer-1]->e);
	
	for(int i=N_layer-2; i>=0; i--)
	{
		layer[i]->e=layer[i+1]->back_prop(layer[i+1]->e);
		layer[i]->compute_dJ_dtheta(layer[i]->e);
	}	
}

VectorXd fdcl_nn::back_prop(VectorXd x, VectorXd y)
{
	VectorXd e, fx;
	e.resize(y.size());
	
	fx=f(x);
	switch (LOSS_FUNC_TYPE)
	{
		case LOSS_FUNC_QUAD :		
		e=f(x)-y;
		break;
		
		case LOSS_FUNC_CROSS_ENTROPY :
		for (int i=0; i<y.size(); i++)
		{
			e(i)=-1./fx(i)*y(i)+1./(1.-fx(i))*(1.-y(i));
		}
		
		break;
	}
	
	return e;
}

void fdcl_nn::dJ_check()
{
	double J, del_J, del_J_approx, eps;
	std::vector<MatrixXd> del_theta;
	VectorXd x, y;
	int N_theta;
	
	x.resize(N_in);
	y.resize(N_out);
	x.setRandom();
	y.setRandom();
	eps=1.e-5;
	
	J=this->J(x,y);
	this->compute_dJ_dtheta(x,y);
	
	// perturb theta
	N_theta=layer[0]->theta.size();
	del_theta.resize(N_theta);
	for (int i=0; i<N_theta; i++)
	{
		del_theta[i].resize(layer[0]->theta[i].rows(),layer[0]->theta[i].cols());
		del_theta[i].setRandom();
		del_theta[i]*=eps;
	}

	for (int i=0; i<N_theta; i++)
		layer[0]->theta[i]+=del_theta[i];
	
	// perturbed J
	del_J=this->J(x,y)-J;
	
	// compute by gradient
	del_J_approx=0.;
	for (int i=0; i<N_theta; i++)
		del_J_approx+=(del_theta[i].transpose()*layer[0]->dJ_dtheta[i]).trace();
			
	cout << "fdcl_nn::dJ_check" << endl;
	cout << "del_J_exact  = " << del_J << endl;
	cout << "del_J_approx = " << del_J_approx << endl;
	
	// return to the original theta 
	for (int i=0; i<N_theta; i++)
		layer[0]->theta[i]-=del_theta[i];
	
}

void fdcl_nn::init_data(int N_data)
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

void fdcl_nn::grad_descent(int N_epoch)
{
	for(int i=0; i < N_epoch; i++)
	{
		grad_descent();
		if (i % 100 == 0)
			cout << "epoch = " << i << "  J = " << J(X_data,Y_data) << endl;
	}
}

void fdcl_nn::grad_descent()
{
	int i, j, k, N_theta;
	double eta=0.9; // learning rate
	double lambda=0.; // quadratic penalty on weight
	
	for(i=0; i< N_data; i++)
	{
		compute_dJ_dtheta(X_data[i], Y_data[i]);
		for (j=0; j<N_layer; j++)
		{
			N_theta=layer[j]->theta.size();
			for (k=0; k< N_theta; k++)
				layer[j]->theta[k]-=eta*(layer[j]->dJ_dtheta[k]+lambda*layer[j]->theta[k]);
		}
	}
}
