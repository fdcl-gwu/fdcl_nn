#include "fdcl_nn.h"

using namespace std;
using namespace Eigen;

int main()
{
	VectorXd x,y;
	fdcl_nn nn;
	std::vector<int> N, type_layer;
	
	N={25, 15, 5};
	type_layer={LAYER_PC, LAYER_SF};
	nn.init(N,type_layer);

	nn.dJ_check();

	nn.init_data(5);

	nn.X_data[0] <<	
		0, 1, 1, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 1, 0, 0,
		0, 1, 1, 1, 0;

	nn.X_data[1] <<	
		1, 1, 1, 1, 0,
		0, 0, 0, 0, 1,
		0, 1, 1, 1, 0,
		1, 0, 0, 0, 0,
		1, 1, 1, 1, 1;

	nn.X_data[2] <<	
		1, 1, 1, 1, 0,
		0, 0, 0, 0, 1,
		0, 1, 1, 1, 0,
		0, 0, 0, 0, 1,
		1, 1, 1, 1, 0;

	nn.X_data[3] <<	
		0, 0, 0, 1, 0,
		0, 0, 1, 1, 0,
		0, 1, 0, 1, 0,
		1, 1, 1, 1, 1,
		0, 0, 0, 1, 0;		
		
	nn.X_data[4] <<	
		1, 1, 1, 1, 1,
		1, 0, 0, 0, 0,
		1, 1, 1, 1, 0,
		0, 0, 0, 0, 1,
		1, 1, 1, 1, 0;		
	
	nn.Y_data[0] << 1, 0, 0, 0, 0;
	nn.Y_data[1] << 0, 1, 0, 0, 0;
	nn.Y_data[2] << 0, 0, 1, 0, 0;
	nn.Y_data[3] << 0, 0, 0, 1, 0;
	nn.Y_data[4] << 0, 0, 0, 0, 1;

	nn.grad_descent(15000);
	
	cout << endl;
	for (int i=0; i<5; i++)
		cout << nn.f(nn.X_data[i]) << endl << endl;


	return 0;
}