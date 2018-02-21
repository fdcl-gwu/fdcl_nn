#include "fdcl_layer.h"

void fdcl_layer::init_io(int N_in, int N_out)
{
	this->N_in=N_in;
	this->N_out=N_out;
	
	x.resize(N_in);
	y.resize(N_out);
	e.resize(N_out);

	x.setZero();
	y.setZero();
	e.setZero();
}
