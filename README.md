# FDCL_NN

Neural network library based on the following formulation:

* [A. Caterini and Dong Eui Chang, "A Novel Representation of Neural Networks," arXiv:1610.01549](https://arxiv.org/abs/1610.01549)

## Installatin

### Dependency

This library is based on the [Eigen library](http://eigen.tuxfamily.org/index.php?title=Main_Page) for linear algebra.

### Compile

To run the example provided in the package, run

```
make
./test_fdcl_nn
```

## Supported Layer

### Multilayer Perceptron Layer

The output of this layer is computed by 

<img src"https://latex.codecogs.com/gif.latex?y=f(Wx&plus;b)" />
