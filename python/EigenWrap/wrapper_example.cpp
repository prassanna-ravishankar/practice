#include <Python.h>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace boost::python;

PyObject* eig(PyArrayObject* numpy_array)
{
    npy_intp *shape = PyArray_DIMS(numpy_array);
    npy_intp n = shape[0];
    Map<Matrix<double, Dynamic, Dynamic, RowMajor> > mat(static_cast<double *>(PyArray_DATA(numpy_array)),shape[0], shape[1]);
    
    SelfAdjointEigenSolver<MatrixXd> solver(mat);
    PyObject *val = PyArray_SimpleNew(1, &n, NPY_DOUBLE);
    PyObject *vec = PyArray_SimpleNew(2, shape, NPY_DOUBLE);
    Map<VectorXd>(static_cast<double *>(PyArray_DATA(val)), mat.rows()) = solver.eigenvalues();
    Map<Matrix<double, Dynamic, Dynamic, RowMajor> >(static_cast<double *>(PyArray_DATA(vec)), n,n) = solver.eigenvectors();

    PyObject * tuple = PyTuple_New(2);
    PyTuple_SetItem(tuple, 0, PyArray_Return((PyArrayObject*)(val)));
    PyTuple_SetItem(tuple, 1, PyArray_Return((PyArrayObject *)(vec)));
    
    return tuple;
}

void* extract_pyarray(PyObject* x)
{
	return x;
}


BOOST_PYTHON_MODULE(eigen)
{
  boost::python::converter::registry::insert(&extract_pyarray, boost::python::type_id<PyArrayObject>());
  def("eig", eig);
  import_array();
}
