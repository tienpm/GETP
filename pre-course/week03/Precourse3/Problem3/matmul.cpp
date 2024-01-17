#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Naive Matrix Multiplication
// C[MxN] = A[MxK] @ B[KxN]
// Do not remove or modify this definitions 
void matmul_c(float* A, float* B, float* C, int M, int N, int K){

  for (int i = 0 ; i < M ; ++i)
  {
    for (int k = 0 ; k < K ; ++k)
    {
      float tmp = A[i * K + k];
      for (int j = 0 ; j < N ; ++j)
      {
        C[i * N + j] += tmp * B[k * N + j];
      }
    }
  }
}

/*
Write your code here
*/
// ----------------
// Python interface
// ----------------
py::object py_matmul(py::array_t<float> arr1, py::array_t<float> arr2, 
                     py::array_t<float> &arr3, int M, int N, int K) {
    py::buffer_info buf1 = arr1.request(), buf2 = arr2.request(), buf3 = arr3.request();
    if (buf1.ndim != 1 || buf2.ndim != 1 || buf3.ndim != 1)
        throw std::runtime_error("Number dimension must be one");

    if (buf1.size != buf2.size)
        throw std::runtime_error("Input shape do not match");

    float *ptrA = static_cast<float*>(buf1.ptr);
    float *ptrB = static_cast<float*>(buf2.ptr);
    float *ptrC = static_cast<float*>(buf3.ptr);
    matmul_c(ptrA, ptrB, ptrC, M, N, K);

    return py::cast<py::none>(Py_None);
}

PYBIND11_MODULE(matmul_c, m) {
    m.doc() = "A matmul_c module create by pybind11"; // module doc string
    m.attr("__version__") = "0.0.1";    // module version
    m.def("matmul_c_interface", &py_matmul, "The matrix multiplication operator");
}
