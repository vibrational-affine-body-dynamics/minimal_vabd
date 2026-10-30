#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "solver.h"
using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(dxslv, m)
{
    m.doc() = "direct solver (cudss) python bindings";

    py::class_<CUSolver>(m, "CUSolver")
        .def(py::init<const Veci &, const Veci &, const Vec &>())
        .def(py::init<const Eigen::SparseMatrix<double> &>())
        .def("analyze_pattern", &CUSolver::analyze_pattern)
        .def("factorize", &CUSolver::factorize)
        .def("solve", &CUSolver::solve)
        .def("solve_cuda", &CUSolver::solve_dev)
        .def("refactorize", &CUSolver::refactorize)
        .def("refactor_cuda", &CUSolver::refactorize_dev);

    py::class_<CUSolverDevice>(m, "CUSolverDevice")
        .def(py::init<uintptr_t, uintptr_t, uintptr_t, int, int>(),
             py::arg("outers_ptr"), py::arg("indices_ptr"), py::arg("values_ptr"),
             py::arg("n"), py::arg("nnz"))
        .def("analyze_pattern", &CUSolverDevice::analyze_pattern)
        .def("factorize", &CUSolverDevice::factorize)
        .def("solve", &CUSolverDevice::solve, py::arg("b_ptr"), py::arg("x_ptr"))
        .def("refactorize", &CUSolverDevice::refactorize, py::arg("new_values_ptr"));

    py::class_<CUSolverDeviceF>(m, "CUSolverDeviceF")
        .def(py::init<uintptr_t, uintptr_t, uintptr_t, int, int>(),
             py::arg("outers_ptr"), py::arg("indices_ptr"), py::arg("values_ptr"),
             py::arg("n"), py::arg("nnz"))
        .def("analyze_pattern", &CUSolverDeviceF::analyze_pattern)
        .def("factorize", &CUSolverDeviceF::factorize)
        .def("solve", &CUSolverDeviceF::solve, py::arg("b_ptr"), py::arg("x_ptr"))
        .def("refactorize", &CUSolverDeviceF::refactorize, py::arg("new_values_ptr"));

    // A descriptive alias for callers that prefer GPU terminology.
    m.attr("CUSolverGPU") = m.attr("CUSolverDevice");
}
