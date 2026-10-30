#include "SQEM.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <Eigen/Dense>
#include <tuple>
namespace py = pybind11;
using namespace std;
using vec3 = Eigen::Vector3d;

// double (SQEM::*minimize1)(vec3 &, double &, const vec3 &, const vec3 &, double) const = &SQEM::minimize<vec3, double>;
// double (SQEM::*minimize2)(vec3 &, double &, const vec3 &, const vec3 &) const = &SQEM::minimize<vec3, double>;
// double (PySQEM::*minimize1)(const vec3 &, const vec3 &, double) = &PySQEM::minimize;

// double (PySQEM::*minimize2)(const vec3 &, const vec3 &) = &PySQEM::minimize;


struct PySQEM : public SQEM {

    PySQEM() : SQEM() {}
    PySQEM(const SQEM &s) : SQEM(s) {}
    PySQEM(const vec3 &a, const vec3 &b) : SQEM(a, b) {}
    PySQEM operator + (const PySQEM &other) const {
        return PySQEM(SQEM::operator+(other));
    }
    PySQEM operator * (double s) const {
        return PySQEM(SQEM::operator*(s));
    }

    PySQEM & operator += (const PySQEM &other) {
        SQEM::operator+=(other);
        return *this;
    }

    PySQEM & operator *= (double s) {
        SQEM::operator*=(s);
        return *this;
    }

    tuple<vec3, double> minimize1(const vec3 &a, const vec3 &b, double r_bound) {
        vec3 c;
        double r;
        SQEM::minimize(c, r, a, b, r_bound);
        return {c, r};
    }
    tuple<vec3, double> minimize2(const vec3 &a, const vec3 &b) {
        vec3 c;
        double r;
        SQEM::minimize(c, r, a, b);
        return {c, r};
    }
};
PYBIND11_MODULE(sqem, m)
{
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: shaysweep

        .. autosummary::
           :toctree: _generate


    )pbdoc";

    py::class_<PySQEM>(m, "Sqem")
        .def(py::init<>())
        .def(py::init<vec3, vec3>())
        .def("set_zero", &PySQEM::setZero)
        .def("set_from_plane", &PySQEM::setFromPlan<vec3>)
        .def(py::self + py::self)
        .def(py::self * double())
        .def(py::self += py::self)
        .def(py::self *= double())
        .def("minimize", &PySQEM::minimize1)
        .def("minimize", &PySQEM::minimize2);


}