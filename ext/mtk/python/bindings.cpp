#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <tuple>
#include "common.cuh"
#include "medials/constraints.h"
#include "medials/distance.h"
#include "medials/collision_detector.h"
#include "medials/ps_addon.h"
#include "medials/ccd.h"
namespace py = pybind11;
using namespace std;
using namespace MIPC;
using vec2f = Eigen::Vector<float, 2>;
using vec3f = Eigen::Vector<float, 3>;
using vec4f = Eigen::Vector<float, 4>; 
using vec12f = Eigen::Vector<float, 12>;
using vec2d = Eigen::Vector<double, 2>;
using vec3d = Eigen::Vector<double, 3>;
using vec4d = Eigen::Vector<double, 4>; 
using vec12d = Eigen::Vector<double, 12>;


using mat22f = Eigen::Matrix<float, 2, 2>;
using mat33f = Eigen::Matrix<float, 3, 3>;
using mat34f = Eigen::Matrix<float, 3, 4>;
using mat44f = Eigen::Matrix<float, 4, 4>;
using mat12f = Eigen::Matrix<float, 12, 12>;

using mat22d = Eigen::Matrix<double, 2, 2>;
using mat33d = Eigen::Matrix<double, 3, 3>;
using mat34d = Eigen::Matrix<double, 3, 4>;
using mat44d = Eigen::Matrix<double, 4, 4>;
using mat12d = Eigen::Matrix<double, 12, 12>;
PYBIND11_MODULE(mtk, m)
{
    m.doc() = "medial ipc bindings";

    m.def("medial_ccd", &MIPC::medial_ccd,
          py::arg("positions_start"), py::arg("positions_end"),
          py::arg("radii"), py::arg("edges"), py::arg("faces"),
          py::arg("body_ids"), py::arg("static_positions"),
          py::arg("static_radii"), py::arg("static_edges"),
          py::arg("static_faces"), py::arg("tolerance") = 1.0e-8,
          "Continuous collision detection for moving medial primitives");

    py::class_<CollideMedialSphere>(m, "MedialSphere")
        .def(py::init([](vec3d a, vec3d b, double f, int i) {
            return new CollideMedialSphere(
                a.cast<scalar>(), b.cast<scalar>(), f, i);  
        }))
        .def(py::init([](vec3f a, vec3f b, float f, int i) {
            return new CollideMedialSphere(
                a.cast<scalar>(), b.cast<scalar>(), f, i);  
        }));

    py::class_<MipcConeConeConstraint>(m, "ConeConeConstraint")
        .def(py::init<vec4, vec4, vec4, vec4>())
        .def(py::init<CollideMedialSphere, CollideMedialSphere, CollideMedialSphere, CollideMedialSphere>())
        .def("compute_distance", &MipcConeConeConstraint::computeDistance)
        .def("get_distance", &MipcConeConeConstraint::getDistance)
        .def("get_distance_mode", &MipcConeConeConstraint::getDistanceMode)
        .def("get_dist_gh", [](MipcConeConeConstraint &self)
             {
            vec12 g12;
            mat12 h12;
            self.get_dist_gh(g12, h12);
            return py::make_tuple(g12, h12); });

    py::class_<MipcSlabSphereConstraint>(m, "SlabSphereConstraint")
        .def(py::init<vec4, vec4, vec4, vec4>())
        .def(py::init<CollideMedialSphere, CollideMedialSphere, CollideMedialSphere, CollideMedialSphere>())
        .def("compute_distance", &MipcSlabSphereConstraint::computeDistance)
        .def("get_distance", &MipcSlabSphereConstraint::getDistance)
        .def("get_distance_mode", &MipcSlabSphereConstraint::getDistanceMode)
        .def("get_dist_gh", [](MipcSlabSphereConstraint &self)
             {
            vec12 g12;
            mat12 h12;
            self.get_dist_gh(g12, h12);
            return py::make_tuple(g12, h12); });

    m.def("cone_geometry", [&](const Mat &V, const Vec &R, const Eigen::MatrixXi &E)
          {
        Mat V_ret;
        Eigen::MatrixXi F_ret;
        cone_points(V, R, E, V_ret, F_ret);
        return py::make_tuple(V_ret, F_ret); }, "generate cone points and faces");

    py::class_<MedialCollisionDetector>(m, "MedialCollider")
        .def(py::init<int, const Eigen::Matrix<scalar, -1, 3> &, const Vec &, const Eigen::Matrix<int, -1, 2> &, const Eigen::Matrix<int, -1, 3> &, scalar>())
        .def("collision_set", &MedialCollisionDetector::collision_set)
        .def("refit", &MedialCollisionDetector::refit)
        .def("collision_set_cc", &MedialCollisionDetector::collision_set_cc)
        .def("collision_set_ss", &MedialCollisionDetector::collision_set_ss)
        .def("get_cc_set", &MedialCollisionDetector::get_cc_set)
        .def("get_ss_set", &MedialCollisionDetector::get_ss_set)
        .def("analyze", [](MedialCollisionDetector &self)
             {
            Vec g;
            Eigen::SparseMatrix<scalar> H;
            std::vector<int> idx_ret;
            self.analyze(g, H, idx_ret);
            return py::make_tuple(g, H, idx_ret); })
        .def("analyze_ss", [](MedialCollisionDetector &self)
             {
            Vec g;
            Mat H;
            std::vector<int> idx_ret;
            self.analyze_ss(g, H, idx_ret);
            return py::make_tuple(g, H, idx_ret); })
        .def("analyze_cc", [](MedialCollisionDetector &self)
             {
            Vec g;
            Mat H;
            std::vector<int> idx_ret;
            self.analyze_cc(g, H, idx_ret);
            return py::make_tuple(g, H, idx_ret); })
        .def("energy", &MedialCollisionDetector::energy);
}
