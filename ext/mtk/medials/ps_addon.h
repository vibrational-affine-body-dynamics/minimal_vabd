#pragma once 
#include "common.cuh"
void cone_points(const Mat &V, const Vec &R, const Eigen::MatrixXi &E, Mat &V_ret, Eigen::MatrixXi &F_ret);
