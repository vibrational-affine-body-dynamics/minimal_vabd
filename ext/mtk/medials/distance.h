#pragma once
#ifndef COLLISION_DETECTION_ON_MEDIAL_MESH_H
#define COLLISION_DETECTION_ON_MEDIAL_MESH_H
// #include "DataCore.h"
// #include "Commom\PolynomialSolver.h"
// #include <queue>
#include "common.cuh"

namespace CDMM
{
#define CHECK_MEDIAL_PRIMITIVE_VALID
#define CDMM_NORROW 1e-8
#define CDMM_NewtonSolverPrecision 1e-15

	// using namespace PolynoimalSolver;
	// typedef ValidRange RootRange;

	// scalar valueOfQuadraticSurface2D(const scalar x, const scalar y, const scalar A, const scalar B, const scalar C, const scalar D, const scalar E, const scalar F);
#define NO_COLLIDE 0
#define INTERSECTION 1
#define PENETRATION 2

	__func__ scalar valueOfQuadraticSurface2D(const scalar x, const scalar y, const scalar A, const scalar B, const scalar C, const scalar D, const scalar E, const scalar F)
	{
		return A * x * x + B * x * y + C * y * y + D * x + E * y + F;
	}
}

#endif