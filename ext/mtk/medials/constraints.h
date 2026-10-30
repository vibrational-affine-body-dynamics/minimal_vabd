#pragma once
#ifndef MIPC_Constraint_H
#define MIPC_Constraint_H
#define CHECK_MEDIAL_PRIMITIVE_VALID
#include "common.cuh"
#include "medials/distance.h"

namespace MIPC
{
	using namespace CDMM;

	class CollideMedialSphere
	{
	public:
		CollideMedialSphere(vec3 c, scalar r) : center(c), radius(r) {}
		CollideMedialSphere(vec3 c, vec3 x_rst, scalar r, int idx) : center(c), x_rest(x_rst), radius(r), index(idx)
		{
		}
		CollideMedialSphere() {}
		vec3 center, x_rest;
		scalar radius;
		int index = 0;
		CollideMedialSphere(vec4 cr) : center(cr.head(3)), radius(cr(3)) {}
	};

	class MipcConstraint
	{
	public:
		enum CollisionType
		{
			CC = 0,
			SS = 1
		};

		int info;
		CollisionType collisionType;
		CollideMedialSphere spheres[4];
		scalar distance;

		//
		vec3 sC1, sC2, sC3;
		vec3 sV1, sV2, sV3;
		scalar sR1, sR2, sR3;
		scalar A, B, C, D, E, F;
		scalar delta;		// 4AC - B^2
		scalar alpha, beta; //

		MipcConstraint(CollideMedialSphere s0, CollideMedialSphere s1, CollideMedialSphere s2, CollideMedialSphere s3)
		{
			spheres[0] = s0;
			spheres[1] = s1;
			spheres[2] = s2;
			spheres[3] = s3;
		}

		MipcConstraint(vec4 s0, vec4 s1, vec4 s2, vec4 s3)
		{
			spheres[0] = CollideMedialSphere(s0);
			spheres[1] = CollideMedialSphere(s1);
			spheres[2] = CollideMedialSphere(s2);
			spheres[3] = CollideMedialSphere(s3);
		}

		CollisionType getCollisionType() { return collisionType; }
		virtual scalar getDistance() { return distance; }

		virtual scalar computeDistance() = 0;
		virtual void get_dist_gh(vec12 &g12, mat12 &h12) = 0;
	};

	class MipcConeConeConstraint : public MipcConstraint
	{
	public:
		enum DistanceMode
		{
			TWO_ENDPOINTS = 0,
			ALPHA_ZERO = 1,
			ALPHA_ONE = 2,
			BETA_ZERO = 3,
			BETA_ONE = 4,
			ALPHA_BETA = 5
		};
		DistanceMode distanceMode;

		MipcConeConeConstraint(vec4 s0, vec4 s1, vec4 s2, vec4 s3) : MipcConstraint(s0, s1, s2, s3)
		{
			collisionType = CollisionType::CC;
			computeDistance();
		}

		MipcConeConeConstraint(CollideMedialSphere s0, CollideMedialSphere s1, CollideMedialSphere s2, CollideMedialSphere s3) : MipcConstraint(s0, s1, s2, s3)
		{
			collisionType = CollisionType::CC;
			computeDistance();
		}

		scalar getDistance() { return distance; }
		int getDistanceMode() { return int(distanceMode); }
		virtual scalar computeDistance();

		virtual void get_dist_gh(vec12 &g12, mat12 &h12);

		inline void diff_F_x(vec12 &diff);
		inline void endPointsHessian(mat12 &hessian);
		inline void alphaIsZeroHessian(mat12 &hessian);
		inline void alphaIsZeroHessian1(mat12 &hessian);
		inline void alphaIsOneHessian(mat12 &hessian);
		inline void betaIsZeroHessian(mat12 &hessian);
		inline void betaIsZeroHessian1(mat12 &hessian);
		inline void betaIsOneHessian1(mat12 &hessian);
		inline void betaIsOneHessian(mat12 &hessian);
		inline void alphaBetaHessian(mat12 &hessian);
	};

	class MipcSlabSphereConstraint : public MipcConstraint
	{
	public:
		enum DistanceMode
		{
			TWO_ENDPOINTS = 0,
			ALPHA_ZERO = 1,
			BETA_ZERO = 2,
			ALPHA_BETA_ONE = 3,
			ALPHA_BETA = 4
		};
		DistanceMode distanceMode;

		MipcSlabSphereConstraint(vec4 s0, vec4 s1, vec4 s2, vec4 s3) : MipcConstraint(s0, s1, s2, s3)
		{
			collisionType = CollisionType::SS;
			computeDistance();
		}
		MipcSlabSphereConstraint(CollideMedialSphere s0, CollideMedialSphere s1, CollideMedialSphere s2, CollideMedialSphere s3) : MipcConstraint(s0, s1, s2, s3)
		{
			collisionType = CollisionType::SS;
			computeDistance();
		}
		int getDistanceMode() { return int(distanceMode); }

		virtual scalar computeDistance();
		scalar getDistance() { return distance; }

		virtual void get_dist_gh(vec12 &g12, mat12 &h12);
		inline void diff_F_x(vec12 &diff);
		inline void endPointsHessian(mat12 &hessian);
		inline void alphaIsZeroHessian(mat12 &hessian);
		inline void betaIsZeroHessian(mat12 &hessian);
		inline void alphaBetaPlusOneHessian(mat12 &hessian);
		inline void alphaBetaHessian(mat12 &hessian);
	};

}

#endif