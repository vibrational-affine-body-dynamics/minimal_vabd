#include "medials/constraints.h"
#include <assert.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace MIPC
{
	void makePD(mat12 &symMtr)
	{
		Eigen::SelfAdjointEigenSolver<mat12> eigenSolver(symMtr);
		if (eigenSolver.eigenvalues()[0] >= 0)
		{
			return;
		}
		int size = symMtr.rows();
		mat12 D = eigenSolver.eigenvalues().asDiagonal();

		for (int i = 0; i < size; ++i)
		{
			if (D.diagonal()[i] < 0)
				D.diagonal()[i] = 0;
			else
				break;
		}

		symMtr = eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
	}

	scalar MipcConeConeConstraint::computeDistance()
	{
		sC1 = spheres[0].center - spheres[1].center;
		sC2 = spheres[3].center - spheres[2].center;
		sC3 = spheres[1].center - spheres[3].center;

		sR1 = spheres[0].radius - spheres[1].radius;
		sR2 = spheres[2].radius - spheres[3].radius;
		sR3 = spheres[1].radius + spheres[3].radius;

		A = sC1.dot(sC1) - sR1 * sR1;
		B = 2.0 * (sC1.dot(sC2) - sR1 * sR2);
		C = sC2.dot(sC2) - sR2 * sR2;
		D = 2.0 * (sC1.dot(sC3) - sR1 * sR3);
		E = 2.0 * (sC2.dot(sC3) - sR2 * sR3);
		F = sC3.dot(sC3) - sR3 * sR3;

		delta = 4 * A * C - B * B;

		scalar temp_alpha, temp_beta;
		scalar temp_dist;

		distanceMode = TWO_ENDPOINTS;
		alpha = 0.0, beta = 0.0;
		distance = valueOfQuadraticSurface2D(alpha, beta, A, B, C, D, E, F);

		temp_alpha = 1.0, temp_beta = 0.0;
		temp_dist = valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
		if (distance > temp_dist)
		{
			distance = temp_dist;
			alpha = temp_alpha;
			beta = temp_beta;
		}

		temp_alpha = 0.0, temp_beta = 1.0;
		temp_dist = valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
		if (distance > temp_dist)
		{
			distance = temp_dist;
			alpha = temp_alpha;
			beta = temp_beta;
		}

		temp_alpha = 1.0, temp_beta = 1.0;
		temp_dist = valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
		if (distance > temp_dist)
		{
			distance = temp_dist;
			alpha = temp_alpha;
			beta = temp_beta;
		}

		temp_alpha = 0.0;
		temp_beta = -E / (2.0 * C);
		if (temp_beta > 0.0 && temp_beta < 1.0)
		{
			temp_dist = valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
			if (distance > temp_dist)
			{
				distance = temp_dist;
				alpha = temp_alpha;
				beta = temp_beta;
				distanceMode = ALPHA_ZERO;
			}
		}

		temp_alpha = 1.0;
		temp_beta = -(B + E) / (2.0 * C);
		if (temp_beta > 0.0 && temp_beta < 1.0)
		{
			temp_dist = valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
			if (distance > temp_dist)
			{
				distance = temp_dist;
				alpha = temp_alpha;
				beta = temp_beta;
				distanceMode = ALPHA_ONE;
			}
		}

		temp_alpha = -D / (2.0 * A);
		temp_beta = 0.0;
		if (temp_alpha > 0.0 && temp_alpha < 1.0)
		{
			temp_dist = valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
			if (distance > temp_dist)
			{
				distance = temp_dist;
				alpha = temp_alpha;
				beta = temp_beta;
				distanceMode = BETA_ZERO;
			}
		}

		temp_alpha = -(B + D) / (2.0 * A);
		temp_beta = 1.0;
		if (temp_alpha > 0.0 && temp_alpha < 1.0)
		{
			temp_dist = valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
			if (distance > temp_dist)
			{
				distance = temp_dist;
				alpha = temp_alpha;
				beta = temp_beta;
				distanceMode = BETA_ONE;
			}
		}

		if (delta != 0.0)
		{
			temp_alpha = (B * E - 2.0 * C * D) / delta;
			temp_beta = (B * D - 2.0 * A * E) / delta;
			if (temp_alpha > 0.0 && temp_alpha < 1.0 && temp_beta > 0.0 && temp_beta < 1.0)
			{
				temp_dist = valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
				if (distance > temp_dist)
				{
					distance = temp_dist;
					alpha = temp_alpha;
					beta = temp_beta;
					distanceMode = ALPHA_BETA;
				}
			}
		}

		// compute cloeset points;
		vec3 cp, cq;
		cp = alpha * spheres[0].center + (1.0 - alpha) * spheres[1].center;
		cq = beta * spheres[2].center + (1.0 - beta) * spheres[3].center;
		scalar rp, rq;
		rp = alpha * spheres[0].radius + +(1.0 - alpha) * spheres[1].radius;
		rq = beta * spheres[2].radius + (1.0 - beta) * spheres[3].radius;

		vec3 dir = cq - cp;
		dir.normalize();

		return distance;
	}

	void MipcConeConeConstraint::get_dist_gh(vec12 &g12, mat12 &h12)
	{
		vec12 distGrad;
		distGrad.setZero();

		mat12 distHessian;
		distHessian.setZero();

		diff_F_x(distGrad);
		// gradient
		//  scalar scale = -1.0 * kappa * barrierGrad;
		g12 = -distGrad;
		// fillOverallGradient(scale, distGrad, gradient);

		// hessian
		switch (distanceMode)
		{
		case TWO_ENDPOINTS:
			endPointsHessian(distHessian);
			break;
		case ALPHA_ZERO:
			alphaIsZeroHessian(distHessian);
			break;
		case ALPHA_ONE:
			alphaIsOneHessian(distHessian);
			break;
		case BETA_ZERO:
			betaIsZeroHessian(distHessian);
			break;
		case BETA_ONE:
			betaIsOneHessian(distHessian);
			break;
		case ALPHA_BETA:
			alphaBetaHessian(distHessian);
			break;
		default:
			break;
		};

		h12 = -distHessian;
		makePD(h12);
	}

	void MipcConeConeConstraint::diff_F_x(vec12 &diff)
	{
		vec3 v = 2.0 * (alpha * sC1 + beta * sC2 + sC3);
		diff[0] = alpha * v.data()[0];
		diff[1] = alpha * v.data()[1];
		diff[2] = alpha * v.data()[2];

		diff[3] = (1.0 - alpha) * v.data()[0];
		diff[4] = (1.0 - alpha) * v.data()[1];
		diff[5] = (1.0 - alpha) * v.data()[2];

		diff[6] = -beta * v.data()[0];
		diff[7] = -beta * v.data()[1];
		diff[8] = -beta * v.data()[2];

		diff[9] = -(1.0 - beta) * v.data()[0];
		diff[10] = -(1.0 - beta) * v.data()[1];
		diff[11] = -(1.0 - beta) * v.data()[2];
	}

	void MipcConeConeConstraint::endPointsHessian(mat12 &hessian)
	{
		// alpha and beta is constant
		scalar C1x = sC1.data()[0];
		scalar C1y = sC1.data()[1];
		scalar C1z = sC1.data()[2];
		scalar C2x = sC2.data()[0];
		scalar C2y = sC2.data()[1];
		scalar C2z = sC2.data()[2];
		scalar C3x = sC3.data()[0];
		scalar C3y = sC3.data()[1];
		scalar C3z = sC3.data()[2];

		scalar vx = 2.0 * (alpha * C1x + beta * C2x + C3x);
		scalar vy = 2.0 * (alpha * C1y + beta * C2y + C3y);
		scalar vz = 2.0 * (alpha * C1z + beta * C2z + C3z);

		scalar dvxdc11_x = 2.0 * alpha;
		scalar dvxdc11_y = 0;
		scalar dvxdc11_z = 0;

		scalar dvxdc12_x = 2.0 * (1 - alpha);
		scalar dvxdc12_y = 0;
		scalar dvxdc12_z = 0;

		scalar dvxdc21_x = 2.0 * -beta;
		scalar dvxdc21_y = 0;
		scalar dvxdc21_z = 0;

		scalar dvxdc22_x = 2.0 * (beta - 1.0);
		scalar dvxdc22_y = 0;
		scalar dvxdc22_z = 0;
		//
		scalar dvydc11_x = 0;
		scalar dvydc11_y = 2.0 * alpha;
		scalar dvydc11_z = 0;

		scalar dvydc12_x = 0;
		scalar dvydc12_y = 2.0 * (1 - alpha);
		scalar dvydc12_z = 0;

		scalar dvydc21_x = 0;
		scalar dvydc21_y = 2.0 * -beta;
		scalar dvydc21_z = 0;

		scalar dvydc22_x = 0;
		scalar dvydc22_y = 2.0 * (beta - 1.0);
		scalar dvydc22_z = 0;
		//
		scalar dvzdc11_x = 0;
		scalar dvzdc11_y = 0;
		scalar dvzdc11_z = 2.0 * alpha;

		scalar dvzdc12_x = 0;
		scalar dvzdc12_y = 0;
		scalar dvzdc12_z = 2.0 * (1 - alpha);

		scalar dvzdc21_x = 0;
		scalar dvzdc21_y = 0;
		scalar dvzdc21_z = 2.0 * -beta;

		scalar dvzdc22_x = 0;
		scalar dvzdc22_y = 0;
		scalar dvzdc22_z = 2.0 * (beta - 1.0);

		hessian.setZero();
		// f / c11x
		hessian.data()[0] = alpha * dvxdc11_x;
		hessian.data()[1] = alpha * dvydc11_x;
		hessian.data()[2] = alpha * dvzdc11_x;

		hessian.data()[3] = (1 - alpha) * dvxdc11_x;
		hessian.data()[4] = (1 - alpha) * dvydc11_x;
		hessian.data()[5] = (1 - alpha) * dvzdc11_x;

		hessian.data()[6] = -beta * dvxdc11_x;
		hessian.data()[7] = -beta * dvydc11_x;
		hessian.data()[8] = -beta * dvzdc11_x;

		hessian.data()[9] = (beta - 1.0) * dvxdc11_x;
		hessian.data()[10] = (beta - 1.0) * dvydc11_x;
		hessian.data()[11] = (beta - 1.0) * dvzdc11_x;
		// f / c11y
		hessian.data()[12] = alpha * dvxdc11_y;
		hessian.data()[13] = alpha * dvydc11_y;
		hessian.data()[14] = alpha * dvzdc11_y;

		hessian.data()[15] = (1 - alpha) * dvxdc11_y;
		hessian.data()[16] = (1 - alpha) * dvydc11_y;
		hessian.data()[17] = (1 - alpha) * dvzdc11_y;

		hessian.data()[18] = -beta * dvxdc11_y;
		hessian.data()[19] = -beta * dvydc11_y;
		hessian.data()[20] = -beta * dvzdc11_y;

		hessian.data()[21] = (beta - 1.0) * dvxdc11_y;
		hessian.data()[22] = (beta - 1.0) * dvydc11_y;
		hessian.data()[23] = (beta - 1.0) * dvzdc11_y;
		// f / c11z
		hessian.data()[24] = alpha * dvxdc11_z;
		hessian.data()[25] = alpha * dvydc11_z;
		hessian.data()[26] = alpha * dvzdc11_z;

		hessian.data()[27] = (1 - alpha) * dvxdc11_z;
		hessian.data()[28] = (1 - alpha) * dvydc11_z;
		hessian.data()[29] = (1 - alpha) * dvzdc11_z;

		hessian.data()[30] = -beta * dvxdc11_z;
		hessian.data()[31] = -beta * dvydc11_z;
		hessian.data()[32] = -beta * dvzdc11_z;

		hessian.data()[33] = (beta - 1.0) * dvxdc11_z;
		hessian.data()[34] = (beta - 1.0) * dvydc11_z;
		hessian.data()[35] = (beta - 1.0) * dvzdc11_z;

		// f / c12x
		hessian.data()[36] = alpha * dvxdc12_x;
		hessian.data()[37] = alpha * dvydc12_x;
		hessian.data()[38] = alpha * dvzdc12_x;

		hessian.data()[39] = (1 - alpha) * dvxdc12_x;
		hessian.data()[40] = (1 - alpha) * dvydc12_x;
		hessian.data()[41] = (1 - alpha) * dvzdc12_x;

		hessian.data()[42] = -beta * dvxdc12_x;
		hessian.data()[43] = -beta * dvydc12_x;
		hessian.data()[44] = -beta * dvzdc12_x;

		hessian.data()[45] = (beta - 1.0) * dvxdc12_x;
		hessian.data()[46] = (beta - 1.0) * dvydc12_x;
		hessian.data()[47] = (beta - 1.0) * dvzdc12_x;
		// f / c12y
		hessian.data()[48] = alpha * dvxdc12_y;
		hessian.data()[49] = alpha * dvydc12_y;
		hessian.data()[50] = alpha * dvzdc12_y;

		hessian.data()[51] = (1 - alpha) * dvxdc12_y;
		hessian.data()[52] = (1 - alpha) * dvydc12_y;
		hessian.data()[53] = (1 - alpha) * dvzdc12_y;

		hessian.data()[54] = -beta * dvxdc12_y;
		hessian.data()[55] = -beta * dvydc12_y;
		hessian.data()[56] = -beta * dvzdc12_y;

		hessian.data()[57] = (beta - 1.0) * dvxdc12_y;
		hessian.data()[58] = (beta - 1.0) * dvydc12_y;
		hessian.data()[59] = (beta - 1.0) * dvzdc12_y;
		// f / c12z
		hessian.data()[60] = alpha * dvxdc12_z;
		hessian.data()[61] = alpha * dvydc12_z;
		hessian.data()[62] = alpha * dvzdc12_z;

		hessian.data()[63] = (1 - alpha) * dvxdc12_z;
		hessian.data()[64] = (1 - alpha) * dvydc12_z;
		hessian.data()[65] = (1 - alpha) * dvzdc12_z;

		hessian.data()[66] = -beta * dvxdc12_z;
		hessian.data()[67] = -beta * dvydc12_z;
		hessian.data()[68] = -beta * dvzdc12_z;

		hessian.data()[69] = (beta - 1.0) * dvxdc12_z;
		hessian.data()[70] = (beta - 1.0) * dvydc12_z;
		hessian.data()[71] = (beta - 1.0) * dvzdc12_z;

		// f / c21x
		hessian.data()[72] = alpha * dvxdc21_x;
		hessian.data()[73] = alpha * dvydc21_x;
		hessian.data()[74] = alpha * dvzdc21_x;

		hessian.data()[75] = (1 - alpha) * dvxdc21_x;
		hessian.data()[76] = (1 - alpha) * dvydc21_x;
		hessian.data()[77] = (1 - alpha) * dvzdc21_x;

		hessian.data()[78] = -beta * dvxdc21_x;
		hessian.data()[79] = -beta * dvydc21_x;
		hessian.data()[80] = -beta * dvzdc21_x;

		hessian.data()[81] = (beta - 1.0) * dvxdc21_x;
		hessian.data()[82] = (beta - 1.0) * dvydc21_x;
		hessian.data()[83] = (beta - 1.0) * dvzdc21_x;
		// f / c21y
		hessian.data()[84] = alpha * dvxdc21_y;
		hessian.data()[85] = alpha * dvydc21_y;
		hessian.data()[86] = alpha * dvzdc21_y;

		hessian.data()[87] = (1 - alpha) * dvxdc21_y;
		hessian.data()[88] = (1 - alpha) * dvydc21_y;
		hessian.data()[89] = (1 - alpha) * dvzdc21_y;

		hessian.data()[90] = -beta * dvxdc21_y;
		hessian.data()[91] = -beta * dvydc21_y;
		hessian.data()[92] = -beta * dvzdc21_y;

		hessian.data()[93] = (beta - 1.0) * dvxdc21_y;
		hessian.data()[94] = (beta - 1.0) * dvydc21_y;
		hessian.data()[95] = (beta - 1.0) * dvzdc21_y;
		// f / c21z
		hessian.data()[96] = alpha * dvxdc21_z;
		hessian.data()[97] = alpha * dvydc21_z;
		hessian.data()[98] = alpha * dvzdc21_z;

		hessian.data()[99] = (1 - alpha) * dvxdc21_z;
		hessian.data()[100] = (1 - alpha) * dvydc21_z;
		hessian.data()[101] = (1 - alpha) * dvzdc21_z;

		hessian.data()[102] = -beta * dvxdc21_z;
		hessian.data()[103] = -beta * dvydc21_z;
		hessian.data()[104] = -beta * dvzdc21_z;

		hessian.data()[105] = (beta - 1.0) * dvxdc21_z;
		hessian.data()[106] = (beta - 1.0) * dvydc21_z;
		hessian.data()[107] = (beta - 1.0) * dvzdc21_z;

		// f / c22x
		hessian.data()[108] = alpha * dvxdc22_x;
		hessian.data()[109] = alpha * dvydc22_x;
		hessian.data()[110] = alpha * dvzdc22_x;

		hessian.data()[111] = (1 - alpha) * dvxdc22_x;
		hessian.data()[112] = (1 - alpha) * dvydc22_x;
		hessian.data()[113] = (1 - alpha) * dvzdc22_x;

		hessian.data()[114] = -beta * dvxdc22_x;
		hessian.data()[115] = -beta * dvydc22_x;
		hessian.data()[116] = -beta * dvzdc22_x;

		hessian.data()[117] = (beta - 1.0) * dvxdc22_x;
		hessian.data()[118] = (beta - 1.0) * dvydc22_x;
		hessian.data()[119] = (beta - 1.0) * dvzdc22_x;
		// f / c22y
		hessian.data()[120] = alpha * dvxdc22_y;
		hessian.data()[121] = alpha * dvydc22_y;
		hessian.data()[122] = alpha * dvzdc22_y;

		hessian.data()[123] = (1 - alpha) * dvxdc22_y;
		hessian.data()[124] = (1 - alpha) * dvydc22_y;
		hessian.data()[125] = (1 - alpha) * dvzdc22_y;

		hessian.data()[126] = -beta * dvxdc22_y;
		hessian.data()[127] = -beta * dvydc22_y;
		hessian.data()[128] = -beta * dvzdc22_y;

		hessian.data()[129] = (beta - 1.0) * dvxdc22_y;
		hessian.data()[130] = (beta - 1.0) * dvydc22_y;
		hessian.data()[131] = (beta - 1.0) * dvzdc22_y;
		// f / c22z
		hessian.data()[132] = alpha * dvxdc22_z;
		hessian.data()[133] = alpha * dvydc22_z;
		hessian.data()[134] = alpha * dvzdc22_z;

		hessian.data()[135] = (1 - alpha) * dvxdc22_z;
		hessian.data()[136] = (1 - alpha) * dvydc22_z;
		hessian.data()[137] = (1 - alpha) * dvzdc22_z;

		hessian.data()[138] = -beta * dvxdc22_z;
		hessian.data()[139] = -beta * dvydc22_z;
		hessian.data()[140] = -beta * dvzdc22_z;

		hessian.data()[141] = (beta - 1.0) * dvxdc22_z;
		hessian.data()[142] = (beta - 1.0) * dvydc22_z;
		hessian.data()[143] = (beta - 1.0) * dvzdc22_z;
	}

	void MipcConeConeConstraint::alphaIsZeroHessian(mat12 &hessian)
	{
		scalar dAdc11_x = 2.0 * sC1.data()[0];
		scalar dAdc11_y = 2.0 * sC1.data()[1];
		scalar dAdc11_z = 2.0 * sC1.data()[2];
		scalar dAdc12_x = -2.0 * sC1.data()[0];
		scalar dAdc12_y = -2.0 * sC1.data()[1];
		scalar dAdc12_z = -2.0 * sC1.data()[2];

		scalar dBdc11_x = 2.0 * sC2.data()[0];
		scalar dBdc11_y = 2.0 * sC2.data()[1];
		scalar dBdc11_z = 2.0 * sC2.data()[2];
		scalar dBdc12_x = -2.0 * sC2.data()[0];
		scalar dBdc12_y = -2.0 * sC2.data()[1];
		scalar dBdc12_z = -2.0 * sC2.data()[2];
		scalar dBdc21_x = -2.0 * sC1.data()[0];
		scalar dBdc21_y = -2.0 * sC1.data()[1];
		scalar dBdc21_z = -2.0 * sC1.data()[2];
		scalar dBdc22_x = 2.0 * sC1.data()[0];
		scalar dBdc22_y = 2.0 * sC1.data()[1];
		scalar dBdc22_z = 2.0 * sC1.data()[2];

		scalar dCdc21_x = -2.0 * sC2.data()[0];
		scalar dCdc21_y = -2.0 * sC2.data()[1];
		scalar dCdc21_z = -2.0 * sC2.data()[2];
		scalar dCdc22_x = 2.0 * sC2.data()[0];
		scalar dCdc22_y = 2.0 * sC2.data()[1];
		scalar dCdc22_z = 2.0 * sC2.data()[2];

		scalar dDdc11_x = 2.0 * sC3.data()[0];
		scalar dDdc11_y = 2.0 * sC3.data()[1];
		scalar dDdc11_z = 2.0 * sC3.data()[2];
		scalar dDdc12_x = 2.0 * (sC1.data()[0] - sC3.data()[0]);
		scalar dDdc12_y = 2.0 * (sC1.data()[1] - sC3.data()[1]);
		scalar dDdc12_z = 2.0 * (sC1.data()[2] - sC3.data()[2]);
		scalar dDdc22_x = -2.0 * sC1.data()[0];
		scalar dDdc22_y = -2.0 * sC1.data()[1];
		scalar dDdc22_z = -2.0 * sC1.data()[2];

		scalar dEdc12_x = 2.0 * sC2.data()[0];
		scalar dEdc12_y = 2.0 * sC2.data()[1];
		scalar dEdc12_z = 2.0 * sC2.data()[2];
		scalar dEdc21_x = -2.0 * sC3.data()[0];
		scalar dEdc21_y = -2.0 * sC3.data()[1];
		scalar dEdc21_z = -2.0 * sC3.data()[2];
		scalar dEdc22_x = 2.0 * (sC3.data()[0] - sC2.data()[0]);
		scalar dEdc22_y = 2.0 * (sC3.data()[1] - sC2.data()[1]);
		scalar dEdc22_z = 2.0 * (sC3.data()[2] - sC2.data()[2]);

		scalar dFdc12_x = 2.0 * sC3.data()[0];
		scalar dFdc12_y = 2.0 * sC3.data()[1];
		scalar dFdc12_z = 2.0 * sC3.data()[2];
		scalar dFdc22 = -2.0 * sC3.data()[0];
		scalar dFdc22_y = -2.0 * sC3.data()[1];
		scalar dFd22_z = -2.0 * sC3.data()[2];

		scalar q1 = 0.5 * E / (C * C);
		scalar q2 = -0.5 / C;

		scalar dBTdc12_x = q2 * dEdc12_x;
		scalar dBTdc12_y = q2 * dEdc12_y;
		scalar dBTdc12_z = q2 * dEdc12_z;

		scalar dBTdc21_x = q1 * dCdc21_x + q2 * dEdc21_x;
		scalar dBTdc21_y = q1 * dCdc21_y + q2 * dEdc21_y;
		scalar dBTdc21_z = q1 * dCdc21_z + q2 * dEdc21_z;

		scalar dBTdc22_x = q1 * dCdc22_x + q2 * dEdc22_x;
		scalar dBTdc22_y = q1 * dCdc22_y + q2 * dEdc22_y;
		scalar dBTdc22_z = q1 * dCdc22_z + q2 * dEdc22_z;

		scalar C1x = sC1.data()[0];
		scalar C1y = sC1.data()[1];
		scalar C1z = sC1.data()[2];
		scalar C2x = sC2.data()[0];
		scalar C2y = sC2.data()[1];
		scalar C2z = sC2.data()[2];
		scalar C3x = sC3.data()[0];
		scalar C3y = sC3.data()[1];
		scalar C3z = sC3.data()[2];

		scalar vx = 2.0 * (beta * C2x + C3x);
		scalar vy = 2.0 * (beta * C2y + C3y);
		scalar vz = 2.0 * (beta * C2z + C3z);

		scalar dvxdc12_x = 2.0 * (C2x * dBTdc12_x + 1);
		scalar dvxdc12_y = 2.0 * (C2x * dBTdc12_y);
		scalar dvxdc12_z = 2.0 * (C2x * dBTdc12_z);

		scalar dvxdc21_x = 2.0 * (C2x * dBTdc21_x - beta);
		scalar dvxdc21_y = 2.0 * (C2x * dBTdc21_y);
		scalar dvxdc21_z = 2.0 * (C2x * dBTdc21_z);

		scalar dvxdc22_x = 2.0 * (C2x * dBTdc22_x + beta - 1);
		scalar dvxdc22_y = 2.0 * (C2x * dBTdc22_y);
		scalar dvxdc22_z = 2.0 * (C2x * dBTdc22_z);
		//
		scalar dvydc12_x = 2.0 * (C2y * dBTdc12_x);
		scalar dvydc12_y = 2.0 * (C2y * dBTdc12_y + 1);
		scalar dvydc12_z = 2.0 * (C2y * dBTdc12_z);

		scalar dvydc21_x = 2.0 * (C2y * dBTdc21_x);
		scalar dvydc21_y = 2.0 * (C2y * dBTdc21_y - beta);
		scalar dvydc21_z = 2.0 * (C2y * dBTdc21_z);

		scalar dvydc22_x = 2.0 * (C2y * dBTdc22_x);
		scalar dvydc22_y = 2.0 * (C2y * dBTdc22_y + beta - 1);
		scalar dvydc22_z = 2.0 * (C2y * dBTdc22_z);
		//

		scalar dvzdc12_x = 2.0 * (C2z * dBTdc12_x);
		scalar dvzdc12_y = 2.0 * (C2z * dBTdc12_y);
		scalar dvzdc12_z = 2.0 * (C2z * dBTdc12_z + 1);

		scalar dvzdc21_x = 2.0 * (C2z * dBTdc21_x);
		scalar dvzdc21_y = 2.0 * (C2z * dBTdc21_y);
		scalar dvzdc21_z = 2.0 * (C2z * dBTdc21_z - beta);

		scalar dvzdc22_x = 2.0 * (C2z * dBTdc22_x);
		scalar dvzdc22_y = 2.0 * (C2z * dBTdc22_y);
		scalar dvzdc22_z = 2.0 * (C2z * dBTdc22_z + beta - 1);

		hessian.setZero();
		vec12 z12;
		z12.setZero(12);
		hessian << z12.transpose(),
			z12.transpose(),
			z12.transpose(),
			0, 0, 0, dvxdc12_x, dvydc12_x, dvzdc12_x, vx * -dBTdc12_x - beta * dvxdc12_x, vy * -dBTdc12_x - beta * dvydc12_x, vz * -dBTdc12_x - beta * dvzdc12_x, vx * dBTdc12_x + (beta - 1.0) * dvxdc12_x, vy * dBTdc12_x + (beta - 1.0) * dvydc12_x, vz * dBTdc12_x + (beta - 1.0) * dvzdc12_x,
			0, 0, 0, dvxdc12_y, dvydc12_y, dvzdc12_y, vx * -dBTdc12_y - beta * dvxdc12_y, vy * -dBTdc12_y - beta * dvydc12_y, vz * -dBTdc12_y - beta * dvzdc12_y, vx * dBTdc12_y + (beta - 1.0) * dvxdc12_y, vy * dBTdc12_y + (beta - 1.0) * dvydc12_y, vz * dBTdc12_y + (beta - 1.0) * dvzdc12_y,
			0, 0, 0, dvxdc12_z, dvydc12_z, dvzdc12_z, vx * -dBTdc12_z - beta * dvxdc12_z, vy * -dBTdc12_z - beta * dvydc12_z, vz * -dBTdc12_z - beta * dvzdc12_z, vx * dBTdc12_z + (beta - 1.0) * dvxdc12_z, vy * dBTdc12_z + (beta - 1.0) * dvydc12_z, vz * dBTdc12_z + (beta - 1.0) * dvzdc12_z,
			0, 0, 0, dvxdc21_x, dvydc21_x, dvzdc21_x, vx * -dBTdc21_x - beta * dvxdc21_x, vy * -dBTdc21_x - beta * dvydc21_x, vz * -dBTdc21_x - beta * dvzdc21_x, vx * dBTdc21_x + (beta - 1.0) * dvxdc21_x, vy * dBTdc21_x + (beta - 1.0) * dvydc21_x, vz * dBTdc21_x + (beta - 1.0) * dvzdc21_x,
			0, 0, 0, dvxdc21_y, dvydc21_y, dvzdc21_y, vx * -dBTdc21_y - beta * dvxdc21_y, vy * -dBTdc21_y - beta * dvydc21_y, vz * -dBTdc21_y - beta * dvzdc21_y, vx * dBTdc21_y + (beta - 1.0) * dvxdc21_y, vy * dBTdc21_y + (beta - 1.0) * dvydc21_y, vz * dBTdc21_y + (beta - 1.0) * dvzdc21_y,
			0, 0, 0, dvxdc21_z, dvydc21_z, dvzdc21_z, vx * -dBTdc21_z - beta * dvxdc21_z, vy * -dBTdc21_z - beta * dvydc21_z, vz * -dBTdc21_z - beta * dvzdc21_z, vx * dBTdc21_z + (beta - 1.0) * dvxdc21_z, vy * dBTdc21_z + (beta - 1.0) * dvydc21_z, vz * dBTdc21_z + (beta - 1.0) * dvzdc21_z,
			0, 0, 0, dvxdc22_x, dvydc22_x, dvzdc22_x, vx * -dBTdc22_x - beta * dvxdc22_x, vy * -dBTdc22_x - beta * dvydc22_x, vz * -dBTdc22_x - beta * dvzdc22_x, vx * dBTdc22_x + (beta - 1.0) * dvxdc22_x, vy * dBTdc22_x + (beta - 1.0) * dvydc22_x, vz * dBTdc22_x + (beta - 1.0) * dvzdc22_x,
			0, 0, 0, dvxdc22_y, dvydc22_y, dvzdc22_y, vx * -dBTdc22_y - beta * dvxdc22_y, vy * -dBTdc22_y - beta * dvydc22_y, vz * -dBTdc22_y - beta * dvzdc22_y, vx * dBTdc22_y + (beta - 1.0) * dvxdc22_y, vy * dBTdc22_y + (beta - 1.0) * dvydc22_y, vz * dBTdc22_y + (beta - 1.0) * dvzdc22_y,
			0, 0, 0, dvxdc22_z, dvydc22_z, dvzdc22_z, vx * -dBTdc22_z - beta * dvxdc22_z, vy * -dBTdc22_z - beta * dvydc22_z, vz * -dBTdc22_z - beta * dvzdc22_z, vx * dBTdc22_z + (beta - 1.0) * dvxdc22_z, vy * dBTdc22_z + (beta - 1.0) * dvydc22_z, vz * dBTdc22_z + (beta - 1.0) * dvzdc22_z;
	}
	void MipcConeConeConstraint::alphaIsZeroHessian1(mat12 &hessian)
	{
		scalar dAdc11_x = 2.0 * sC1.data()[0];
		scalar dAdc11_y = 2.0 * sC1.data()[1];
		scalar dAdc11_z = 2.0 * sC1.data()[2];
		scalar dAdc12_x = -2.0 * sC1.data()[0];
		scalar dAdc12_y = -2.0 * sC1.data()[1];
		scalar dAdc12_z = -2.0 * sC1.data()[2];

		scalar dBdc11_x = 2.0 * sC2.data()[0];
		scalar dBdc11_y = 2.0 * sC2.data()[1];
		scalar dBdc11_z = 2.0 * sC2.data()[2];
		scalar dBdc12_x = -2.0 * sC2.data()[0];
		scalar dBdc12_y = -2.0 * sC2.data()[1];
		scalar dBdc12_z = -2.0 * sC2.data()[2];
		scalar dBdc21_x = -2.0 * sC1.data()[0];
		scalar dBdc21_y = -2.0 * sC1.data()[1];
		scalar dBdc21_z = -2.0 * sC1.data()[2];
		scalar dBdc22_x = 2.0 * sC1.data()[0];
		scalar dBdc22_y = 2.0 * sC1.data()[1];
		scalar dBdc22_z = 2.0 * sC1.data()[2];

		scalar dCdc21_x = -2.0 * sC2.data()[0];
		scalar dCdc21_y = -2.0 * sC2.data()[1];
		scalar dCdc21_z = -2.0 * sC2.data()[2];
		scalar dCdc22_x = 2.0 * sC2.data()[0];
		scalar dCdc22_y = 2.0 * sC2.data()[1];
		scalar dCdc22_z = 2.0 * sC2.data()[2];

		scalar dDdc11_x = 2.0 * sC3.data()[0];
		scalar dDdc11_y = 2.0 * sC3.data()[1];
		scalar dDdc11_z = 2.0 * sC3.data()[2];
		scalar dDdc12_x = 2.0 * (sC1.data()[0] - sC3.data()[0]);
		scalar dDdc12_y = 2.0 * (sC1.data()[1] - sC3.data()[1]);
		scalar dDdc12_z = 2.0 * (sC1.data()[2] - sC3.data()[2]);
		scalar dDdc22_x = -2.0 * sC1.data()[0];
		scalar dDdc22_y = -2.0 * sC1.data()[1];
		scalar dDdc22_z = -2.0 * sC1.data()[2];

		scalar dEdc12_x = 2.0 * sC2.data()[0];
		scalar dEdc12_y = 2.0 * sC2.data()[1];
		scalar dEdc12_z = 2.0 * sC2.data()[2];
		scalar dEdc21_x = -2.0 * sC3.data()[0];
		scalar dEdc21_y = -2.0 * sC3.data()[1];
		scalar dEdc21_z = -2.0 * sC3.data()[2];
		scalar dEdc22_x = 2.0 * (sC3.data()[0] - sC2.data()[0]);
		scalar dEdc22_y = 2.0 * (sC3.data()[1] - sC2.data()[1]);
		scalar dEdc22_z = 2.0 * (sC3.data()[2] - sC2.data()[2]);

		scalar dFdc12_x = 2.0 * sC3.data()[0];
		scalar dFdc12_y = 2.0 * sC3.data()[1];
		scalar dFdc12_z = 2.0 * sC3.data()[2];
		scalar dFdc22 = -2.0 * sC3.data()[0];
		scalar dFdc22_y = -2.0 * sC3.data()[1];
		scalar dFd22_z = -2.0 * sC3.data()[2];

		//
		scalar q1 = 0.5 * E / (C * C);
		scalar q2 = -0.5 / C;

		scalar dBTdc12_x = q2 * dEdc12_x;
		scalar dBTdc12_y = q2 * dEdc12_y;
		scalar dBTdc12_z = q2 * dEdc12_z;

		scalar dBTdc21_x = q1 * dCdc21_x + q2 * dEdc21_x;
		scalar dBTdc21_y = q1 * dCdc21_y + q2 * dEdc21_y;
		scalar dBTdc21_z = q1 * dCdc21_z + q2 * dEdc21_z;

		scalar dBTdc22_x = q1 * dCdc22_x + q2 * dEdc22_x;
		scalar dBTdc22_y = q1 * dCdc22_y + q2 * dEdc22_y;
		scalar dBTdc22_z = q1 * dCdc22_z + q2 * dEdc22_z;

		scalar C1x = sC1.data()[0];
		scalar C1y = sC1.data()[1];
		scalar C1z = sC1.data()[2];
		scalar C2x = sC2.data()[0];
		scalar C2y = sC2.data()[1];
		scalar C2z = sC2.data()[2];
		scalar C3x = sC3.data()[0];
		scalar C3y = sC3.data()[1];
		scalar C3z = sC3.data()[2];

		scalar vx = 2.0 * (beta * C2x + C3x);
		scalar vy = 2.0 * (beta * C2y + C3y);
		scalar vz = 2.0 * (beta * C2z + C3z);

		scalar dvxdc12_x = 2.0 * (C2x * dBTdc12_x + 1);
		scalar dvxdc12_y = 2.0 * (C2x * dBTdc12_y);
		scalar dvxdc12_z = 2.0 * (C2x * dBTdc12_z);

		scalar dvxdc21_x = 2.0 * (C2x * dBTdc21_x - beta);
		scalar dvxdc21_y = 2.0 * (C2x * dBTdc21_y);
		scalar dvxdc21_z = 2.0 * (C2x * dBTdc21_z);

		scalar dvxdc22_x = 2.0 * (C2x * dBTdc22_x + beta - 1);
		scalar dvxdc22_y = 2.0 * (C2x * dBTdc22_y);
		scalar dvxdc22_z = 2.0 * (C2x * dBTdc22_z);
		//
		scalar dvydc12_x = 2.0 * (C2y * dBTdc12_x);
		scalar dvydc12_y = 2.0 * (C2y * dBTdc12_y + 1);
		scalar dvydc12_z = 2.0 * (C2y * dBTdc12_z);

		scalar dvydc21_x = 2.0 * (C2y * dBTdc21_x);
		scalar dvydc21_y = 2.0 * (C2y * dBTdc21_y - beta);
		scalar dvydc21_z = 2.0 * (C2y * dBTdc21_z);

		scalar dvydc22_x = 2.0 * (C2y * dBTdc22_x);
		scalar dvydc22_y = 2.0 * (C2y * dBTdc22_y + beta - 1);
		scalar dvydc22_z = 2.0 * (C2y * dBTdc22_z);
		//

		scalar dvzdc12_x = 2.0 * (C2z * dBTdc12_x);
		scalar dvzdc12_y = 2.0 * (C2z * dBTdc12_y);
		scalar dvzdc12_z = 2.0 * (C2z * dBTdc12_z + 1);

		scalar dvzdc21_x = 2.0 * (C2z * dBTdc21_x);
		scalar dvzdc21_y = 2.0 * (C2z * dBTdc21_y);
		scalar dvzdc21_z = 2.0 * (C2z * dBTdc21_z - beta);

		scalar dvzdc22_x = 2.0 * (C2z * dBTdc22_x);
		scalar dvzdc22_y = 2.0 * (C2z * dBTdc22_y);
		scalar dvzdc22_z = 2.0 * (C2z * dBTdc22_z + beta - 1);

		hessian.setZero();
		// f / c11x
		hessian.data()[0] = 0;
		hessian.data()[1] = 0;
		hessian.data()[2] = 0;

		hessian.data()[3] = 0;
		hessian.data()[4] = 0;
		hessian.data()[5] = 0;

		hessian.data()[6] = 0;
		hessian.data()[7] = 0;
		hessian.data()[8] = 0;

		hessian.data()[9] = 0;
		hessian.data()[10] = 0;
		hessian.data()[11] = 0;
		// f / c11y
		hessian.data()[12] = 0;
		hessian.data()[13] = 0;
		hessian.data()[14] = 0;

		hessian.data()[15] = 0;
		hessian.data()[16] = 0;
		hessian.data()[17] = 0;

		hessian.data()[18] = 0;
		hessian.data()[19] = 0;
		hessian.data()[20] = 0;

		hessian.data()[21] = 0;
		hessian.data()[22] = 0;
		hessian.data()[23] = 0;
		// f / c11z
		hessian.data()[24] = 0;
		hessian.data()[25] = 0;
		hessian.data()[26] = 0;

		hessian.data()[27] = 0;
		hessian.data()[28] = 0;
		hessian.data()[29] = 0;

		hessian.data()[30] = 0;
		hessian.data()[31] = 0;
		hessian.data()[32] = 0;

		hessian.data()[33] = 0;
		hessian.data()[34] = 0;
		hessian.data()[35] = 0;

		// f / c12x
		hessian.data()[36] = 0;
		hessian.data()[37] = 0;
		hessian.data()[38] = 0;

		hessian.data()[39] = dvxdc12_x;
		hessian.data()[40] = dvydc12_x;
		hessian.data()[41] = dvzdc12_x;

		hessian.data()[42] = vx * -dBTdc12_x - beta * dvxdc12_x;
		hessian.data()[43] = vy * -dBTdc12_x - beta * dvydc12_x;
		hessian.data()[44] = vz * -dBTdc12_x - beta * dvzdc12_x;

		hessian.data()[45] = vx * dBTdc12_x + (beta - 1.0) * dvxdc12_x;
		hessian.data()[46] = vy * dBTdc12_x + (beta - 1.0) * dvydc12_x;
		hessian.data()[47] = vz * dBTdc12_x + (beta - 1.0) * dvzdc12_x;
		// f / c12y
		hessian.data()[48] = 0;
		hessian.data()[49] = 0;
		hessian.data()[50] = 0;

		hessian.data()[51] = dvxdc12_y;
		hessian.data()[52] = dvydc12_y;
		hessian.data()[53] = dvzdc12_y;

		hessian.data()[54] = vx * -dBTdc12_y - beta * dvxdc12_y;
		hessian.data()[55] = vy * -dBTdc12_y - beta * dvydc12_y;
		hessian.data()[56] = vz * -dBTdc12_y - beta * dvzdc12_y;

		hessian.data()[57] = vx * dBTdc12_y + (beta - 1.0) * dvxdc12_y;
		hessian.data()[58] = vy * dBTdc12_y + (beta - 1.0) * dvydc12_y;
		hessian.data()[59] = vz * dBTdc12_y + (beta - 1.0) * dvzdc12_y;
		// f / c12z
		hessian.data()[60] = 0;
		hessian.data()[61] = 0;
		hessian.data()[62] = 0;

		hessian.data()[63] = dvxdc12_z;
		hessian.data()[64] = dvydc12_z;
		hessian.data()[65] = dvzdc12_z;

		hessian.data()[66] = vx * -dBTdc12_z - beta * dvxdc12_z;
		hessian.data()[67] = vy * -dBTdc12_z - beta * dvydc12_z;
		hessian.data()[68] = vz * -dBTdc12_z - beta * dvzdc12_z;

		hessian.data()[69] = vx * dBTdc12_z + (beta - 1.0) * dvxdc12_z;
		hessian.data()[70] = vy * dBTdc12_z + (beta - 1.0) * dvydc12_z;
		hessian.data()[71] = vz * dBTdc12_z + (beta - 1.0) * dvzdc12_z;

		// f / c21x
		hessian.data()[72] = 0;
		hessian.data()[73] = 0;
		hessian.data()[74] = 0;

		hessian.data()[75] = dvxdc21_x;
		hessian.data()[76] = dvydc21_x;
		hessian.data()[77] = dvzdc21_x;

		hessian.data()[78] = vx * -dBTdc21_x - beta * dvxdc21_x;
		hessian.data()[79] = vy * -dBTdc21_x - beta * dvydc21_x;
		hessian.data()[80] = vz * -dBTdc21_x - beta * dvzdc21_x;

		hessian.data()[81] = vx * dBTdc21_x + (beta - 1.0) * dvxdc21_x;
		hessian.data()[82] = vy * dBTdc21_x + (beta - 1.0) * dvydc21_x;
		hessian.data()[83] = vz * dBTdc21_x + (beta - 1.0) * dvzdc21_x;
		// f / c21y
		hessian.data()[84] = 0;
		hessian.data()[85] = 0;
		hessian.data()[86] = 0;

		hessian.data()[87] = dvxdc21_y;
		hessian.data()[88] = dvydc21_y;
		hessian.data()[89] = dvzdc21_y;

		hessian.data()[90] = vx * -dBTdc21_y - beta * dvxdc21_y;
		hessian.data()[91] = vy * -dBTdc21_y - beta * dvydc21_y;
		hessian.data()[92] = vz * -dBTdc21_y - beta * dvzdc21_y;

		hessian.data()[93] = vx * dBTdc21_y + (beta - 1.0) * dvxdc21_y;
		hessian.data()[94] = vy * dBTdc21_y + (beta - 1.0) * dvydc21_y;
		hessian.data()[95] = vz * dBTdc21_y + (beta - 1.0) * dvzdc21_y;
		// f / c21z
		hessian.data()[96] = 0;
		hessian.data()[97] = 0;
		hessian.data()[98] = 0;

		hessian.data()[99] = dvxdc21_z;
		hessian.data()[100] = dvydc21_z;
		hessian.data()[101] = dvzdc21_z;

		hessian.data()[102] = vx * -dBTdc21_z - beta * dvxdc21_z;
		hessian.data()[103] = vy * -dBTdc21_z - beta * dvydc21_z;
		hessian.data()[104] = vz * -dBTdc21_z - beta * dvzdc21_z;

		hessian.data()[105] = vx * dBTdc21_z + (beta - 1.0) * dvxdc21_z;
		hessian.data()[106] = vy * dBTdc21_z + (beta - 1.0) * dvydc21_z;
		hessian.data()[107] = vz * dBTdc21_z + (beta - 1.0) * dvzdc21_z;

		// f / c22x
		hessian.data()[108] = 0;
		hessian.data()[109] = 0;
		hessian.data()[110] = 0;

		hessian.data()[111] = dvxdc22_x;
		hessian.data()[112] = dvydc22_x;
		hessian.data()[113] = dvzdc22_x;

		hessian.data()[114] = vx * -dBTdc22_x - beta * dvxdc22_x;
		hessian.data()[115] = vy * -dBTdc22_x - beta * dvydc22_x;
		hessian.data()[116] = vz * -dBTdc22_x - beta * dvzdc22_x;

		hessian.data()[117] = vx * dBTdc22_x + (beta - 1.0) * dvxdc22_x;
		hessian.data()[118] = vy * dBTdc22_x + (beta - 1.0) * dvydc22_x;
		hessian.data()[119] = vz * dBTdc22_x + (beta - 1.0) * dvzdc22_x;
		// f / c22y
		hessian.data()[120] = 0;
		hessian.data()[121] = 0;
		hessian.data()[122] = 0;

		hessian.data()[123] = dvxdc22_y;
		hessian.data()[124] = dvydc22_y;
		hessian.data()[125] = dvzdc22_y;

		hessian.data()[126] = vx * -dBTdc22_y - beta * dvxdc22_y;
		hessian.data()[127] = vy * -dBTdc22_y - beta * dvydc22_y;
		hessian.data()[128] = vz * -dBTdc22_y - beta * dvzdc22_y;

		hessian.data()[129] = vx * dBTdc22_y + (beta - 1.0) * dvxdc22_y;
		hessian.data()[130] = vy * dBTdc22_y + (beta - 1.0) * dvydc22_y;
		hessian.data()[131] = vz * dBTdc22_y + (beta - 1.0) * dvzdc22_y;
		// f / c22z
		hessian.data()[132] = 0;
		hessian.data()[133] = 0;
		hessian.data()[134] = 0;

		hessian.data()[135] = dvxdc22_z;
		hessian.data()[136] = dvydc22_z;
		hessian.data()[137] = dvzdc22_z;

		hessian.data()[138] = vx * -dBTdc22_z - beta * dvxdc22_z;
		hessian.data()[139] = vy * -dBTdc22_z - beta * dvydc22_z;
		hessian.data()[140] = vz * -dBTdc22_z - beta * dvzdc22_z;

		hessian.data()[141] = vx * dBTdc22_z + (beta - 1.0) * dvxdc22_z;
		hessian.data()[142] = vy * dBTdc22_z + (beta - 1.0) * dvydc22_z;
		hessian.data()[143] = vz * dBTdc22_z + (beta - 1.0) * dvzdc22_z;
	}

	void MipcConeConeConstraint::alphaIsOneHessian(mat12 &hessian)
	{
		scalar dAdc11_x = 2.0 * sC1.data()[0];
		scalar dAdc11_y = 2.0 * sC1.data()[1];
		scalar dAdc11_z = 2.0 * sC1.data()[2];
		scalar dAdc12_x = -2.0 * sC1.data()[0];
		scalar dAdc12_y = -2.0 * sC1.data()[1];
		scalar dAdc12_z = -2.0 * sC1.data()[2];

		scalar dBdc11_x = 2.0 * sC2.data()[0];
		scalar dBdc11_y = 2.0 * sC2.data()[1];
		scalar dBdc11_z = 2.0 * sC2.data()[2];
		scalar dBdc12_x = -2.0 * sC2.data()[0];
		scalar dBdc12_y = -2.0 * sC2.data()[1];
		scalar dBdc12_z = -2.0 * sC2.data()[2];
		scalar dBdc21_x = -2.0 * sC1.data()[0];
		scalar dBdc21_y = -2.0 * sC1.data()[1];
		scalar dBdc21_z = -2.0 * sC1.data()[2];
		scalar dBdc22_x = 2.0 * sC1.data()[0];
		scalar dBdc22_y = 2.0 * sC1.data()[1];
		scalar dBdc22_z = 2.0 * sC1.data()[2];

		scalar dCdc21_x = -2.0 * sC2.data()[0];
		scalar dCdc21_y = -2.0 * sC2.data()[1];
		scalar dCdc21_z = -2.0 * sC2.data()[2];
		scalar dCdc22_x = 2.0 * sC2.data()[0];
		scalar dCdc22_y = 2.0 * sC2.data()[1];
		scalar dCdc22_z = 2.0 * sC2.data()[2];

		scalar dDdc11_x = 2.0 * sC3.data()[0];
		scalar dDdc11_y = 2.0 * sC3.data()[1];
		scalar dDdc11_z = 2.0 * sC3.data()[2];
		scalar dDdc12_x = 2.0 * (sC1.data()[0] - sC3.data()[0]);
		scalar dDdc12_y = 2.0 * (sC1.data()[1] - sC3.data()[1]);
		scalar dDdc12_z = 2.0 * (sC1.data()[2] - sC3.data()[2]);
		scalar dDdc22_x = -2.0 * sC1.data()[0];
		scalar dDdc22_y = -2.0 * sC1.data()[1];
		scalar dDdc22_z = -2.0 * sC1.data()[2];

		scalar dEdc12_x = 2.0 * sC2.data()[0];
		scalar dEdc12_y = 2.0 * sC2.data()[1];
		scalar dEdc12_z = 2.0 * sC2.data()[2];
		scalar dEdc21_x = -2.0 * sC3.data()[0];
		scalar dEdc21_y = -2.0 * sC3.data()[1];
		scalar dEdc21_z = -2.0 * sC3.data()[2];
		scalar dEdc22_x = 2.0 * (sC3.data()[0] - sC2.data()[0]);
		scalar dEdc22_y = 2.0 * (sC3.data()[1] - sC2.data()[1]);
		scalar dEdc22_z = 2.0 * (sC3.data()[2] - sC2.data()[2]);

		scalar dFdc12_x = 2.0 * sC3.data()[0];
		scalar dFdc12_y = 2.0 * sC3.data()[1];
		scalar dFdc12_z = 2.0 * sC3.data()[2];
		scalar dFdc22 = -2.0 * sC3.data()[0];
		scalar dFdc22_y = -2.0 * sC3.data()[1];
		scalar dFd22_z = -2.0 * sC3.data()[2];

		//
		scalar q1 = 0.5 * (B + E) / (C * C);
		scalar q2 = -0.5 / C;

		scalar dALdc11_x = 0;
		scalar dALdc11_y = 0;
		scalar dALdc11_z = 0;

		scalar dALdc12_x = 0;
		scalar dALdc12_y = 0;
		scalar dALdc12_z = 0;

		scalar dALdc21_x = 0;
		scalar dALdc21_y = 0;
		scalar dALdc21_z = 0;

		scalar dALdc22_x = 0;
		scalar dALdc22_y = 0;
		scalar dALdc22_z = 0;

		scalar dBTdc11_x = q2 * dBdc11_x;
		scalar dBTdc11_y = q2 * dBdc11_y;
		scalar dBTdc11_z = q2 * dBdc11_z;

		scalar dBTdc12_x = q2 * (dBdc12_x + dEdc12_x);
		scalar dBTdc12_y = q2 * (dBdc12_y + dEdc12_y);
		scalar dBTdc12_z = q2 * (dBdc12_z + dEdc12_z);

		scalar dBTdc21_x = q1 * dCdc21_x + q2 * (dBdc21_x + dEdc21_x);
		scalar dBTdc21_y = q1 * dCdc21_y + q2 * (dBdc21_y + dEdc21_y);
		scalar dBTdc21_z = q1 * dCdc21_z + q2 * (dBdc21_z + dEdc21_z);

		scalar dBTdc22_x = q1 * dCdc22_x + q2 * (dBdc22_x + dEdc22_x);
		scalar dBTdc22_y = q1 * dCdc22_y + q2 * (dBdc22_y + dEdc22_y);
		scalar dBTdc22_z = q1 * dCdc22_z + q2 * (dBdc22_z + dEdc22_z);

		scalar C1x = sC1.data()[0];
		scalar C1y = sC1.data()[1];
		scalar C1z = sC1.data()[2];
		scalar C2x = sC2.data()[0];
		scalar C2y = sC2.data()[1];
		scalar C2z = sC2.data()[2];
		scalar C3x = sC3.data()[0];
		scalar C3y = sC3.data()[1];
		scalar C3z = sC3.data()[2];

		scalar vx = 2.0 * (alpha * C1x + beta * C2x + C3x);
		scalar vy = 2.0 * (alpha * C1y + beta * C2y + C3y);
		scalar vz = 2.0 * (alpha * C1z + beta * C2z + C3z);

		scalar dvxdc11_x = 2.0 * (C1x * dALdc11_x + C2x * dBTdc11_x + alpha);
		scalar dvxdc11_y = 2.0 * (C1x * dALdc11_y + C2x * dBTdc11_y);
		scalar dvxdc11_z = 2.0 * (C1x * dALdc11_z + C2x * dBTdc11_z);

		scalar dvxdc12_x = 2.0 * (C1x * dALdc12_x + C2x * dBTdc12_x - alpha + 1);
		scalar dvxdc12_y = 2.0 * (C1x * dALdc12_y + C2x * dBTdc12_y);
		scalar dvxdc12_z = 2.0 * (C1x * dALdc12_z + C2x * dBTdc12_z);

		scalar dvxdc21_x = 2.0 * (C1x * dALdc21_x + C2x * dBTdc21_x - beta);
		scalar dvxdc21_y = 2.0 * (C1x * dALdc21_y + C2x * dBTdc21_y);
		scalar dvxdc21_z = 2.0 * (C1x * dALdc21_z + C2x * dBTdc21_z);

		scalar dvxdc22_x = 2.0 * (C1x * dALdc22_x + C2x * dBTdc22_x + beta - 1);
		scalar dvxdc22_y = 2.0 * (C1x * dALdc22_y + C2x * dBTdc22_y);
		scalar dvxdc22_z = 2.0 * (C1x * dALdc22_z + C2x * dBTdc22_z);
		//
		scalar dvydc11_x = 2.0 * (C1y * dALdc11_x + C2y * dBTdc11_x);
		scalar dvydc11_y = 2.0 * (C1y * dALdc11_y + C2y * dBTdc11_y + alpha);
		scalar dvydc11_z = 2.0 * (C1y * dALdc11_z + C2y * dBTdc11_z);

		scalar dvydc12_x = 2.0 * (C1y * dALdc12_x + C2y * dBTdc12_x);
		scalar dvydc12_y = 2.0 * (C1y * dALdc12_y + C2y * dBTdc12_y - alpha + 1);
		scalar dvydc12_z = 2.0 * (C1y * dALdc12_z + C2y * dBTdc12_z);

		scalar dvydc21_x = 2.0 * (C1y * dALdc21_x + C2y * dBTdc21_x);
		scalar dvydc21_y = 2.0 * (C1y * dALdc21_y + C2y * dBTdc21_y - beta);
		scalar dvydc21_z = 2.0 * (C1y * dALdc21_z + C2y * dBTdc21_z);

		scalar dvydc22_x = 2.0 * (C1y * dALdc22_x + C2y * dBTdc22_x);
		scalar dvydc22_y = 2.0 * (C1y * dALdc22_y + C2y * dBTdc22_y + beta - 1);
		scalar dvydc22_z = 2.0 * (C1y * dALdc22_z + C2y * dBTdc22_z);
		//
		scalar dvzdc11_x = 2.0 * (C1z * dALdc11_x + C2z * dBTdc11_x);
		scalar dvzdc11_y = 2.0 * (C1z * dALdc11_y + C2z * dBTdc11_y);
		scalar dvzdc11_z = 2.0 * (C1z * dALdc11_z + C2z * dBTdc11_z + alpha);

		scalar dvzdc12_x = 2.0 * (C1z * dALdc12_x + C2z * dBTdc12_x);
		scalar dvzdc12_y = 2.0 * (C1z * dALdc12_y + C2z * dBTdc12_y);
		scalar dvzdc12_z = 2.0 * (C1z * dALdc12_z + C2z * dBTdc12_z - alpha + 1);

		scalar dvzdc21_x = 2.0 * (C1z * dALdc21_x + C2z * dBTdc21_x);
		scalar dvzdc21_y = 2.0 * (C1z * dALdc21_y + C2z * dBTdc21_y);
		scalar dvzdc21_z = 2.0 * (C1z * dALdc21_z + C2z * dBTdc21_z - beta);

		scalar dvzdc22_x = 2.0 * (C1z * dALdc22_x + C2z * dBTdc22_x);
		scalar dvzdc22_y = 2.0 * (C1z * dALdc22_y + C2z * dBTdc22_y);
		scalar dvzdc22_z = 2.0 * (C1z * dALdc22_z + C2z * dBTdc22_z + beta - 1);

		hessian.setZero();
		// f / c11x
		hessian.data()[0] = vx * dALdc11_x + alpha * dvxdc11_x;
		hessian.data()[1] = vy * dALdc11_x + alpha * dvydc11_x;
		hessian.data()[2] = vz * dALdc11_x + alpha * dvzdc11_x;

		hessian.data()[3] = vx * -dALdc11_x + (1 - alpha) * dvxdc11_x;
		hessian.data()[4] = vy * -dALdc11_x + (1 - alpha) * dvydc11_x;
		hessian.data()[5] = vz * -dALdc11_x + (1 - alpha) * dvzdc11_x;

		hessian.data()[6] = vx * -dBTdc11_x - beta * dvxdc11_x;
		hessian.data()[7] = vy * -dBTdc11_x - beta * dvydc11_x;
		hessian.data()[8] = vz * -dBTdc11_x - beta * dvzdc11_x;

		hessian.data()[9] = vx * dBTdc11_x + (beta - 1.0) * dvxdc11_x;
		hessian.data()[10] = vy * dBTdc11_x + (beta - 1.0) * dvydc11_x;
		hessian.data()[11] = vz * dBTdc11_x + (beta - 1.0) * dvzdc11_x;
		// f / c11y
		hessian.data()[12] = vx * dALdc11_y + alpha * dvxdc11_y;
		hessian.data()[13] = vy * dALdc11_y + alpha * dvydc11_y;
		hessian.data()[14] = vz * dALdc11_y + alpha * dvzdc11_y;

		hessian.data()[15] = vx * -dALdc11_y + (1 - alpha) * dvxdc11_y;
		hessian.data()[16] = vy * -dALdc11_y + (1 - alpha) * dvydc11_y;
		hessian.data()[17] = vz * -dALdc11_y + (1 - alpha) * dvzdc11_y;

		hessian.data()[18] = vx * -dBTdc11_y - beta * dvxdc11_y;
		hessian.data()[19] = vy * -dBTdc11_y - beta * dvydc11_y;
		hessian.data()[20] = vz * -dBTdc11_y - beta * dvzdc11_y;

		hessian.data()[21] = vx * dBTdc11_y + (beta - 1.0) * dvxdc11_y;
		hessian.data()[22] = vy * dBTdc11_y + (beta - 1.0) * dvydc11_y;
		hessian.data()[23] = vz * dBTdc11_y + (beta - 1.0) * dvzdc11_y;
		// f / c11z
		hessian.data()[24] = vx * dALdc11_z + alpha * dvxdc11_z;
		hessian.data()[25] = vy * dALdc11_z + alpha * dvydc11_z;
		hessian.data()[26] = vz * dALdc11_z + alpha * dvzdc11_z;

		hessian.data()[27] = vx * -dALdc11_z + (1 - alpha) * dvxdc11_z;
		hessian.data()[28] = vy * -dALdc11_z + (1 - alpha) * dvydc11_z;
		hessian.data()[29] = vz * -dALdc11_z + (1 - alpha) * dvzdc11_z;

		hessian.data()[30] = vx * -dBTdc11_z - beta * dvxdc11_z;
		hessian.data()[31] = vy * -dBTdc11_z - beta * dvydc11_z;
		hessian.data()[32] = vz * -dBTdc11_z - beta * dvzdc11_z;

		hessian.data()[33] = vx * dBTdc11_z + (beta - 1.0) * dvxdc11_z;
		hessian.data()[34] = vy * dBTdc11_z + (beta - 1.0) * dvydc11_z;
		hessian.data()[35] = vz * dBTdc11_z + (beta - 1.0) * dvzdc11_z;

		// f / c12x
		hessian.data()[36] = vx * dALdc12_x + alpha * dvxdc12_x;
		hessian.data()[37] = vy * dALdc12_x + alpha * dvydc12_x;
		hessian.data()[38] = vz * dALdc12_x + alpha * dvzdc12_x;

		hessian.data()[39] = vx * -dALdc12_x + (1 - alpha) * dvxdc12_x;
		hessian.data()[40] = vy * -dALdc12_x + (1 - alpha) * dvydc12_x;
		hessian.data()[41] = vz * -dALdc12_x + (1 - alpha) * dvzdc12_x;

		hessian.data()[42] = vx * -dBTdc12_x - beta * dvxdc12_x;
		hessian.data()[43] = vy * -dBTdc12_x - beta * dvydc12_x;
		hessian.data()[44] = vz * -dBTdc12_x - beta * dvzdc12_x;

		hessian.data()[45] = vx * dBTdc12_x + (beta - 1.0) * dvxdc12_x;
		hessian.data()[46] = vy * dBTdc12_x + (beta - 1.0) * dvydc12_x;
		hessian.data()[47] = vz * dBTdc12_x + (beta - 1.0) * dvzdc12_x;
		// f / c12y
		hessian.data()[48] = vx * dALdc12_y + alpha * dvxdc12_y;
		hessian.data()[49] = vy * dALdc12_y + alpha * dvydc12_y;
		hessian.data()[50] = vz * dALdc12_y + alpha * dvzdc12_y;

		hessian.data()[51] = vx * -dALdc12_y + (1 - alpha) * dvxdc12_y;
		hessian.data()[52] = vy * -dALdc12_y + (1 - alpha) * dvydc12_y;
		hessian.data()[53] = vz * -dALdc12_y + (1 - alpha) * dvzdc12_y;

		hessian.data()[54] = vx * -dBTdc12_y - beta * dvxdc12_y;
		hessian.data()[55] = vy * -dBTdc12_y - beta * dvydc12_y;
		hessian.data()[56] = vz * -dBTdc12_y - beta * dvzdc12_y;

		hessian.data()[57] = vx * dBTdc12_y + (beta - 1.0) * dvxdc12_y;
		hessian.data()[58] = vy * dBTdc12_y + (beta - 1.0) * dvydc12_y;
		hessian.data()[59] = vz * dBTdc12_y + (beta - 1.0) * dvzdc12_y;
		// f / c12z
		hessian.data()[60] = vx * dALdc12_z + alpha * dvxdc12_z;
		hessian.data()[61] = vy * dALdc12_z + alpha * dvydc12_z;
		hessian.data()[62] = vz * dALdc12_z + alpha * dvzdc12_z;

		hessian.data()[63] = vx * -dALdc12_z + (1 - alpha) * dvxdc12_z;
		hessian.data()[64] = vy * -dALdc12_z + (1 - alpha) * dvydc12_z;
		hessian.data()[65] = vz * -dALdc12_z + (1 - alpha) * dvzdc12_z;

		hessian.data()[66] = vx * -dBTdc12_z - beta * dvxdc12_z;
		hessian.data()[67] = vy * -dBTdc12_z - beta * dvydc12_z;
		hessian.data()[68] = vz * -dBTdc12_z - beta * dvzdc12_z;

		hessian.data()[69] = vx * dBTdc12_z + (beta - 1.0) * dvxdc12_z;
		hessian.data()[70] = vy * dBTdc12_z + (beta - 1.0) * dvydc12_z;
		hessian.data()[71] = vz * dBTdc12_z + (beta - 1.0) * dvzdc12_z;

		// f / c21x
		hessian.data()[72] = vx * dALdc21_x + alpha * dvxdc21_x;
		hessian.data()[73] = vy * dALdc21_x + alpha * dvydc21_x;
		hessian.data()[74] = vz * dALdc21_x + alpha * dvzdc21_x;

		hessian.data()[75] = vx * -dALdc21_x + (1 - alpha) * dvxdc21_x;
		hessian.data()[76] = vy * -dALdc21_x + (1 - alpha) * dvydc21_x;
		hessian.data()[77] = vz * -dALdc21_x + (1 - alpha) * dvzdc21_x;

		hessian.data()[78] = vx * -dBTdc21_x - beta * dvxdc21_x;
		hessian.data()[79] = vy * -dBTdc21_x - beta * dvydc21_x;
		hessian.data()[80] = vz * -dBTdc21_x - beta * dvzdc21_x;

		hessian.data()[81] = vx * dBTdc21_x + (beta - 1.0) * dvxdc21_x;
		hessian.data()[82] = vy * dBTdc21_x + (beta - 1.0) * dvydc21_x;
		hessian.data()[83] = vz * dBTdc21_x + (beta - 1.0) * dvzdc21_x;
		// f / c21y
		hessian.data()[84] = vx * dALdc21_y + alpha * dvxdc21_y;
		hessian.data()[85] = vy * dALdc21_y + alpha * dvydc21_y;
		hessian.data()[86] = vz * dALdc21_y + alpha * dvzdc21_y;

		hessian.data()[87] = vx * -dALdc21_y + (1 - alpha) * dvxdc21_y;
		hessian.data()[88] = vy * -dALdc21_y + (1 - alpha) * dvydc21_y;
		hessian.data()[89] = vz * -dALdc21_y + (1 - alpha) * dvzdc21_y;

		hessian.data()[90] = vx * -dBTdc21_y - beta * dvxdc21_y;
		hessian.data()[91] = vy * -dBTdc21_y - beta * dvydc21_y;
		hessian.data()[92] = vz * -dBTdc21_y - beta * dvzdc21_y;

		hessian.data()[93] = vx * dBTdc21_y + (beta - 1.0) * dvxdc21_y;
		hessian.data()[94] = vy * dBTdc21_y + (beta - 1.0) * dvydc21_y;
		hessian.data()[95] = vz * dBTdc21_y + (beta - 1.0) * dvzdc21_y;
		// f / c21z
		hessian.data()[96] = vx * dALdc21_z + alpha * dvxdc21_z;
		hessian.data()[97] = vy * dALdc21_z + alpha * dvydc21_z;
		hessian.data()[98] = vz * dALdc21_z + alpha * dvzdc21_z;

		hessian.data()[99] = vx * -dALdc21_z + (1 - alpha) * dvxdc21_z;
		hessian.data()[100] = vy * -dALdc21_z + (1 - alpha) * dvydc21_z;
		hessian.data()[101] = vz * -dALdc21_z + (1 - alpha) * dvzdc21_z;

		hessian.data()[102] = vx * -dBTdc21_z - beta * dvxdc21_z;
		hessian.data()[103] = vy * -dBTdc21_z - beta * dvydc21_z;
		hessian.data()[104] = vz * -dBTdc21_z - beta * dvzdc21_z;

		hessian.data()[105] = vx * dBTdc21_z + (beta - 1.0) * dvxdc21_z;
		hessian.data()[106] = vy * dBTdc21_z + (beta - 1.0) * dvydc21_z;
		hessian.data()[107] = vz * dBTdc21_z + (beta - 1.0) * dvzdc21_z;

		// f / c22x
		hessian.data()[108] = vx * dALdc22_x + alpha * dvxdc22_x;
		hessian.data()[109] = vy * dALdc22_x + alpha * dvydc22_x;
		hessian.data()[110] = vz * dALdc22_x + alpha * dvzdc22_x;

		hessian.data()[111] = vx * -dALdc22_x + (1 - alpha) * dvxdc22_x;
		hessian.data()[112] = vy * -dALdc22_x + (1 - alpha) * dvydc22_x;
		hessian.data()[113] = vz * -dALdc22_x + (1 - alpha) * dvzdc22_x;

		hessian.data()[114] = vx * -dBTdc22_x - beta * dvxdc22_x;
		hessian.data()[115] = vy * -dBTdc22_x - beta * dvydc22_x;
		hessian.data()[116] = vz * -dBTdc22_x - beta * dvzdc22_x;

		hessian.data()[117] = vx * dBTdc22_x + (beta - 1.0) * dvxdc22_x;
		hessian.data()[118] = vy * dBTdc22_x + (beta - 1.0) * dvydc22_x;
		hessian.data()[119] = vz * dBTdc22_x + (beta - 1.0) * dvzdc22_x;
		// f / c22y
		hessian.data()[120] = vx * dALdc22_y + alpha * dvxdc22_y;
		hessian.data()[121] = vy * dALdc22_y + alpha * dvydc22_y;
		hessian.data()[122] = vz * dALdc22_y + alpha * dvzdc22_y;

		hessian.data()[123] = vx * -dALdc22_y + (1 - alpha) * dvxdc22_y;
		hessian.data()[124] = vy * -dALdc22_y + (1 - alpha) * dvydc22_y;
		hessian.data()[125] = vz * -dALdc22_y + (1 - alpha) * dvzdc22_y;

		hessian.data()[126] = vx * -dBTdc22_y - beta * dvxdc22_y;
		hessian.data()[127] = vy * -dBTdc22_y - beta * dvydc22_y;
		hessian.data()[128] = vz * -dBTdc22_y - beta * dvzdc22_y;

		hessian.data()[129] = vx * dBTdc22_y + (beta - 1.0) * dvxdc22_y;
		hessian.data()[130] = vy * dBTdc22_y + (beta - 1.0) * dvydc22_y;
		hessian.data()[131] = vz * dBTdc22_y + (beta - 1.0) * dvzdc22_y;
		// f / c22z
		hessian.data()[132] = vx * dALdc22_z + alpha * dvxdc22_z;
		hessian.data()[133] = vy * dALdc22_z + alpha * dvydc22_z;
		hessian.data()[134] = vz * dALdc22_z + alpha * dvzdc22_z;

		hessian.data()[135] = vx * -dALdc22_z + (1 - alpha) * dvxdc22_z;
		hessian.data()[136] = vy * -dALdc22_z + (1 - alpha) * dvydc22_z;
		hessian.data()[137] = vz * -dALdc22_z + (1 - alpha) * dvzdc22_z;

		hessian.data()[138] = vx * -dBTdc22_z - beta * dvxdc22_z;
		hessian.data()[139] = vy * -dBTdc22_z - beta * dvydc22_z;
		hessian.data()[140] = vz * -dBTdc22_z - beta * dvzdc22_z;

		hessian.data()[141] = vx * dBTdc22_z + (beta - 1.0) * dvxdc22_z;
		hessian.data()[142] = vy * dBTdc22_z + (beta - 1.0) * dvydc22_z;
		hessian.data()[143] = vz * dBTdc22_z + (beta - 1.0) * dvzdc22_z;
	}
	void MipcConeConeConstraint::betaIsZeroHessian(mat12 &hessian)
	{
		scalar dAdc11_x = 2.0 * sC1.data()[0];
		scalar dAdc11_y = 2.0 * sC1.data()[1];
		scalar dAdc11_z = 2.0 * sC1.data()[2];
		scalar dAdc12_x = -2.0 * sC1.data()[0];
		scalar dAdc12_y = -2.0 * sC1.data()[1];
		scalar dAdc12_z = -2.0 * sC1.data()[2];

		scalar dBdc11_x = 2.0 * sC2.data()[0];
		scalar dBdc11_y = 2.0 * sC2.data()[1];
		scalar dBdc11_z = 2.0 * sC2.data()[2];
		scalar dBdc12_x = -2.0 * sC2.data()[0];
		scalar dBdc12_y = -2.0 * sC2.data()[1];
		scalar dBdc12_z = -2.0 * sC2.data()[2];
		scalar dBdc21_x = -2.0 * sC1.data()[0];
		scalar dBdc21_y = -2.0 * sC1.data()[1];
		scalar dBdc21_z = -2.0 * sC1.data()[2];
		scalar dBdc22_x = 2.0 * sC1.data()[0];
		scalar dBdc22_y = 2.0 * sC1.data()[1];
		scalar dBdc22_z = 2.0 * sC1.data()[2];

		scalar dCdc21_x = -2.0 * sC2.data()[0];
		scalar dCdc21_y = -2.0 * sC2.data()[1];
		scalar dCdc21_z = -2.0 * sC2.data()[2];
		scalar dCdc22_x = 2.0 * sC2.data()[0];
		scalar dCdc22_y = 2.0 * sC2.data()[1];
		scalar dCdc22_z = 2.0 * sC2.data()[2];

		scalar dDdc11_x = 2.0 * sC3.data()[0];
		scalar dDdc11_y = 2.0 * sC3.data()[1];
		scalar dDdc11_z = 2.0 * sC3.data()[2];
		scalar dDdc12_x = 2.0 * (sC1.data()[0] - sC3.data()[0]);
		scalar dDdc12_y = 2.0 * (sC1.data()[1] - sC3.data()[1]);
		scalar dDdc12_z = 2.0 * (sC1.data()[2] - sC3.data()[2]);
		scalar dDdc22_x = -2.0 * sC1.data()[0];
		scalar dDdc22_y = -2.0 * sC1.data()[1];
		scalar dDdc22_z = -2.0 * sC1.data()[2];

		scalar dEdc12_x = 2.0 * sC2.data()[0];
		scalar dEdc12_y = 2.0 * sC2.data()[1];
		scalar dEdc12_z = 2.0 * sC2.data()[2];
		scalar dEdc21_x = -2.0 * sC3.data()[0];
		scalar dEdc21_y = -2.0 * sC3.data()[1];
		scalar dEdc21_z = -2.0 * sC3.data()[2];
		scalar dEdc22_x = 2.0 * (sC3.data()[0] - sC2.data()[0]);
		scalar dEdc22_y = 2.0 * (sC3.data()[1] - sC2.data()[1]);
		scalar dEdc22_z = 2.0 * (sC3.data()[2] - sC2.data()[2]);

		scalar dFdc12_x = 2.0 * sC3.data()[0];
		scalar dFdc12_y = 2.0 * sC3.data()[1];
		scalar dFdc12_z = 2.0 * sC3.data()[2];
		scalar dFdc22 = -2.0 * sC3.data()[0];
		scalar dFdc22_y = -2.0 * sC3.data()[1];
		scalar dFd22_z = -2.0 * sC3.data()[2];

		//
		scalar q1 = 0.5 * D / (A * A);
		scalar q2 = -0.5 / A;

		scalar dALdc11_x = q1 * dAdc11_x + q2 * dDdc11_x;
		scalar dALdc11_y = q1 * dAdc11_y + q2 * dDdc11_y;
		scalar dALdc11_z = q1 * dAdc11_z + q2 * dDdc11_z;

		scalar dALdc12_x = q1 * dAdc12_x + q2 * dDdc12_x;
		scalar dALdc12_y = q1 * dAdc12_y + q2 * dDdc12_y;
		scalar dALdc12_z = q1 * dAdc12_z + q2 * dDdc12_z;

		scalar dALdc21_x = 0;
		scalar dALdc21_y = 0;
		scalar dALdc21_z = 0;

		scalar dALdc22_x = q2 * dDdc22_x;
		scalar dALdc22_y = q2 * dDdc22_y;
		scalar dALdc22_z = q2 * dDdc22_z;

		scalar dBTdc11_x = 0;
		scalar dBTdc11_y = 0;
		scalar dBTdc11_z = 0;

		scalar dBTdc12_x = 0;
		scalar dBTdc12_y = 0;
		scalar dBTdc12_z = 0;

		scalar dBTdc21_x = 0;
		scalar dBTdc21_y = 0;
		scalar dBTdc21_z = 0;

		scalar dBTdc22_x = 0;
		scalar dBTdc22_y = 0;
		scalar dBTdc22_z = 0;

		scalar C1x = sC1.data()[0];
		scalar C1y = sC1.data()[1];
		scalar C1z = sC1.data()[2];
		scalar C2x = sC2.data()[0];
		scalar C2y = sC2.data()[1];
		scalar C2z = sC2.data()[2];
		scalar C3x = sC3.data()[0];
		scalar C3y = sC3.data()[1];
		scalar C3z = sC3.data()[2];

		scalar vx = 2.0 * (alpha * C1x + C3x);
		scalar vy = 2.0 * (alpha * C1y + C3y);
		scalar vz = 2.0 * (alpha * C1z + C3z);

		scalar dvxdc11_x = 2.0 * (C1x * dALdc11_x + C2x * dBTdc11_x + alpha);
		scalar dvxdc11_y = 2.0 * (C1x * dALdc11_y + C2x * dBTdc11_y);
		scalar dvxdc11_z = 2.0 * (C1x * dALdc11_z + C2x * dBTdc11_z);

		scalar dvxdc12_x = 2.0 * (C1x * dALdc12_x + C2x * dBTdc12_x - alpha + 1);
		scalar dvxdc12_y = 2.0 * (C1x * dALdc12_y + C2x * dBTdc12_y);
		scalar dvxdc12_z = 2.0 * (C1x * dALdc12_z + C2x * dBTdc12_z);

		scalar dvxdc21_x = 0;
		scalar dvxdc21_y = 0;
		scalar dvxdc21_z = 0;

		scalar dvxdc22_x = 2.0 * (C1x * dALdc22_x + C2x * dBTdc22_x + beta - 1);
		scalar dvxdc22_y = 2.0 * (C1x * dALdc22_y + C2x * dBTdc22_y);
		scalar dvxdc22_z = 2.0 * (C1x * dALdc22_z + C2x * dBTdc22_z);
		//
		scalar dvydc11_x = 2.0 * (C1y * dALdc11_x + C2y * dBTdc11_x);
		scalar dvydc11_y = 2.0 * (C1y * dALdc11_y + C2y * dBTdc11_y + alpha);
		scalar dvydc11_z = 2.0 * (C1y * dALdc11_z + C2y * dBTdc11_z);

		scalar dvydc12_x = 2.0 * (C1y * dALdc12_x + C2y * dBTdc12_x);
		scalar dvydc12_y = 2.0 * (C1y * dALdc12_y + C2y * dBTdc12_y - alpha + 1);
		scalar dvydc12_z = 2.0 * (C1y * dALdc12_z + C2y * dBTdc12_z);

		scalar dvydc21_x = 0;
		scalar dvydc21_y = 0;
		scalar dvydc21_z = 0;

		scalar dvydc22_x = 2.0 * (C1y * dALdc22_x + C2y * dBTdc22_x);
		scalar dvydc22_y = 2.0 * (C1y * dALdc22_y + C2y * dBTdc22_y + beta - 1);
		scalar dvydc22_z = 2.0 * (C1y * dALdc22_z + C2y * dBTdc22_z);
		//
		scalar dvzdc11_x = 2.0 * (C1z * dALdc11_x + C2z * dBTdc11_x);
		scalar dvzdc11_y = 2.0 * (C1z * dALdc11_y + C2z * dBTdc11_y);
		scalar dvzdc11_z = 2.0 * (C1z * dALdc11_z + C2z * dBTdc11_z + alpha);

		scalar dvzdc12_x = 2.0 * (C1z * dALdc12_x + C2z * dBTdc12_x);
		scalar dvzdc12_y = 2.0 * (C1z * dALdc12_y + C2z * dBTdc12_y);
		scalar dvzdc12_z = 2.0 * (C1z * dALdc12_z + C2z * dBTdc12_z - alpha + 1);

		scalar dvzdc21_x = 0;
		scalar dvzdc21_y = 0;
		scalar dvzdc21_z = 0;

		scalar dvzdc22_x = 2.0 * (C1z * dALdc22_x + C2z * dBTdc22_x);
		scalar dvzdc22_y = 2.0 * (C1z * dALdc22_y + C2z * dBTdc22_y);
		scalar dvzdc22_z = 2.0 * (C1z * dALdc22_z + C2z * dBTdc22_z + beta - 1);

		hessian.setZero();

		hessian << vx * dALdc11_x + alpha * dvxdc11_x, vy * dALdc11_x + alpha * dvydc11_x, vz * dALdc11_x + alpha * dvzdc11_x, vx * -dALdc11_x + (1 - alpha) * dvxdc11_x, vy * -dALdc11_x + (1 - alpha) * dvydc11_x, vz * -dALdc11_x + (1 - alpha) * dvzdc11_x, vx * -dBTdc11_x - beta * dvxdc11_x, vy * -dBTdc11_x - beta * dvydc11_x, vz * -dBTdc11_x - beta * dvzdc11_x, vx * dBTdc11_x + (beta - 1.0) * dvxdc11_x, vy * dBTdc11_x + (beta - 1.0) * dvydc11_x, vz * dBTdc11_x + (beta - 1.0) * dvzdc11_x,
			vx * dALdc11_y + alpha * dvxdc11_y, vy * dALdc11_y + alpha * dvydc11_y, vz * dALdc11_y + alpha * dvzdc11_y, vx * -dALdc11_y + (1 - alpha) * dvxdc11_y, vy * -dALdc11_y + (1 - alpha) * dvydc11_y, vz * -dALdc11_y + (1 - alpha) * dvzdc11_y, vx * -dBTdc11_y - beta * dvxdc11_y, vy * -dBTdc11_y - beta * dvydc11_y, vz * -dBTdc11_y - beta * dvzdc11_y, vx * dBTdc11_y + (beta - 1.0) * dvxdc11_y, vy * dBTdc11_y + (beta - 1.0) * dvydc11_y, vz * dBTdc11_y + (beta - 1.0) * dvzdc11_y,
			vx * dALdc11_z + alpha * dvxdc11_z, vy * dALdc11_z + alpha * dvydc11_z, vz * dALdc11_z + alpha * dvzdc11_z, vx * -dALdc11_z + (1 - alpha) * dvxdc11_z, vy * -dALdc11_z + (1 - alpha) * dvydc11_z, vz * -dALdc11_z + (1 - alpha) * dvzdc11_z, vx * -dBTdc11_z - beta * dvxdc11_z, vy * -dBTdc11_z - beta * dvydc11_z, vz * -dBTdc11_z - beta * dvzdc11_z, vx * dBTdc11_z + (beta - 1.0) * dvxdc11_z, vy * dBTdc11_z + (beta - 1.0) * dvydc11_z, vz * dBTdc11_z + (beta - 1.0) * dvzdc11_z,
			vx * dALdc12_x + alpha * dvxdc12_x, vy * dALdc12_x + alpha * dvydc12_x, vz * dALdc12_x + alpha * dvzdc12_x, vx * -dALdc12_x + (1 - alpha) * dvxdc12_x, vy * -dALdc12_x + (1 - alpha) * dvydc12_x, vz * -dALdc12_x + (1 - alpha) * dvzdc12_x, vx * -dBTdc12_x - beta * dvxdc12_x, vy * -dBTdc12_x - beta * dvydc12_x, vz * -dBTdc12_x - beta * dvzdc12_x, vx * dBTdc12_x + (beta - 1.0) * dvxdc12_x, vy * dBTdc12_x + (beta - 1.0) * dvydc12_x, vz * dBTdc12_x + (beta - 1.0) * dvzdc12_x,
			vx * dALdc12_y + alpha * dvxdc12_y, vy * dALdc12_y + alpha * dvydc12_y, vz * dALdc12_y + alpha * dvzdc12_y, vx * -dALdc12_y + (1 - alpha) * dvxdc12_y, vy * -dALdc12_y + (1 - alpha) * dvydc12_y, vz * -dALdc12_y + (1 - alpha) * dvzdc12_y, vx * -dBTdc12_y - beta * dvxdc12_y, vy * -dBTdc12_y - beta * dvydc12_y, vz * -dBTdc12_y - beta * dvzdc12_y, vx * dBTdc12_y + (beta - 1.0) * dvxdc12_y, vy * dBTdc12_y + (beta - 1.0) * dvydc12_y, vz * dBTdc12_y + (beta - 1.0) * dvzdc12_y,
			vx * dALdc12_z + alpha * dvxdc12_z, vy * dALdc12_z + alpha * dvydc12_z, vz * dALdc12_z + alpha * dvzdc12_z, vx * -dALdc12_z + (1 - alpha) * dvxdc12_z, vy * -dALdc12_z + (1 - alpha) * dvydc12_z, vz * -dALdc12_z + (1 - alpha) * dvzdc12_z, vx * -dBTdc12_z - beta * dvxdc12_z, vy * -dBTdc12_z - beta * dvydc12_z, vz * -dBTdc12_z - beta * dvzdc12_z, vx * dBTdc12_z + (beta - 1.0) * dvxdc12_z, vy * dBTdc12_z + (beta - 1.0) * dvydc12_z, vz * dBTdc12_z + (beta - 1.0) * dvzdc12_z,
			vx * dALdc21_x + alpha * dvxdc21_x, vy * dALdc21_x + alpha * dvydc21_x, vz * dALdc21_x + alpha * dvzdc21_x, vx * -dALdc21_x + (1 - alpha) * dvxdc21_x, vy * -dALdc21_x + (1 - alpha) * dvydc21_x, vz * -dALdc21_x + (1 - alpha) * dvzdc21_x, vx * -dBTdc21_x - beta * dvxdc21_x, vy * -dBTdc21_x - beta * dvydc21_x, vz * -dBTdc21_x - beta * dvzdc21_x, vx * dBTdc21_x + (beta - 1.0) * dvxdc21_x, vy * dBTdc21_x + (beta - 1.0) * dvydc21_x, vz * dBTdc21_x + (beta - 1.0) * dvzdc21_x,
			vx * dALdc21_y + alpha * dvxdc21_y, vy * dALdc21_y + alpha * dvydc21_y, vz * dALdc21_y + alpha * dvzdc21_y, vx * -dALdc21_y + (1 - alpha) * dvxdc21_y, vy * -dALdc21_y + (1 - alpha) * dvydc21_y, vz * -dALdc21_y + (1 - alpha) * dvzdc21_y, vx * -dBTdc21_y - beta * dvxdc21_y, vy * -dBTdc21_y - beta * dvydc21_y, vz * -dBTdc21_y - beta * dvzdc21_y, vx * dBTdc21_y + (beta - 1.0) * dvxdc21_y, vy * dBTdc21_y + (beta - 1.0) * dvydc21_y, vz * dBTdc21_y + (beta - 1.0) * dvzdc21_y,
			vx * dALdc21_z + alpha * dvxdc21_z, vy * dALdc21_z + alpha * dvydc21_z, vz * dALdc21_z + alpha * dvzdc21_z, vx * -dALdc21_z + (1 - alpha) * dvxdc21_z, vy * -dALdc21_z + (1 - alpha) * dvydc21_z, vz * -dALdc21_z + (1 - alpha) * dvzdc21_z, vx * -dBTdc21_z - beta * dvxdc21_z, vy * -dBTdc21_z - beta * dvydc21_z, vz * -dBTdc21_z - beta * dvzdc21_z, vx * dBTdc21_z + (beta - 1.0) * dvxdc21_z, vy * dBTdc21_z + (beta - 1.0) * dvydc21_z, vz * dBTdc21_z + (beta - 1.0) * dvzdc21_z,
			vx * dALdc22_x + alpha * dvxdc22_x, vy * dALdc22_x + alpha * dvydc22_x, vz * dALdc22_x + alpha * dvzdc22_x, vx * -dALdc22_x + (1 - alpha) * dvxdc22_x, vy * -dALdc22_x + (1 - alpha) * dvydc22_x, vz * -dALdc22_x + (1 - alpha) * dvzdc22_x, vx * -dBTdc22_x - beta * dvxdc22_x, vy * -dBTdc22_x - beta * dvydc22_x, vz * -dBTdc22_x - beta * dvzdc22_x, vx * dBTdc22_x + (beta - 1.0) * dvxdc22_x, vy * dBTdc22_x + (beta - 1.0) * dvydc22_x, vz * dBTdc22_x + (beta - 1.0) * dvzdc22_x,
			vx * dALdc22_y + alpha * dvxdc22_y, vy * dALdc22_y + alpha * dvydc22_y, vz * dALdc22_y + alpha * dvzdc22_y, vx * -dALdc22_y + (1 - alpha) * dvxdc22_y, vy * -dALdc22_y + (1 - alpha) * dvydc22_y, vz * -dALdc22_y + (1 - alpha) * dvzdc22_y, vx * -dBTdc22_y - beta * dvxdc22_y, vy * -dBTdc22_y - beta * dvydc22_y, vz * -dBTdc22_y - beta * dvzdc22_y, vx * dBTdc22_y + (beta - 1.0) * dvxdc22_y, vy * dBTdc22_y + (beta - 1.0) * dvydc22_y, vz * dBTdc22_y + (beta - 1.0) * dvzdc22_y,
			vx * dALdc22_z + alpha * dvxdc22_z, vy * dALdc22_z + alpha * dvydc22_z, vz * dALdc22_z + alpha * dvzdc22_z, vx * -dALdc22_z + (1 - alpha) * dvxdc22_z, vy * -dALdc22_z + (1 - alpha) * dvydc22_z, vz * -dALdc22_z + (1 - alpha) * dvzdc22_z, vx * -dBTdc22_z - beta * dvxdc22_z, vy * -dBTdc22_z - beta * dvydc22_z, vz * -dBTdc22_z - beta * dvzdc22_z, vx * dBTdc22_z + (beta - 1.0) * dvxdc22_z, vy * dBTdc22_z + (beta - 1.0) * dvydc22_z, vz * dBTdc22_z + (beta - 1.0) * dvzdc22_z;

		hessian = hessian.transpose();
	}
	void MipcConeConeConstraint::betaIsZeroHessian1(mat12 &hessian)
	{
		scalar dAdc11_x = 2.0 * sC1.data()[0];
		scalar dAdc11_y = 2.0 * sC1.data()[1];
		scalar dAdc11_z = 2.0 * sC1.data()[2];
		scalar dAdc12_x = -2.0 * sC1.data()[0];
		scalar dAdc12_y = -2.0 * sC1.data()[1];
		scalar dAdc12_z = -2.0 * sC1.data()[2];

		scalar dBdc11_x = 2.0 * sC2.data()[0];
		scalar dBdc11_y = 2.0 * sC2.data()[1];
		scalar dBdc11_z = 2.0 * sC2.data()[2];
		scalar dBdc12_x = -2.0 * sC2.data()[0];
		scalar dBdc12_y = -2.0 * sC2.data()[1];
		scalar dBdc12_z = -2.0 * sC2.data()[2];
		scalar dBdc21_x = -2.0 * sC1.data()[0];
		scalar dBdc21_y = -2.0 * sC1.data()[1];
		scalar dBdc21_z = -2.0 * sC1.data()[2];
		scalar dBdc22_x = 2.0 * sC1.data()[0];
		scalar dBdc22_y = 2.0 * sC1.data()[1];
		scalar dBdc22_z = 2.0 * sC1.data()[2];

		scalar dCdc21_x = -2.0 * sC2.data()[0];
		scalar dCdc21_y = -2.0 * sC2.data()[1];
		scalar dCdc21_z = -2.0 * sC2.data()[2];
		scalar dCdc22_x = 2.0 * sC2.data()[0];
		scalar dCdc22_y = 2.0 * sC2.data()[1];
		scalar dCdc22_z = 2.0 * sC2.data()[2];

		scalar dDdc11_x = 2.0 * sC3.data()[0];
		scalar dDdc11_y = 2.0 * sC3.data()[1];
		scalar dDdc11_z = 2.0 * sC3.data()[2];
		scalar dDdc12_x = 2.0 * (sC1.data()[0] - sC3.data()[0]);
		scalar dDdc12_y = 2.0 * (sC1.data()[1] - sC3.data()[1]);
		scalar dDdc12_z = 2.0 * (sC1.data()[2] - sC3.data()[2]);
		scalar dDdc22_x = -2.0 * sC1.data()[0];
		scalar dDdc22_y = -2.0 * sC1.data()[1];
		scalar dDdc22_z = -2.0 * sC1.data()[2];

		scalar dEdc12_x = 2.0 * sC2.data()[0];
		scalar dEdc12_y = 2.0 * sC2.data()[1];
		scalar dEdc12_z = 2.0 * sC2.data()[2];
		scalar dEdc21_x = -2.0 * sC3.data()[0];
		scalar dEdc21_y = -2.0 * sC3.data()[1];
		scalar dEdc21_z = -2.0 * sC3.data()[2];
		scalar dEdc22_x = 2.0 * (sC3.data()[0] - sC2.data()[0]);
		scalar dEdc22_y = 2.0 * (sC3.data()[1] - sC2.data()[1]);
		scalar dEdc22_z = 2.0 * (sC3.data()[2] - sC2.data()[2]);

		scalar dFdc12_x = 2.0 * sC3.data()[0];
		scalar dFdc12_y = 2.0 * sC3.data()[1];
		scalar dFdc12_z = 2.0 * sC3.data()[2];
		scalar dFdc22 = -2.0 * sC3.data()[0];
		scalar dFdc22_y = -2.0 * sC3.data()[1];
		scalar dFd22_z = -2.0 * sC3.data()[2];

		//
		scalar q1 = 0.5 * D / (A * A);
		scalar q2 = -0.5 / A;

		scalar dALdc11_x = q1 * dAdc11_x + q2 * dDdc11_x;
		scalar dALdc11_y = q1 * dAdc11_y + q2 * dDdc11_y;
		scalar dALdc11_z = q1 * dAdc11_z + q2 * dDdc11_z;

		scalar dALdc12_x = q1 * dAdc12_x + q2 * dDdc12_x;
		scalar dALdc12_y = q1 * dAdc12_y + q2 * dDdc12_y;
		scalar dALdc12_z = q1 * dAdc12_z + q2 * dDdc12_z;

		scalar dALdc21_x = 0;
		scalar dALdc21_y = 0;
		scalar dALdc21_z = 0;

		scalar dALdc22_x = q2 * dDdc22_x;
		scalar dALdc22_y = q2 * dDdc22_y;
		scalar dALdc22_z = q2 * dDdc22_z;

		scalar dBTdc11_x = 0;
		scalar dBTdc11_y = 0;
		scalar dBTdc11_z = 0;

		scalar dBTdc12_x = 0;
		scalar dBTdc12_y = 0;
		scalar dBTdc12_z = 0;

		scalar dBTdc21_x = 0;
		scalar dBTdc21_y = 0;
		scalar dBTdc21_z = 0;

		scalar dBTdc22_x = 0;
		scalar dBTdc22_y = 0;
		scalar dBTdc22_z = 0;

		scalar C1x = sC1.data()[0];
		scalar C1y = sC1.data()[1];
		scalar C1z = sC1.data()[2];
		scalar C2x = sC2.data()[0];
		scalar C2y = sC2.data()[1];
		scalar C2z = sC2.data()[2];
		scalar C3x = sC3.data()[0];
		scalar C3y = sC3.data()[1];
		scalar C3z = sC3.data()[2];

		scalar vx = 2.0 * (alpha * C1x + C3x);
		scalar vy = 2.0 * (alpha * C1y + C3y);
		scalar vz = 2.0 * (alpha * C1z + C3z);

		scalar dvxdc11_x = 2.0 * (C1x * dALdc11_x + C2x * dBTdc11_x + alpha);
		scalar dvxdc11_y = 2.0 * (C1x * dALdc11_y + C2x * dBTdc11_y);
		scalar dvxdc11_z = 2.0 * (C1x * dALdc11_z + C2x * dBTdc11_z);

		scalar dvxdc12_x = 2.0 * (C1x * dALdc12_x + C2x * dBTdc12_x - alpha + 1);
		scalar dvxdc12_y = 2.0 * (C1x * dALdc12_y + C2x * dBTdc12_y);
		scalar dvxdc12_z = 2.0 * (C1x * dALdc12_z + C2x * dBTdc12_z);

		scalar dvxdc21_x = 0;
		scalar dvxdc21_y = 0;
		scalar dvxdc21_z = 0;

		scalar dvxdc22_x = 2.0 * (C1x * dALdc22_x + C2x * dBTdc22_x + beta - 1);
		scalar dvxdc22_y = 2.0 * (C1x * dALdc22_y + C2x * dBTdc22_y);
		scalar dvxdc22_z = 2.0 * (C1x * dALdc22_z + C2x * dBTdc22_z);
		//
		scalar dvydc11_x = 2.0 * (C1y * dALdc11_x + C2y * dBTdc11_x);
		scalar dvydc11_y = 2.0 * (C1y * dALdc11_y + C2y * dBTdc11_y + alpha);
		scalar dvydc11_z = 2.0 * (C1y * dALdc11_z + C2y * dBTdc11_z);

		scalar dvydc12_x = 2.0 * (C1y * dALdc12_x + C2y * dBTdc12_x);
		scalar dvydc12_y = 2.0 * (C1y * dALdc12_y + C2y * dBTdc12_y - alpha + 1);
		scalar dvydc12_z = 2.0 * (C1y * dALdc12_z + C2y * dBTdc12_z);

		scalar dvydc21_x = 0;
		scalar dvydc21_y = 0;
		scalar dvydc21_z = 0;

		scalar dvydc22_x = 2.0 * (C1y * dALdc22_x + C2y * dBTdc22_x);
		scalar dvydc22_y = 2.0 * (C1y * dALdc22_y + C2y * dBTdc22_y + beta - 1);
		scalar dvydc22_z = 2.0 * (C1y * dALdc22_z + C2y * dBTdc22_z);
		//
		scalar dvzdc11_x = 2.0 * (C1z * dALdc11_x + C2z * dBTdc11_x);
		scalar dvzdc11_y = 2.0 * (C1z * dALdc11_y + C2z * dBTdc11_y);
		scalar dvzdc11_z = 2.0 * (C1z * dALdc11_z + C2z * dBTdc11_z + alpha);

		scalar dvzdc12_x = 2.0 * (C1z * dALdc12_x + C2z * dBTdc12_x);
		scalar dvzdc12_y = 2.0 * (C1z * dALdc12_y + C2z * dBTdc12_y);
		scalar dvzdc12_z = 2.0 * (C1z * dALdc12_z + C2z * dBTdc12_z - alpha + 1);

		scalar dvzdc21_x = 0;
		scalar dvzdc21_y = 0;
		scalar dvzdc21_z = 0;

		scalar dvzdc22_x = 2.0 * (C1z * dALdc22_x + C2z * dBTdc22_x);
		scalar dvzdc22_y = 2.0 * (C1z * dALdc22_y + C2z * dBTdc22_y);
		scalar dvzdc22_z = 2.0 * (C1z * dALdc22_z + C2z * dBTdc22_z + beta - 1);

		hessian.setZero();
		// f / c11x
		hessian.data()[0] = vx * dALdc11_x + alpha * dvxdc11_x;
		hessian.data()[1] = vy * dALdc11_x + alpha * dvydc11_x;
		hessian.data()[2] = vz * dALdc11_x + alpha * dvzdc11_x;

		hessian.data()[3] = vx * -dALdc11_x + (1 - alpha) * dvxdc11_x;
		hessian.data()[4] = vy * -dALdc11_x + (1 - alpha) * dvydc11_x;
		hessian.data()[5] = vz * -dALdc11_x + (1 - alpha) * dvzdc11_x;

		hessian.data()[6] = vx * -dBTdc11_x - beta * dvxdc11_x;
		hessian.data()[7] = vy * -dBTdc11_x - beta * dvydc11_x;
		hessian.data()[8] = vz * -dBTdc11_x - beta * dvzdc11_x;

		hessian.data()[9] = vx * dBTdc11_x + (beta - 1.0) * dvxdc11_x;
		hessian.data()[10] = vy * dBTdc11_x + (beta - 1.0) * dvydc11_x;
		hessian.data()[11] = vz * dBTdc11_x + (beta - 1.0) * dvzdc11_x;
		// f / c11y
		hessian.data()[12] = vx * dALdc11_y + alpha * dvxdc11_y;
		hessian.data()[13] = vy * dALdc11_y + alpha * dvydc11_y;
		hessian.data()[14] = vz * dALdc11_y + alpha * dvzdc11_y;

		hessian.data()[15] = vx * -dALdc11_y + (1 - alpha) * dvxdc11_y;
		hessian.data()[16] = vy * -dALdc11_y + (1 - alpha) * dvydc11_y;
		hessian.data()[17] = vz * -dALdc11_y + (1 - alpha) * dvzdc11_y;

		hessian.data()[18] = vx * -dBTdc11_y - beta * dvxdc11_y;
		hessian.data()[19] = vy * -dBTdc11_y - beta * dvydc11_y;
		hessian.data()[20] = vz * -dBTdc11_y - beta * dvzdc11_y;

		hessian.data()[21] = vx * dBTdc11_y + (beta - 1.0) * dvxdc11_y;
		hessian.data()[22] = vy * dBTdc11_y + (beta - 1.0) * dvydc11_y;
		hessian.data()[23] = vz * dBTdc11_y + (beta - 1.0) * dvzdc11_y;
		// f / c11z
		hessian.data()[24] = vx * dALdc11_z + alpha * dvxdc11_z;
		hessian.data()[25] = vy * dALdc11_z + alpha * dvydc11_z;
		hessian.data()[26] = vz * dALdc11_z + alpha * dvzdc11_z;

		hessian.data()[27] = vx * -dALdc11_z + (1 - alpha) * dvxdc11_z;
		hessian.data()[28] = vy * -dALdc11_z + (1 - alpha) * dvydc11_z;
		hessian.data()[29] = vz * -dALdc11_z + (1 - alpha) * dvzdc11_z;

		hessian.data()[30] = vx * -dBTdc11_z - beta * dvxdc11_z;
		hessian.data()[31] = vy * -dBTdc11_z - beta * dvydc11_z;
		hessian.data()[32] = vz * -dBTdc11_z - beta * dvzdc11_z;

		hessian.data()[33] = vx * dBTdc11_z + (beta - 1.0) * dvxdc11_z;
		hessian.data()[34] = vy * dBTdc11_z + (beta - 1.0) * dvydc11_z;
		hessian.data()[35] = vz * dBTdc11_z + (beta - 1.0) * dvzdc11_z;

		// f / c12x
		hessian.data()[36] = vx * dALdc12_x + alpha * dvxdc12_x;
		hessian.data()[37] = vy * dALdc12_x + alpha * dvydc12_x;
		hessian.data()[38] = vz * dALdc12_x + alpha * dvzdc12_x;

		hessian.data()[39] = vx * -dALdc12_x + (1 - alpha) * dvxdc12_x;
		hessian.data()[40] = vy * -dALdc12_x + (1 - alpha) * dvydc12_x;
		hessian.data()[41] = vz * -dALdc12_x + (1 - alpha) * dvzdc12_x;

		hessian.data()[42] = vx * -dBTdc12_x - beta * dvxdc12_x;
		hessian.data()[43] = vy * -dBTdc12_x - beta * dvydc12_x;
		hessian.data()[44] = vz * -dBTdc12_x - beta * dvzdc12_x;

		hessian.data()[45] = vx * dBTdc12_x + (beta - 1.0) * dvxdc12_x;
		hessian.data()[46] = vy * dBTdc12_x + (beta - 1.0) * dvydc12_x;
		hessian.data()[47] = vz * dBTdc12_x + (beta - 1.0) * dvzdc12_x;
		// f / c12y
		hessian.data()[48] = vx * dALdc12_y + alpha * dvxdc12_y;
		hessian.data()[49] = vy * dALdc12_y + alpha * dvydc12_y;
		hessian.data()[50] = vz * dALdc12_y + alpha * dvzdc12_y;

		hessian.data()[51] = vx * -dALdc12_y + (1 - alpha) * dvxdc12_y;
		hessian.data()[52] = vy * -dALdc12_y + (1 - alpha) * dvydc12_y;
		hessian.data()[53] = vz * -dALdc12_y + (1 - alpha) * dvzdc12_y;

		hessian.data()[54] = vx * -dBTdc12_y - beta * dvxdc12_y;
		hessian.data()[55] = vy * -dBTdc12_y - beta * dvydc12_y;
		hessian.data()[56] = vz * -dBTdc12_y - beta * dvzdc12_y;

		hessian.data()[57] = vx * dBTdc12_y + (beta - 1.0) * dvxdc12_y;
		hessian.data()[58] = vy * dBTdc12_y + (beta - 1.0) * dvydc12_y;
		hessian.data()[59] = vz * dBTdc12_y + (beta - 1.0) * dvzdc12_y;
		// f / c12z
		hessian.data()[60] = vx * dALdc12_z + alpha * dvxdc12_z;
		hessian.data()[61] = vy * dALdc12_z + alpha * dvydc12_z;
		hessian.data()[62] = vz * dALdc12_z + alpha * dvzdc12_z;

		hessian.data()[63] = vx * -dALdc12_z + (1 - alpha) * dvxdc12_z;
		hessian.data()[64] = vy * -dALdc12_z + (1 - alpha) * dvydc12_z;
		hessian.data()[65] = vz * -dALdc12_z + (1 - alpha) * dvzdc12_z;

		hessian.data()[66] = vx * -dBTdc12_z - beta * dvxdc12_z;
		hessian.data()[67] = vy * -dBTdc12_z - beta * dvydc12_z;
		hessian.data()[68] = vz * -dBTdc12_z - beta * dvzdc12_z;

		hessian.data()[69] = vx * dBTdc12_z + (beta - 1.0) * dvxdc12_z;
		hessian.data()[70] = vy * dBTdc12_z + (beta - 1.0) * dvydc12_z;
		hessian.data()[71] = vz * dBTdc12_z + (beta - 1.0) * dvzdc12_z;

		// f / c21x
		hessian.data()[72] = vx * dALdc21_x + alpha * dvxdc21_x;
		hessian.data()[73] = vy * dALdc21_x + alpha * dvydc21_x;
		hessian.data()[74] = vz * dALdc21_x + alpha * dvzdc21_x;

		hessian.data()[75] = vx * -dALdc21_x + (1 - alpha) * dvxdc21_x;
		hessian.data()[76] = vy * -dALdc21_x + (1 - alpha) * dvydc21_x;
		hessian.data()[77] = vz * -dALdc21_x + (1 - alpha) * dvzdc21_x;

		hessian.data()[78] = vx * -dBTdc21_x - beta * dvxdc21_x;
		hessian.data()[79] = vy * -dBTdc21_x - beta * dvydc21_x;
		hessian.data()[80] = vz * -dBTdc21_x - beta * dvzdc21_x;

		hessian.data()[81] = vx * dBTdc21_x + (beta - 1.0) * dvxdc21_x;
		hessian.data()[82] = vy * dBTdc21_x + (beta - 1.0) * dvydc21_x;
		hessian.data()[83] = vz * dBTdc21_x + (beta - 1.0) * dvzdc21_x;
		// f / c21y
		hessian.data()[84] = vx * dALdc21_y + alpha * dvxdc21_y;
		hessian.data()[85] = vy * dALdc21_y + alpha * dvydc21_y;
		hessian.data()[86] = vz * dALdc21_y + alpha * dvzdc21_y;

		hessian.data()[87] = vx * -dALdc21_y + (1 - alpha) * dvxdc21_y;
		hessian.data()[88] = vy * -dALdc21_y + (1 - alpha) * dvydc21_y;
		hessian.data()[89] = vz * -dALdc21_y + (1 - alpha) * dvzdc21_y;

		hessian.data()[90] = vx * -dBTdc21_y - beta * dvxdc21_y;
		hessian.data()[91] = vy * -dBTdc21_y - beta * dvydc21_y;
		hessian.data()[92] = vz * -dBTdc21_y - beta * dvzdc21_y;

		hessian.data()[93] = vx * dBTdc21_y + (beta - 1.0) * dvxdc21_y;
		hessian.data()[94] = vy * dBTdc21_y + (beta - 1.0) * dvydc21_y;
		hessian.data()[95] = vz * dBTdc21_y + (beta - 1.0) * dvzdc21_y;
		// f / c21z
		hessian.data()[96] = vx * dALdc21_z + alpha * dvxdc21_z;
		hessian.data()[97] = vy * dALdc21_z + alpha * dvydc21_z;
		hessian.data()[98] = vz * dALdc21_z + alpha * dvzdc21_z;

		hessian.data()[99] = vx * -dALdc21_z + (1 - alpha) * dvxdc21_z;
		hessian.data()[100] = vy * -dALdc21_z + (1 - alpha) * dvydc21_z;
		hessian.data()[101] = vz * -dALdc21_z + (1 - alpha) * dvzdc21_z;

		hessian.data()[102] = vx * -dBTdc21_z - beta * dvxdc21_z;
		hessian.data()[103] = vy * -dBTdc21_z - beta * dvydc21_z;
		hessian.data()[104] = vz * -dBTdc21_z - beta * dvzdc21_z;

		hessian.data()[105] = vx * dBTdc21_z + (beta - 1.0) * dvxdc21_z;
		hessian.data()[106] = vy * dBTdc21_z + (beta - 1.0) * dvydc21_z;
		hessian.data()[107] = vz * dBTdc21_z + (beta - 1.0) * dvzdc21_z;

		// f / c22x
		hessian.data()[108] = vx * dALdc22_x + alpha * dvxdc22_x;
		hessian.data()[109] = vy * dALdc22_x + alpha * dvydc22_x;
		hessian.data()[110] = vz * dALdc22_x + alpha * dvzdc22_x;

		hessian.data()[111] = vx * -dALdc22_x + (1 - alpha) * dvxdc22_x;
		hessian.data()[112] = vy * -dALdc22_x + (1 - alpha) * dvydc22_x;
		hessian.data()[113] = vz * -dALdc22_x + (1 - alpha) * dvzdc22_x;

		hessian.data()[114] = vx * -dBTdc22_x - beta * dvxdc22_x;
		hessian.data()[115] = vy * -dBTdc22_x - beta * dvydc22_x;
		hessian.data()[116] = vz * -dBTdc22_x - beta * dvzdc22_x;

		hessian.data()[117] = vx * dBTdc22_x + (beta - 1.0) * dvxdc22_x;
		hessian.data()[118] = vy * dBTdc22_x + (beta - 1.0) * dvydc22_x;
		hessian.data()[119] = vz * dBTdc22_x + (beta - 1.0) * dvzdc22_x;
		// f / c22y
		hessian.data()[120] = vx * dALdc22_y + alpha * dvxdc22_y;
		hessian.data()[121] = vy * dALdc22_y + alpha * dvydc22_y;
		hessian.data()[122] = vz * dALdc22_y + alpha * dvzdc22_y;

		hessian.data()[123] = vx * -dALdc22_y + (1 - alpha) * dvxdc22_y;
		hessian.data()[124] = vy * -dALdc22_y + (1 - alpha) * dvydc22_y;
		hessian.data()[125] = vz * -dALdc22_y + (1 - alpha) * dvzdc22_y;

		hessian.data()[126] = vx * -dBTdc22_y - beta * dvxdc22_y;
		hessian.data()[127] = vy * -dBTdc22_y - beta * dvydc22_y;
		hessian.data()[128] = vz * -dBTdc22_y - beta * dvzdc22_y;

		hessian.data()[129] = vx * dBTdc22_y + (beta - 1.0) * dvxdc22_y;
		hessian.data()[130] = vy * dBTdc22_y + (beta - 1.0) * dvydc22_y;
		hessian.data()[131] = vz * dBTdc22_y + (beta - 1.0) * dvzdc22_y;
		// f / c22z
		hessian.data()[132] = vx * dALdc22_z + alpha * dvxdc22_z;
		hessian.data()[133] = vy * dALdc22_z + alpha * dvydc22_z;
		hessian.data()[134] = vz * dALdc22_z + alpha * dvzdc22_z;

		hessian.data()[135] = vx * -dALdc22_z + (1 - alpha) * dvxdc22_z;
		hessian.data()[136] = vy * -dALdc22_z + (1 - alpha) * dvydc22_z;
		hessian.data()[137] = vz * -dALdc22_z + (1 - alpha) * dvzdc22_z;

		hessian.data()[138] = vx * -dBTdc22_z - beta * dvxdc22_z;
		hessian.data()[139] = vy * -dBTdc22_z - beta * dvydc22_z;
		hessian.data()[140] = vz * -dBTdc22_z - beta * dvzdc22_z;

		hessian.data()[141] = vx * dBTdc22_z + (beta - 1.0) * dvxdc22_z;
		hessian.data()[142] = vy * dBTdc22_z + (beta - 1.0) * dvydc22_z;
		hessian.data()[143] = vz * dBTdc22_z + (beta - 1.0) * dvzdc22_z;
	}

	void MipcConeConeConstraint::betaIsOneHessian(mat12 &hessian)
	{
		scalar dAdc11_x = 2.0 * sC1.data()[0];
		scalar dAdc11_y = 2.0 * sC1.data()[1];
		scalar dAdc11_z = 2.0 * sC1.data()[2];
		scalar dAdc12_x = -2.0 * sC1.data()[0];
		scalar dAdc12_y = -2.0 * sC1.data()[1];
		scalar dAdc12_z = -2.0 * sC1.data()[2];

		scalar dBdc11_x = 2.0 * sC2.data()[0];
		scalar dBdc11_y = 2.0 * sC2.data()[1];
		scalar dBdc11_z = 2.0 * sC2.data()[2];
		scalar dBdc12_x = -2.0 * sC2.data()[0];
		scalar dBdc12_y = -2.0 * sC2.data()[1];
		scalar dBdc12_z = -2.0 * sC2.data()[2];
		scalar dBdc21_x = -2.0 * sC1.data()[0];
		scalar dBdc21_y = -2.0 * sC1.data()[1];
		scalar dBdc21_z = -2.0 * sC1.data()[2];
		scalar dBdc22_x = 2.0 * sC1.data()[0];
		scalar dBdc22_y = 2.0 * sC1.data()[1];
		scalar dBdc22_z = 2.0 * sC1.data()[2];

		scalar dCdc21_x = -2.0 * sC2.data()[0];
		scalar dCdc21_y = -2.0 * sC2.data()[1];
		scalar dCdc21_z = -2.0 * sC2.data()[2];
		scalar dCdc22_x = 2.0 * sC2.data()[0];
		scalar dCdc22_y = 2.0 * sC2.data()[1];
		scalar dCdc22_z = 2.0 * sC2.data()[2];

		scalar dDdc11_x = 2.0 * sC3.data()[0];
		scalar dDdc11_y = 2.0 * sC3.data()[1];
		scalar dDdc11_z = 2.0 * sC3.data()[2];
		scalar dDdc12_x = 2.0 * (sC1.data()[0] - sC3.data()[0]);
		scalar dDdc12_y = 2.0 * (sC1.data()[1] - sC3.data()[1]);
		scalar dDdc12_z = 2.0 * (sC1.data()[2] - sC3.data()[2]);
		scalar dDdc22_x = -2.0 * sC1.data()[0];
		scalar dDdc22_y = -2.0 * sC1.data()[1];
		scalar dDdc22_z = -2.0 * sC1.data()[2];

		scalar dEdc12_x = 2.0 * sC2.data()[0];
		scalar dEdc12_y = 2.0 * sC2.data()[1];
		scalar dEdc12_z = 2.0 * sC2.data()[2];
		scalar dEdc21_x = -2.0 * sC3.data()[0];
		scalar dEdc21_y = -2.0 * sC3.data()[1];
		scalar dEdc21_z = -2.0 * sC3.data()[2];
		scalar dEdc22_x = 2.0 * (sC3.data()[0] - sC2.data()[0]);
		scalar dEdc22_y = 2.0 * (sC3.data()[1] - sC2.data()[1]);
		scalar dEdc22_z = 2.0 * (sC3.data()[2] - sC2.data()[2]);

		scalar dFdc12_x = 2.0 * sC3.data()[0];
		scalar dFdc12_y = 2.0 * sC3.data()[1];
		scalar dFdc12_z = 2.0 * sC3.data()[2];
		scalar dFdc22 = -2.0 * sC3.data()[0];
		scalar dFdc22_y = -2.0 * sC3.data()[1];
		scalar dFd22_z = -2.0 * sC3.data()[2];

		//
		scalar q1 = 0.5 * (B + D) / (A * A);
		scalar q2 = -0.5 / A;
		scalar dALdc11_x = q1 * dAdc11_x + q2 * (dBdc11_x + dDdc11_x);
		scalar dALdc11_y = q1 * dAdc11_y + q2 * (dBdc11_y + dDdc11_y);
		scalar dALdc11_z = q1 * dAdc11_z + q2 * (dBdc11_z + dDdc11_z);

		scalar dALdc12_x = q1 * dAdc12_x + q2 * (dBdc12_x + dDdc12_x);
		scalar dALdc12_y = q1 * dAdc12_y + q2 * (dBdc12_y + dDdc12_y);
		scalar dALdc12_z = q1 * dAdc12_z + q2 * (dBdc12_z + dDdc12_z);

		scalar dALdc21_x = q2 * dBdc21_x;
		scalar dALdc21_y = q2 * dBdc21_y;
		scalar dALdc21_z = q2 * dBdc21_z;

		scalar dALdc22_x = q2 * (dBdc22_x + dDdc22_x);
		scalar dALdc22_y = q2 * (dBdc22_x + dDdc22_y);
		scalar dALdc22_z = q2 * (dBdc22_x + dDdc22_z);

		scalar dBTdc11_x = 0;
		scalar dBTdc11_y = 0;
		scalar dBTdc11_z = 0;

		scalar dBTdc12_x = 0;
		scalar dBTdc12_y = 0;
		scalar dBTdc12_z = 0;

		scalar dBTdc21_x = 0;
		scalar dBTdc21_y = 0;
		scalar dBTdc21_z = 0;

		scalar dBTdc22_x = 0;
		scalar dBTdc22_y = 0;
		scalar dBTdc22_z = 0;

		scalar C1x = sC1.data()[0];
		scalar C1y = sC1.data()[1];
		scalar C1z = sC1.data()[2];
		scalar C2x = sC2.data()[0];
		scalar C2y = sC2.data()[1];
		scalar C2z = sC2.data()[2];
		scalar C3x = sC3.data()[0];
		scalar C3y = sC3.data()[1];
		scalar C3z = sC3.data()[2];

		scalar vx = 2.0 * (alpha * C1x + beta * C2x + C3x);
		scalar vy = 2.0 * (alpha * C1y + beta * C2y + C3y);
		scalar vz = 2.0 * (alpha * C1z + beta * C2z + C3z);

		scalar dvxdc11_x = 2.0 * (C1x * dALdc11_x + C2x * dBTdc11_x + alpha);
		scalar dvxdc11_y = 2.0 * (C1x * dALdc11_y + C2x * dBTdc11_y);
		scalar dvxdc11_z = 2.0 * (C1x * dALdc11_z + C2x * dBTdc11_z);

		scalar dvxdc12_x = 2.0 * (C1x * dALdc12_x + C2x * dBTdc12_x - alpha + 1);
		scalar dvxdc12_y = 2.0 * (C1x * dALdc12_y + C2x * dBTdc12_y);
		scalar dvxdc12_z = 2.0 * (C1x * dALdc12_z + C2x * dBTdc12_z);

		scalar dvxdc21_x = 2.0 * (C1x * dALdc21_x + C2x * dBTdc21_x - beta);
		scalar dvxdc21_y = 2.0 * (C1x * dALdc21_y + C2x * dBTdc21_y);
		scalar dvxdc21_z = 2.0 * (C1x * dALdc21_z + C2x * dBTdc21_z);

		scalar dvxdc22_x = 2.0 * (C1x * dALdc22_x + C2x * dBTdc22_x + beta - 1);
		scalar dvxdc22_y = 2.0 * (C1x * dALdc22_y + C2x * dBTdc22_y);
		scalar dvxdc22_z = 2.0 * (C1x * dALdc22_z + C2x * dBTdc22_z);
		//
		scalar dvydc11_x = 2.0 * (C1y * dALdc11_x + C2y * dBTdc11_x);
		scalar dvydc11_y = 2.0 * (C1y * dALdc11_y + C2y * dBTdc11_y + alpha);
		scalar dvydc11_z = 2.0 * (C1y * dALdc11_z + C2y * dBTdc11_z);

		scalar dvydc12_x = 2.0 * (C1y * dALdc12_x + C2y * dBTdc12_x);
		scalar dvydc12_y = 2.0 * (C1y * dALdc12_y + C2y * dBTdc12_y - alpha + 1);
		scalar dvydc12_z = 2.0 * (C1y * dALdc12_z + C2y * dBTdc12_z);

		scalar dvydc21_x = 2.0 * (C1y * dALdc21_x + C2y * dBTdc21_x);
		scalar dvydc21_y = 2.0 * (C1y * dALdc21_y + C2y * dBTdc21_y - beta);
		scalar dvydc21_z = 2.0 * (C1y * dALdc21_z + C2y * dBTdc21_z);

		scalar dvydc22_x = 2.0 * (C1y * dALdc22_x + C2y * dBTdc22_x);
		scalar dvydc22_y = 2.0 * (C1y * dALdc22_y + C2y * dBTdc22_y + beta - 1);
		scalar dvydc22_z = 2.0 * (C1y * dALdc22_z + C2y * dBTdc22_z);
		//
		scalar dvzdc11_x = 2.0 * (C1z * dALdc11_x + C2z * dBTdc11_x);
		scalar dvzdc11_y = 2.0 * (C1z * dALdc11_y + C2z * dBTdc11_y);
		scalar dvzdc11_z = 2.0 * (C1z * dALdc11_z + C2z * dBTdc11_z + alpha);

		scalar dvzdc12_x = 2.0 * (C1z * dALdc12_x + C2z * dBTdc12_x);
		scalar dvzdc12_y = 2.0 * (C1z * dALdc12_y + C2z * dBTdc12_y);
		scalar dvzdc12_z = 2.0 * (C1z * dALdc12_z + C2z * dBTdc12_z - alpha + 1);

		scalar dvzdc21_x = 2.0 * (C1z * dALdc21_x + C2z * dBTdc21_x);
		scalar dvzdc21_y = 2.0 * (C1z * dALdc21_y + C2z * dBTdc21_y);
		scalar dvzdc21_z = 2.0 * (C1z * dALdc21_z + C2z * dBTdc21_z - beta);

		scalar dvzdc22_x = 2.0 * (C1z * dALdc22_x + C2z * dBTdc22_x);
		scalar dvzdc22_y = 2.0 * (C1z * dALdc22_y + C2z * dBTdc22_y);
		scalar dvzdc22_z = 2.0 * (C1z * dALdc22_z + C2z * dBTdc22_z + beta - 1);

		hessian.setZero();
		hessian << vx * dALdc11_x + alpha * dvxdc11_x, vy * dALdc11_x + alpha * dvydc11_x, vz * dALdc11_x + alpha * dvzdc11_x, vx * -dALdc11_x + (1 - alpha) * dvxdc11_x, vy * -dALdc11_x + (1 - alpha) * dvydc11_x, vz * -dALdc11_x + (1 - alpha) * dvzdc11_x, vx * -dBTdc11_x - beta * dvxdc11_x, vy * -dBTdc11_x - beta * dvydc11_x, vz * -dBTdc11_x - beta * dvzdc11_x, vx * dBTdc11_x + (beta - 1.0) * dvxdc11_x, vy * dBTdc11_x + (beta - 1.0) * dvydc11_x, vz * dBTdc11_x + (beta - 1.0) * dvzdc11_x,
			vx * dALdc11_y + alpha * dvxdc11_y, vy * dALdc11_y + alpha * dvydc11_y, vz * dALdc11_y + alpha * dvzdc11_y, vx * -dALdc11_y + (1 - alpha) * dvxdc11_y, vy * -dALdc11_y + (1 - alpha) * dvydc11_y, vz * -dALdc11_y + (1 - alpha) * dvzdc11_y, vx * -dBTdc11_y - beta * dvxdc11_y, vy * -dBTdc11_y - beta * dvydc11_y, vz * -dBTdc11_y - beta * dvzdc11_y, vx * dBTdc11_y + (beta - 1.0) * dvxdc11_y, vy * dBTdc11_y + (beta - 1.0) * dvydc11_y, vz * dBTdc11_y + (beta - 1.0) * dvzdc11_y,
			vx * dALdc11_z + alpha * dvxdc11_z, vy * dALdc11_z + alpha * dvydc11_z, vz * dALdc11_z + alpha * dvzdc11_z, vx * -dALdc11_z + (1 - alpha) * dvxdc11_z, vy * -dALdc11_z + (1 - alpha) * dvydc11_z, vz * -dALdc11_z + (1 - alpha) * dvzdc11_z, vx * -dBTdc11_z - beta * dvxdc11_z, vy * -dBTdc11_z - beta * dvydc11_z, vz * -dBTdc11_z - beta * dvzdc11_z, vx * dBTdc11_z + (beta - 1.0) * dvxdc11_z, vy * dBTdc11_z + (beta - 1.0) * dvydc11_z, vz * dBTdc11_z + (beta - 1.0) * dvzdc11_z,
			vx * dALdc12_x + alpha * dvxdc12_x, vy * dALdc12_x + alpha * dvydc12_x, vz * dALdc12_x + alpha * dvzdc12_x, vx * -dALdc12_x + (1 - alpha) * dvxdc12_x, vy * -dALdc12_x + (1 - alpha) * dvydc12_x, vz * -dALdc12_x + (1 - alpha) * dvzdc12_x, vx * -dBTdc12_x - beta * dvxdc12_x, vy * -dBTdc12_x - beta * dvydc12_x, vz * -dBTdc12_x - beta * dvzdc12_x, vx * dBTdc12_x + (beta - 1.0) * dvxdc12_x, vy * dBTdc12_x + (beta - 1.0) * dvydc12_x, vz * dBTdc12_x + (beta - 1.0) * dvzdc12_x,
			vx * dALdc12_y + alpha * dvxdc12_y, vy * dALdc12_y + alpha * dvydc12_y, vz * dALdc12_y + alpha * dvzdc12_y, vx * -dALdc12_y + (1 - alpha) * dvxdc12_y, vy * -dALdc12_y + (1 - alpha) * dvydc12_y, vz * -dALdc12_y + (1 - alpha) * dvzdc12_y, vx * -dBTdc12_y - beta * dvxdc12_y, vy * -dBTdc12_y - beta * dvydc12_y, vz * -dBTdc12_y - beta * dvzdc12_y, vx * dBTdc12_y + (beta - 1.0) * dvxdc12_y, vy * dBTdc12_y + (beta - 1.0) * dvydc12_y, vz * dBTdc12_y + (beta - 1.0) * dvzdc12_y,
			vx * dALdc12_z + alpha * dvxdc12_z, vy * dALdc12_z + alpha * dvydc12_z, vz * dALdc12_z + alpha * dvzdc12_z, vx * -dALdc12_z + (1 - alpha) * dvxdc12_z, vy * -dALdc12_z + (1 - alpha) * dvydc12_z, vz * -dALdc12_z + (1 - alpha) * dvzdc12_z, vx * -dBTdc12_z - beta * dvxdc12_z, vy * -dBTdc12_z - beta * dvydc12_z, vz * -dBTdc12_z - beta * dvzdc12_z, vx * dBTdc12_z + (beta - 1.0) * dvxdc12_z, vy * dBTdc12_z + (beta - 1.0) * dvydc12_z, vz * dBTdc12_z + (beta - 1.0) * dvzdc12_z,
			vx * dALdc21_x + alpha * dvxdc21_x, vy * dALdc21_x + alpha * dvydc21_x, vz * dALdc21_x + alpha * dvzdc21_x, vx * -dALdc21_x + (1 - alpha) * dvxdc21_x, vy * -dALdc21_x + (1 - alpha) * dvydc21_x, vz * -dALdc21_x + (1 - alpha) * dvzdc21_x, vx * -dBTdc21_x - beta * dvxdc21_x, vy * -dBTdc21_x - beta * dvydc21_x, vz * -dBTdc21_x - beta * dvzdc21_x, vx * dBTdc21_x + (beta - 1.0) * dvxdc21_x, vy * dBTdc21_x + (beta - 1.0) * dvydc21_x, vz * dBTdc21_x + (beta - 1.0) * dvzdc21_x,
			vx * dALdc21_y + alpha * dvxdc21_y, vy * dALdc21_y + alpha * dvydc21_y, vz * dALdc21_y + alpha * dvzdc21_y, vx * -dALdc21_y + (1 - alpha) * dvxdc21_y, vy * -dALdc21_y + (1 - alpha) * dvydc21_y, vz * -dALdc21_y + (1 - alpha) * dvzdc21_y, vx * -dBTdc21_y - beta * dvxdc21_y, vy * -dBTdc21_y - beta * dvydc21_y, vz * -dBTdc21_y - beta * dvzdc21_y, vx * dBTdc21_y + (beta - 1.0) * dvxdc21_y, vy * dBTdc21_y + (beta - 1.0) * dvydc21_y, vz * dBTdc21_y + (beta - 1.0) * dvzdc21_y,
			vx * dALdc21_z + alpha * dvxdc21_z, vy * dALdc21_z + alpha * dvydc21_z, vz * dALdc21_z + alpha * dvzdc21_z, vx * -dALdc21_z + (1 - alpha) * dvxdc21_z, vy * -dALdc21_z + (1 - alpha) * dvydc21_z, vz * -dALdc21_z + (1 - alpha) * dvzdc21_z, vx * -dBTdc21_z - beta * dvxdc21_z, vy * -dBTdc21_z - beta * dvydc21_z, vz * -dBTdc21_z - beta * dvzdc21_z, vx * dBTdc21_z + (beta - 1.0) * dvxdc21_z, vy * dBTdc21_z + (beta - 1.0) * dvydc21_z, vz * dBTdc21_z + (beta - 1.0) * dvzdc21_z,
			vx * dALdc22_x + alpha * dvxdc22_x, vy * dALdc22_x + alpha * dvydc22_x, vz * dALdc22_x + alpha * dvzdc22_x, vx * -dALdc22_x + (1 - alpha) * dvxdc22_x, vy * -dALdc22_x + (1 - alpha) * dvydc22_x, vz * -dALdc22_x + (1 - alpha) * dvzdc22_x, vx * -dBTdc22_x - beta * dvxdc22_x, vy * -dBTdc22_x - beta * dvydc22_x, vz * -dBTdc22_x - beta * dvzdc22_x, vx * dBTdc22_x + (beta - 1.0) * dvxdc22_x, vy * dBTdc22_x + (beta - 1.0) * dvydc22_x, vz * dBTdc22_x + (beta - 1.0) * dvzdc22_x,
			vx * dALdc22_y + alpha * dvxdc22_y, vy * dALdc22_y + alpha * dvydc22_y, vz * dALdc22_y + alpha * dvzdc22_y, vx * -dALdc22_y + (1 - alpha) * dvxdc22_y, vy * -dALdc22_y + (1 - alpha) * dvydc22_y, vz * -dALdc22_y + (1 - alpha) * dvzdc22_y, vx * -dBTdc22_y - beta * dvxdc22_y, vy * -dBTdc22_y - beta * dvydc22_y, vz * -dBTdc22_y - beta * dvzdc22_y, vx * dBTdc22_y + (beta - 1.0) * dvxdc22_y, vy * dBTdc22_y + (beta - 1.0) * dvydc22_y, vz * dBTdc22_y + (beta - 1.0) * dvzdc22_y,
			vx * dALdc22_z + alpha * dvxdc22_z, vy * dALdc22_z + alpha * dvydc22_z, vz * dALdc22_z + alpha * dvzdc22_z, vx * -dALdc22_z + (1 - alpha) * dvxdc22_z, vy * -dALdc22_z + (1 - alpha) * dvydc22_z, vz * -dALdc22_z + (1 - alpha) * dvzdc22_z, vx * -dBTdc22_z - beta * dvxdc22_z, vy * -dBTdc22_z - beta * dvydc22_z, vz * -dBTdc22_z - beta * dvzdc22_z, vx * dBTdc22_z + (beta - 1.0) * dvxdc22_z, vy * dBTdc22_z + (beta - 1.0) * dvydc22_z, vz * dBTdc22_z + (beta - 1.0) * dvzdc22_z;
		hessian.transposeInPlace();
	}

	void MipcConeConeConstraint::betaIsOneHessian1(mat12 &hessian)
	{
		scalar dAdc11_x = 2.0 * sC1.data()[0];
		scalar dAdc11_y = 2.0 * sC1.data()[1];
		scalar dAdc11_z = 2.0 * sC1.data()[2];
		scalar dAdc12_x = -2.0 * sC1.data()[0];
		scalar dAdc12_y = -2.0 * sC1.data()[1];
		scalar dAdc12_z = -2.0 * sC1.data()[2];

		scalar dBdc11_x = 2.0 * sC2.data()[0];
		scalar dBdc11_y = 2.0 * sC2.data()[1];
		scalar dBdc11_z = 2.0 * sC2.data()[2];
		scalar dBdc12_x = -2.0 * sC2.data()[0];
		scalar dBdc12_y = -2.0 * sC2.data()[1];
		scalar dBdc12_z = -2.0 * sC2.data()[2];
		scalar dBdc21_x = -2.0 * sC1.data()[0];
		scalar dBdc21_y = -2.0 * sC1.data()[1];
		scalar dBdc21_z = -2.0 * sC1.data()[2];
		scalar dBdc22_x = 2.0 * sC1.data()[0];
		scalar dBdc22_y = 2.0 * sC1.data()[1];
		scalar dBdc22_z = 2.0 * sC1.data()[2];

		scalar dCdc21_x = -2.0 * sC2.data()[0];
		scalar dCdc21_y = -2.0 * sC2.data()[1];
		scalar dCdc21_z = -2.0 * sC2.data()[2];
		scalar dCdc22_x = 2.0 * sC2.data()[0];
		scalar dCdc22_y = 2.0 * sC2.data()[1];
		scalar dCdc22_z = 2.0 * sC2.data()[2];

		scalar dDdc11_x = 2.0 * sC3.data()[0];
		scalar dDdc11_y = 2.0 * sC3.data()[1];
		scalar dDdc11_z = 2.0 * sC3.data()[2];
		scalar dDdc12_x = 2.0 * (sC1.data()[0] - sC3.data()[0]);
		scalar dDdc12_y = 2.0 * (sC1.data()[1] - sC3.data()[1]);
		scalar dDdc12_z = 2.0 * (sC1.data()[2] - sC3.data()[2]);
		scalar dDdc22_x = -2.0 * sC1.data()[0];
		scalar dDdc22_y = -2.0 * sC1.data()[1];
		scalar dDdc22_z = -2.0 * sC1.data()[2];

		scalar dEdc12_x = 2.0 * sC2.data()[0];
		scalar dEdc12_y = 2.0 * sC2.data()[1];
		scalar dEdc12_z = 2.0 * sC2.data()[2];
		scalar dEdc21_x = -2.0 * sC3.data()[0];
		scalar dEdc21_y = -2.0 * sC3.data()[1];
		scalar dEdc21_z = -2.0 * sC3.data()[2];
		scalar dEdc22_x = 2.0 * (sC3.data()[0] - sC2.data()[0]);
		scalar dEdc22_y = 2.0 * (sC3.data()[1] - sC2.data()[1]);
		scalar dEdc22_z = 2.0 * (sC3.data()[2] - sC2.data()[2]);

		scalar dFdc12_x = 2.0 * sC3.data()[0];
		scalar dFdc12_y = 2.0 * sC3.data()[1];
		scalar dFdc12_z = 2.0 * sC3.data()[2];
		scalar dFdc22 = -2.0 * sC3.data()[0];
		scalar dFdc22_y = -2.0 * sC3.data()[1];
		scalar dFd22_z = -2.0 * sC3.data()[2];

		//
		scalar q1 = 0.5 * (B + D) / (A * A);
		scalar q2 = -0.5 / A;
		scalar dALdc11_x = q1 * dAdc11_x + q2 * (dBdc11_x + dDdc11_x);
		scalar dALdc11_y = q1 * dAdc11_y + q2 * (dBdc11_y + dDdc11_y);
		scalar dALdc11_z = q1 * dAdc11_z + q2 * (dBdc11_z + dDdc11_z);

		scalar dALdc12_x = q1 * dAdc12_x + q2 * (dBdc12_x + dDdc12_x);
		scalar dALdc12_y = q1 * dAdc12_y + q2 * (dBdc12_y + dDdc12_y);
		scalar dALdc12_z = q1 * dAdc12_z + q2 * (dBdc12_z + dDdc12_z);

		scalar dALdc21_x = q2 * dBdc21_x;
		scalar dALdc21_y = q2 * dBdc21_y;
		scalar dALdc21_z = q2 * dBdc21_z;

		scalar dALdc22_x = q2 * (dBdc22_x + dDdc22_x);
		scalar dALdc22_y = q2 * (dBdc22_x + dDdc22_y);
		scalar dALdc22_z = q2 * (dBdc22_x + dDdc22_z);

		scalar dBTdc11_x = 0;
		scalar dBTdc11_y = 0;
		scalar dBTdc11_z = 0;

		scalar dBTdc12_x = 0;
		scalar dBTdc12_y = 0;
		scalar dBTdc12_z = 0;

		scalar dBTdc21_x = 0;
		scalar dBTdc21_y = 0;
		scalar dBTdc21_z = 0;

		scalar dBTdc22_x = 0;
		scalar dBTdc22_y = 0;
		scalar dBTdc22_z = 0;

		scalar C1x = sC1.data()[0];
		scalar C1y = sC1.data()[1];
		scalar C1z = sC1.data()[2];
		scalar C2x = sC2.data()[0];
		scalar C2y = sC2.data()[1];
		scalar C2z = sC2.data()[2];
		scalar C3x = sC3.data()[0];
		scalar C3y = sC3.data()[1];
		scalar C3z = sC3.data()[2];

		scalar vx = 2.0 * (alpha * C1x + beta * C2x + C3x);
		scalar vy = 2.0 * (alpha * C1y + beta * C2y + C3y);
		scalar vz = 2.0 * (alpha * C1z + beta * C2z + C3z);

		scalar dvxdc11_x = 2.0 * (C1x * dALdc11_x + C2x * dBTdc11_x + alpha);
		scalar dvxdc11_y = 2.0 * (C1x * dALdc11_y + C2x * dBTdc11_y);
		scalar dvxdc11_z = 2.0 * (C1x * dALdc11_z + C2x * dBTdc11_z);

		scalar dvxdc12_x = 2.0 * (C1x * dALdc12_x + C2x * dBTdc12_x - alpha + 1);
		scalar dvxdc12_y = 2.0 * (C1x * dALdc12_y + C2x * dBTdc12_y);
		scalar dvxdc12_z = 2.0 * (C1x * dALdc12_z + C2x * dBTdc12_z);

		scalar dvxdc21_x = 2.0 * (C1x * dALdc21_x + C2x * dBTdc21_x - beta);
		scalar dvxdc21_y = 2.0 * (C1x * dALdc21_y + C2x * dBTdc21_y);
		scalar dvxdc21_z = 2.0 * (C1x * dALdc21_z + C2x * dBTdc21_z);

		scalar dvxdc22_x = 2.0 * (C1x * dALdc22_x + C2x * dBTdc22_x + beta - 1);
		scalar dvxdc22_y = 2.0 * (C1x * dALdc22_y + C2x * dBTdc22_y);
		scalar dvxdc22_z = 2.0 * (C1x * dALdc22_z + C2x * dBTdc22_z);
		//
		scalar dvydc11_x = 2.0 * (C1y * dALdc11_x + C2y * dBTdc11_x);
		scalar dvydc11_y = 2.0 * (C1y * dALdc11_y + C2y * dBTdc11_y + alpha);
		scalar dvydc11_z = 2.0 * (C1y * dALdc11_z + C2y * dBTdc11_z);

		scalar dvydc12_x = 2.0 * (C1y * dALdc12_x + C2y * dBTdc12_x);
		scalar dvydc12_y = 2.0 * (C1y * dALdc12_y + C2y * dBTdc12_y - alpha + 1);
		scalar dvydc12_z = 2.0 * (C1y * dALdc12_z + C2y * dBTdc12_z);

		scalar dvydc21_x = 2.0 * (C1y * dALdc21_x + C2y * dBTdc21_x);
		scalar dvydc21_y = 2.0 * (C1y * dALdc21_y + C2y * dBTdc21_y - beta);
		scalar dvydc21_z = 2.0 * (C1y * dALdc21_z + C2y * dBTdc21_z);

		scalar dvydc22_x = 2.0 * (C1y * dALdc22_x + C2y * dBTdc22_x);
		scalar dvydc22_y = 2.0 * (C1y * dALdc22_y + C2y * dBTdc22_y + beta - 1);
		scalar dvydc22_z = 2.0 * (C1y * dALdc22_z + C2y * dBTdc22_z);
		//
		scalar dvzdc11_x = 2.0 * (C1z * dALdc11_x + C2z * dBTdc11_x);
		scalar dvzdc11_y = 2.0 * (C1z * dALdc11_y + C2z * dBTdc11_y);
		scalar dvzdc11_z = 2.0 * (C1z * dALdc11_z + C2z * dBTdc11_z + alpha);

		scalar dvzdc12_x = 2.0 * (C1z * dALdc12_x + C2z * dBTdc12_x);
		scalar dvzdc12_y = 2.0 * (C1z * dALdc12_y + C2z * dBTdc12_y);
		scalar dvzdc12_z = 2.0 * (C1z * dALdc12_z + C2z * dBTdc12_z - alpha + 1);

		scalar dvzdc21_x = 2.0 * (C1z * dALdc21_x + C2z * dBTdc21_x);
		scalar dvzdc21_y = 2.0 * (C1z * dALdc21_y + C2z * dBTdc21_y);
		scalar dvzdc21_z = 2.0 * (C1z * dALdc21_z + C2z * dBTdc21_z - beta);

		scalar dvzdc22_x = 2.0 * (C1z * dALdc22_x + C2z * dBTdc22_x);
		scalar dvzdc22_y = 2.0 * (C1z * dALdc22_y + C2z * dBTdc22_y);
		scalar dvzdc22_z = 2.0 * (C1z * dALdc22_z + C2z * dBTdc22_z + beta - 1);

		hessian.setZero();
		// f / c11x
		hessian.data()[0] = vx * dALdc11_x + alpha * dvxdc11_x;
		hessian.data()[1] = vy * dALdc11_x + alpha * dvydc11_x;
		hessian.data()[2] = vz * dALdc11_x + alpha * dvzdc11_x;

		hessian.data()[3] = vx * -dALdc11_x + (1 - alpha) * dvxdc11_x;
		hessian.data()[4] = vy * -dALdc11_x + (1 - alpha) * dvydc11_x;
		hessian.data()[5] = vz * -dALdc11_x + (1 - alpha) * dvzdc11_x;

		hessian.data()[6] = vx * -dBTdc11_x - beta * dvxdc11_x;
		hessian.data()[7] = vy * -dBTdc11_x - beta * dvydc11_x;
		hessian.data()[8] = vz * -dBTdc11_x - beta * dvzdc11_x;

		hessian.data()[9] = vx * dBTdc11_x + (beta - 1.0) * dvxdc11_x;
		hessian.data()[10] = vy * dBTdc11_x + (beta - 1.0) * dvydc11_x;
		hessian.data()[11] = vz * dBTdc11_x + (beta - 1.0) * dvzdc11_x;
		// f / c11y
		hessian.data()[12] = vx * dALdc11_y + alpha * dvxdc11_y;
		hessian.data()[13] = vy * dALdc11_y + alpha * dvydc11_y;
		hessian.data()[14] = vz * dALdc11_y + alpha * dvzdc11_y;

		hessian.data()[15] = vx * -dALdc11_y + (1 - alpha) * dvxdc11_y;
		hessian.data()[16] = vy * -dALdc11_y + (1 - alpha) * dvydc11_y;
		hessian.data()[17] = vz * -dALdc11_y + (1 - alpha) * dvzdc11_y;

		hessian.data()[18] = vx * -dBTdc11_y - beta * dvxdc11_y;
		hessian.data()[19] = vy * -dBTdc11_y - beta * dvydc11_y;
		hessian.data()[20] = vz * -dBTdc11_y - beta * dvzdc11_y;

		hessian.data()[21] = vx * dBTdc11_y + (beta - 1.0) * dvxdc11_y;
		hessian.data()[22] = vy * dBTdc11_y + (beta - 1.0) * dvydc11_y;
		hessian.data()[23] = vz * dBTdc11_y + (beta - 1.0) * dvzdc11_y;
		// f / c11z
		hessian.data()[24] = vx * dALdc11_z + alpha * dvxdc11_z;
		hessian.data()[25] = vy * dALdc11_z + alpha * dvydc11_z;
		hessian.data()[26] = vz * dALdc11_z + alpha * dvzdc11_z;

		hessian.data()[27] = vx * -dALdc11_z + (1 - alpha) * dvxdc11_z;
		hessian.data()[28] = vy * -dALdc11_z + (1 - alpha) * dvydc11_z;
		hessian.data()[29] = vz * -dALdc11_z + (1 - alpha) * dvzdc11_z;

		hessian.data()[30] = vx * -dBTdc11_z - beta * dvxdc11_z;
		hessian.data()[31] = vy * -dBTdc11_z - beta * dvydc11_z;
		hessian.data()[32] = vz * -dBTdc11_z - beta * dvzdc11_z;

		hessian.data()[33] = vx * dBTdc11_z + (beta - 1.0) * dvxdc11_z;
		hessian.data()[34] = vy * dBTdc11_z + (beta - 1.0) * dvydc11_z;
		hessian.data()[35] = vz * dBTdc11_z + (beta - 1.0) * dvzdc11_z;

		// f / c12x
		hessian.data()[36] = vx * dALdc12_x + alpha * dvxdc12_x;
		hessian.data()[37] = vy * dALdc12_x + alpha * dvydc12_x;
		hessian.data()[38] = vz * dALdc12_x + alpha * dvzdc12_x;

		hessian.data()[39] = vx * -dALdc12_x + (1 - alpha) * dvxdc12_x;
		hessian.data()[40] = vy * -dALdc12_x + (1 - alpha) * dvydc12_x;
		hessian.data()[41] = vz * -dALdc12_x + (1 - alpha) * dvzdc12_x;

		hessian.data()[42] = vx * -dBTdc12_x - beta * dvxdc12_x;
		hessian.data()[43] = vy * -dBTdc12_x - beta * dvydc12_x;
		hessian.data()[44] = vz * -dBTdc12_x - beta * dvzdc12_x;

		hessian.data()[45] = vx * dBTdc12_x + (beta - 1.0) * dvxdc12_x;
		hessian.data()[46] = vy * dBTdc12_x + (beta - 1.0) * dvydc12_x;
		hessian.data()[47] = vz * dBTdc12_x + (beta - 1.0) * dvzdc12_x;
		// f / c12y
		hessian.data()[48] = vx * dALdc12_y + alpha * dvxdc12_y;
		hessian.data()[49] = vy * dALdc12_y + alpha * dvydc12_y;
		hessian.data()[50] = vz * dALdc12_y + alpha * dvzdc12_y;

		hessian.data()[51] = vx * -dALdc12_y + (1 - alpha) * dvxdc12_y;
		hessian.data()[52] = vy * -dALdc12_y + (1 - alpha) * dvydc12_y;
		hessian.data()[53] = vz * -dALdc12_y + (1 - alpha) * dvzdc12_y;

		hessian.data()[54] = vx * -dBTdc12_y - beta * dvxdc12_y;
		hessian.data()[55] = vy * -dBTdc12_y - beta * dvydc12_y;
		hessian.data()[56] = vz * -dBTdc12_y - beta * dvzdc12_y;

		hessian.data()[57] = vx * dBTdc12_y + (beta - 1.0) * dvxdc12_y;
		hessian.data()[58] = vy * dBTdc12_y + (beta - 1.0) * dvydc12_y;
		hessian.data()[59] = vz * dBTdc12_y + (beta - 1.0) * dvzdc12_y;
		// f / c12z
		hessian.data()[60] = vx * dALdc12_z + alpha * dvxdc12_z;
		hessian.data()[61] = vy * dALdc12_z + alpha * dvydc12_z;
		hessian.data()[62] = vz * dALdc12_z + alpha * dvzdc12_z;

		hessian.data()[63] = vx * -dALdc12_z + (1 - alpha) * dvxdc12_z;
		hessian.data()[64] = vy * -dALdc12_z + (1 - alpha) * dvydc12_z;
		hessian.data()[65] = vz * -dALdc12_z + (1 - alpha) * dvzdc12_z;

		hessian.data()[66] = vx * -dBTdc12_z - beta * dvxdc12_z;
		hessian.data()[67] = vy * -dBTdc12_z - beta * dvydc12_z;
		hessian.data()[68] = vz * -dBTdc12_z - beta * dvzdc12_z;

		hessian.data()[69] = vx * dBTdc12_z + (beta - 1.0) * dvxdc12_z;
		hessian.data()[70] = vy * dBTdc12_z + (beta - 1.0) * dvydc12_z;
		hessian.data()[71] = vz * dBTdc12_z + (beta - 1.0) * dvzdc12_z;

		// f / c21x
		hessian.data()[72] = vx * dALdc21_x + alpha * dvxdc21_x;
		hessian.data()[73] = vy * dALdc21_x + alpha * dvydc21_x;
		hessian.data()[74] = vz * dALdc21_x + alpha * dvzdc21_x;

		hessian.data()[75] = vx * -dALdc21_x + (1 - alpha) * dvxdc21_x;
		hessian.data()[76] = vy * -dALdc21_x + (1 - alpha) * dvydc21_x;
		hessian.data()[77] = vz * -dALdc21_x + (1 - alpha) * dvzdc21_x;

		hessian.data()[78] = vx * -dBTdc21_x - beta * dvxdc21_x;
		hessian.data()[79] = vy * -dBTdc21_x - beta * dvydc21_x;
		hessian.data()[80] = vz * -dBTdc21_x - beta * dvzdc21_x;

		hessian.data()[81] = vx * dBTdc21_x + (beta - 1.0) * dvxdc21_x;
		hessian.data()[82] = vy * dBTdc21_x + (beta - 1.0) * dvydc21_x;
		hessian.data()[83] = vz * dBTdc21_x + (beta - 1.0) * dvzdc21_x;
		// f / c21y
		hessian.data()[84] = vx * dALdc21_y + alpha * dvxdc21_y;
		hessian.data()[85] = vy * dALdc21_y + alpha * dvydc21_y;
		hessian.data()[86] = vz * dALdc21_y + alpha * dvzdc21_y;

		hessian.data()[87] = vx * -dALdc21_y + (1 - alpha) * dvxdc21_y;
		hessian.data()[88] = vy * -dALdc21_y + (1 - alpha) * dvydc21_y;
		hessian.data()[89] = vz * -dALdc21_y + (1 - alpha) * dvzdc21_y;

		hessian.data()[90] = vx * -dBTdc21_y - beta * dvxdc21_y;
		hessian.data()[91] = vy * -dBTdc21_y - beta * dvydc21_y;
		hessian.data()[92] = vz * -dBTdc21_y - beta * dvzdc21_y;

		hessian.data()[93] = vx * dBTdc21_y + (beta - 1.0) * dvxdc21_y;
		hessian.data()[94] = vy * dBTdc21_y + (beta - 1.0) * dvydc21_y;
		hessian.data()[95] = vz * dBTdc21_y + (beta - 1.0) * dvzdc21_y;
		// f / c21z
		hessian.data()[96] = vx * dALdc21_z + alpha * dvxdc21_z;
		hessian.data()[97] = vy * dALdc21_z + alpha * dvydc21_z;
		hessian.data()[98] = vz * dALdc21_z + alpha * dvzdc21_z;

		hessian.data()[99] = vx * -dALdc21_z + (1 - alpha) * dvxdc21_z;
		hessian.data()[100] = vy * -dALdc21_z + (1 - alpha) * dvydc21_z;
		hessian.data()[101] = vz * -dALdc21_z + (1 - alpha) * dvzdc21_z;

		hessian.data()[102] = vx * -dBTdc21_z - beta * dvxdc21_z;
		hessian.data()[103] = vy * -dBTdc21_z - beta * dvydc21_z;
		hessian.data()[104] = vz * -dBTdc21_z - beta * dvzdc21_z;

		hessian.data()[105] = vx * dBTdc21_z + (beta - 1.0) * dvxdc21_z;
		hessian.data()[106] = vy * dBTdc21_z + (beta - 1.0) * dvydc21_z;
		hessian.data()[107] = vz * dBTdc21_z + (beta - 1.0) * dvzdc21_z;

		// f / c22x
		hessian.data()[108] = vx * dALdc22_x + alpha * dvxdc22_x;
		hessian.data()[109] = vy * dALdc22_x + alpha * dvydc22_x;
		hessian.data()[110] = vz * dALdc22_x + alpha * dvzdc22_x;

		hessian.data()[111] = vx * -dALdc22_x + (1 - alpha) * dvxdc22_x;
		hessian.data()[112] = vy * -dALdc22_x + (1 - alpha) * dvydc22_x;
		hessian.data()[113] = vz * -dALdc22_x + (1 - alpha) * dvzdc22_x;

		hessian.data()[114] = vx * -dBTdc22_x - beta * dvxdc22_x;
		hessian.data()[115] = vy * -dBTdc22_x - beta * dvydc22_x;
		hessian.data()[116] = vz * -dBTdc22_x - beta * dvzdc22_x;

		hessian.data()[117] = vx * dBTdc22_x + (beta - 1.0) * dvxdc22_x;
		hessian.data()[118] = vy * dBTdc22_x + (beta - 1.0) * dvydc22_x;
		hessian.data()[119] = vz * dBTdc22_x + (beta - 1.0) * dvzdc22_x;
		// f / c22y
		hessian.data()[120] = vx * dALdc22_y + alpha * dvxdc22_y;
		hessian.data()[121] = vy * dALdc22_y + alpha * dvydc22_y;
		hessian.data()[122] = vz * dALdc22_y + alpha * dvzdc22_y;

		hessian.data()[123] = vx * -dALdc22_y + (1 - alpha) * dvxdc22_y;
		hessian.data()[124] = vy * -dALdc22_y + (1 - alpha) * dvydc22_y;
		hessian.data()[125] = vz * -dALdc22_y + (1 - alpha) * dvzdc22_y;

		hessian.data()[126] = vx * -dBTdc22_y - beta * dvxdc22_y;
		hessian.data()[127] = vy * -dBTdc22_y - beta * dvydc22_y;
		hessian.data()[128] = vz * -dBTdc22_y - beta * dvzdc22_y;

		hessian.data()[129] = vx * dBTdc22_y + (beta - 1.0) * dvxdc22_y;
		hessian.data()[130] = vy * dBTdc22_y + (beta - 1.0) * dvydc22_y;
		hessian.data()[131] = vz * dBTdc22_y + (beta - 1.0) * dvzdc22_y;
		// f / c22z
		hessian.data()[132] = vx * dALdc22_z + alpha * dvxdc22_z;
		hessian.data()[133] = vy * dALdc22_z + alpha * dvydc22_z;
		hessian.data()[134] = vz * dALdc22_z + alpha * dvzdc22_z;

		hessian.data()[135] = vx * -dALdc22_z + (1 - alpha) * dvxdc22_z;
		hessian.data()[136] = vy * -dALdc22_z + (1 - alpha) * dvydc22_z;
		hessian.data()[137] = vz * -dALdc22_z + (1 - alpha) * dvzdc22_z;

		hessian.data()[138] = vx * -dBTdc22_z - beta * dvxdc22_z;
		hessian.data()[139] = vy * -dBTdc22_z - beta * dvydc22_z;
		hessian.data()[140] = vz * -dBTdc22_z - beta * dvzdc22_z;

		hessian.data()[141] = vx * dBTdc22_z + (beta - 1.0) * dvxdc22_z;
		hessian.data()[142] = vy * dBTdc22_z + (beta - 1.0) * dvydc22_z;
		hessian.data()[143] = vz * dBTdc22_z + (beta - 1.0) * dvzdc22_z;
	}

	void MipcConeConeConstraint::alphaBetaHessian(mat12 &hessian)
	{
		scalar dAdc11_x = 2.0 * sC1.data()[0];
		scalar dAdc11_y = 2.0 * sC1.data()[1];
		scalar dAdc11_z = 2.0 * sC1.data()[2];
		scalar dAdc12_x = -2.0 * sC1.data()[0];
		scalar dAdc12_y = -2.0 * sC1.data()[1];
		scalar dAdc12_z = -2.0 * sC1.data()[2];

		scalar dBdc11_x = 2.0 * sC2.data()[0];
		scalar dBdc11_y = 2.0 * sC2.data()[1];
		scalar dBdc11_z = 2.0 * sC2.data()[2];
		scalar dBdc12_x = -2.0 * sC2.data()[0];
		scalar dBdc12_y = -2.0 * sC2.data()[1];
		scalar dBdc12_z = -2.0 * sC2.data()[2];
		scalar dBdc21_x = -2.0 * sC1.data()[0];
		scalar dBdc21_y = -2.0 * sC1.data()[1];
		scalar dBdc21_z = -2.0 * sC1.data()[2];
		scalar dBdc22_x = 2.0 * sC1.data()[0];
		scalar dBdc22_y = 2.0 * sC1.data()[1];
		scalar dBdc22_z = 2.0 * sC1.data()[2];

		scalar dCdc21_x = -2.0 * sC2.data()[0];
		scalar dCdc21_y = -2.0 * sC2.data()[1];
		scalar dCdc21_z = -2.0 * sC2.data()[2];
		scalar dCdc22_x = 2.0 * sC2.data()[0];
		scalar dCdc22_y = 2.0 * sC2.data()[1];
		scalar dCdc22_z = 2.0 * sC2.data()[2];

		scalar dDdc11_x = 2.0 * sC3.data()[0];
		scalar dDdc11_y = 2.0 * sC3.data()[1];
		scalar dDdc11_z = 2.0 * sC3.data()[2];
		scalar dDdc12_x = 2.0 * (sC1.data()[0] - sC3.data()[0]);
		scalar dDdc12_y = 2.0 * (sC1.data()[1] - sC3.data()[1]);
		scalar dDdc12_z = 2.0 * (sC1.data()[2] - sC3.data()[2]);
		scalar dDdc22_x = -2.0 * sC1.data()[0];
		scalar dDdc22_y = -2.0 * sC1.data()[1];
		scalar dDdc22_z = -2.0 * sC1.data()[2];

		scalar dEdc12_x = 2.0 * sC2.data()[0];
		scalar dEdc12_y = 2.0 * sC2.data()[1];
		scalar dEdc12_z = 2.0 * sC2.data()[2];
		scalar dEdc21_x = -2.0 * sC3.data()[0];
		scalar dEdc21_y = -2.0 * sC3.data()[1];
		scalar dEdc21_z = -2.0 * sC3.data()[2];
		scalar dEdc22_x = 2.0 * (sC3.data()[0] - sC2.data()[0]);
		scalar dEdc22_y = 2.0 * (sC3.data()[1] - sC2.data()[1]);
		scalar dEdc22_z = 2.0 * (sC3.data()[2] - sC2.data()[2]);

		scalar dFdc12_x = 2.0 * sC3.data()[0];
		scalar dFdc12_y = 2.0 * sC3.data()[1];
		scalar dFdc12_z = 2.0 * sC3.data()[2];
		scalar dFdc22 = -2.0 * sC3.data()[0];
		scalar dFdc22_y = -2.0 * sC3.data()[1];
		scalar dFd22_z = -2.0 * sC3.data()[2];

		//
		scalar q1 = 1.0 / delta;
		scalar q2 = alpha / delta;
		scalar q3 = beta / delta;

		scalar dALdc11_x = q1 * (E * dBdc11_x - 2.0 * C * dDdc11_x) - q2 * (4.0 * C * dAdc11_x - 2.0 * B * dBdc11_x);
		scalar dALdc11_y = q1 * (E * dBdc11_y - 2.0 * C * dDdc11_y) - q2 * (4.0 * C * dAdc11_y - 2.0 * B * dBdc11_y);
		scalar dALdc11_z = q1 * (E * dBdc11_z - 2.0 * C * dDdc11_z) - q2 * (4.0 * C * dAdc11_z - 2.0 * B * dBdc11_z);

		scalar dALdc12_x = q1 * (E * dBdc12_x + B * dEdc12_x - 2.0 * C * dDdc12_x) - q2 * (4.0 * C * dAdc12_x - 2.0 * B * dBdc12_x);
		scalar dALdc12_y = q1 * (E * dBdc12_y + B * dEdc12_y - 2.0 * C * dDdc12_y) - q2 * (4.0 * C * dAdc12_y - 2.0 * B * dBdc12_y);
		scalar dALdc12_z = q1 * (E * dBdc12_z + B * dEdc12_z - 2.0 * C * dDdc12_z) - q2 * (4.0 * C * dAdc12_z - 2.0 * B * dBdc12_z);

		scalar dALdc21_x = q1 * (E * dBdc21_x + B * dEdc21_x - 2.0 * D * dCdc21_x) - q2 * (4.0 * A * dCdc21_x - 2.0 * B * dBdc21_x);
		scalar dALdc21_y = q1 * (E * dBdc21_y + B * dEdc21_y - 2.0 * D * dCdc21_y) - q2 * (4.0 * A * dCdc21_y - 2.0 * B * dBdc21_y);
		scalar dALdc21_z = q1 * (E * dBdc21_z + B * dEdc21_z - 2.0 * D * dCdc21_z) - q2 * (4.0 * A * dCdc21_z - 2.0 * B * dBdc21_z);

		scalar dALdc22_x = q1 * (E * dBdc22_x + B * dEdc22_x - 2.0 * (D * dCdc22_x + C * dDdc22_x)) - q2 * (4.0 * A * dCdc22_x - 2.0 * B * dBdc22_x);
		scalar dALdc22_y = q1 * (E * dBdc22_y + B * dEdc22_y - 2.0 * (D * dCdc22_y + C * dDdc22_y)) - q2 * (4.0 * A * dCdc22_y - 2.0 * B * dBdc22_y);
		scalar dALdc22_z = q1 * (E * dBdc22_z + B * dEdc22_z - 2.0 * (D * dCdc22_z + C * dDdc22_z)) - q2 * (4.0 * A * dCdc22_z - 2.0 * B * dBdc22_z);

		scalar dBTdc11_x = q1 * (D * dBdc11_x + B * dDdc11_x - 2.0 * E * dAdc11_x) - q3 * (4.0 * C * dAdc11_x - 2.0 * B * dBdc11_x);
		scalar dBTdc11_y = q1 * (D * dBdc11_y + B * dDdc11_y - 2.0 * E * dAdc11_y) - q3 * (4.0 * C * dAdc11_y - 2.0 * B * dBdc11_y);
		scalar dBTdc11_z = q1 * (D * dBdc11_z + B * dDdc11_z - 2.0 * E * dAdc11_z) - q3 * (4.0 * C * dAdc11_z - 2.0 * B * dBdc11_z);

		scalar dBTdc12_x = q1 * (D * dBdc12_x + B * dDdc12_x - 2.0 * (E * dAdc12_x + A * dEdc12_x)) - q3 * (4.0 * C * dAdc12_x - 2.0 * B * dBdc12_x);
		scalar dBTdc12_y = q1 * (D * dBdc12_y + B * dDdc12_y - 2.0 * (E * dAdc12_y + A * dEdc12_y)) - q3 * (4.0 * C * dAdc12_y - 2.0 * B * dBdc12_y);
		scalar dBTdc12_z = q1 * (D * dBdc12_z + B * dDdc12_z - 2.0 * (E * dAdc12_z + A * dEdc12_z)) - q3 * (4.0 * C * dAdc12_z - 2.0 * B * dBdc12_z);

		scalar dBTdc21_x = q1 * (D * dBdc21_x - 2.0 * A * dEdc21_x) - q3 * (4.0 * A * dCdc21_x - 2.0 * B * dBdc21_x);
		scalar dBTdc21_y = q1 * (D * dBdc21_y - 2.0 * A * dEdc21_y) - q3 * (4.0 * A * dCdc21_y - 2.0 * B * dBdc21_y);
		scalar dBTdc21_z = q1 * (D * dBdc21_z - 2.0 * A * dEdc21_z) - q3 * (4.0 * A * dCdc21_z - 2.0 * B * dBdc21_z);

		scalar dBTdc22_x = q1 * (D * dBdc22_x + B * dDdc22_x - 2.0 * A * dEdc22_x) - q3 * (4.0 * A * dCdc22_x - 2.0 * B * dBdc22_x);
		scalar dBTdc22_y = q1 * (D * dBdc22_y + B * dDdc22_y - 2.0 * A * dEdc22_y) - q3 * (4.0 * A * dCdc22_y - 2.0 * B * dBdc22_y);
		scalar dBTdc22_z = q1 * (D * dBdc22_z + B * dDdc22_z - 2.0 * A * dEdc22_z) - q3 * (4.0 * A * dCdc22_z - 2.0 * B * dBdc22_z);

		scalar C1x = sC1.data()[0];
		scalar C1y = sC1.data()[1];
		scalar C1z = sC1.data()[2];
		scalar C2x = sC2.data()[0];
		scalar C2y = sC2.data()[1];
		scalar C2z = sC2.data()[2];
		scalar C3x = sC3.data()[0];
		scalar C3y = sC3.data()[1];
		scalar C3z = sC3.data()[2];

		scalar vx = 2.0 * (alpha * C1x + beta * C2x + C3x);
		scalar vy = 2.0 * (alpha * C1y + beta * C2y + C3y);
		scalar vz = 2.0 * (alpha * C1z + beta * C2z + C3z);

		scalar dvxdc11_x = 2.0 * (C1x * dALdc11_x + C2x * dBTdc11_x + alpha);
		scalar dvxdc11_y = 2.0 * (C1x * dALdc11_y + C2x * dBTdc11_y);
		scalar dvxdc11_z = 2.0 * (C1x * dALdc11_z + C2x * dBTdc11_z);

		scalar dvxdc12_x = 2.0 * (C1x * dALdc12_x + C2x * dBTdc12_x - alpha + 1);
		scalar dvxdc12_y = 2.0 * (C1x * dALdc12_y + C2x * dBTdc12_y);
		scalar dvxdc12_z = 2.0 * (C1x * dALdc12_z + C2x * dBTdc12_z);

		scalar dvxdc21_x = 2.0 * (C1x * dALdc21_x + C2x * dBTdc21_x - beta);
		scalar dvxdc21_y = 2.0 * (C1x * dALdc21_y + C2x * dBTdc21_y);
		scalar dvxdc21_z = 2.0 * (C1x * dALdc21_z + C2x * dBTdc21_z);

		scalar dvxdc22_x = 2.0 * (C1x * dALdc22_x + C2x * dBTdc22_x + beta - 1);
		scalar dvxdc22_y = 2.0 * (C1x * dALdc22_y + C2x * dBTdc22_y);
		scalar dvxdc22_z = 2.0 * (C1x * dALdc22_z + C2x * dBTdc22_z);
		//
		scalar dvydc11_x = 2.0 * (C1y * dALdc11_x + C2y * dBTdc11_x);
		scalar dvydc11_y = 2.0 * (C1y * dALdc11_y + C2y * dBTdc11_y + alpha);
		scalar dvydc11_z = 2.0 * (C1y * dALdc11_z + C2y * dBTdc11_z);

		scalar dvydc12_x = 2.0 * (C1y * dALdc12_x + C2y * dBTdc12_x);
		scalar dvydc12_y = 2.0 * (C1y * dALdc12_y + C2y * dBTdc12_y - alpha + 1);
		scalar dvydc12_z = 2.0 * (C1y * dALdc12_z + C2y * dBTdc12_z);

		scalar dvydc21_x = 2.0 * (C1y * dALdc21_x + C2y * dBTdc21_x);
		scalar dvydc21_y = 2.0 * (C1y * dALdc21_y + C2y * dBTdc21_y - beta);
		scalar dvydc21_z = 2.0 * (C1y * dALdc21_z + C2y * dBTdc21_z);

		scalar dvydc22_x = 2.0 * (C1y * dALdc22_x + C2y * dBTdc22_x);
		scalar dvydc22_y = 2.0 * (C1y * dALdc22_y + C2y * dBTdc22_y + beta - 1);
		scalar dvydc22_z = 2.0 * (C1y * dALdc22_z + C2y * dBTdc22_z);
		//
		scalar dvzdc11_x = 2.0 * (C1z * dALdc11_x + C2z * dBTdc11_x);
		scalar dvzdc11_y = 2.0 * (C1z * dALdc11_y + C2z * dBTdc11_y);
		scalar dvzdc11_z = 2.0 * (C1z * dALdc11_z + C2z * dBTdc11_z + alpha);

		scalar dvzdc12_x = 2.0 * (C1z * dALdc12_x + C2z * dBTdc12_x);
		scalar dvzdc12_y = 2.0 * (C1z * dALdc12_y + C2z * dBTdc12_y);
		scalar dvzdc12_z = 2.0 * (C1z * dALdc12_z + C2z * dBTdc12_z - alpha + 1);

		scalar dvzdc21_x = 2.0 * (C1z * dALdc21_x + C2z * dBTdc21_x);
		scalar dvzdc21_y = 2.0 * (C1z * dALdc21_y + C2z * dBTdc21_y);
		scalar dvzdc21_z = 2.0 * (C1z * dALdc21_z + C2z * dBTdc21_z - beta);

		scalar dvzdc22_x = 2.0 * (C1z * dALdc22_x + C2z * dBTdc22_x);
		scalar dvzdc22_y = 2.0 * (C1z * dALdc22_y + C2z * dBTdc22_y);
		scalar dvzdc22_z = 2.0 * (C1z * dALdc22_z + C2z * dBTdc22_z + beta - 1);

		hessian.setZero();
		// f / c11x
		hessian.data()[0] = vx * dALdc11_x + alpha * dvxdc11_x;
		hessian.data()[1] = vy * dALdc11_x + alpha * dvydc11_x;
		hessian.data()[2] = vz * dALdc11_x + alpha * dvzdc11_x;

		hessian.data()[3] = vx * -dALdc11_x + (1 - alpha) * dvxdc11_x;
		hessian.data()[4] = vy * -dALdc11_x + (1 - alpha) * dvydc11_x;
		hessian.data()[5] = vz * -dALdc11_x + (1 - alpha) * dvzdc11_x;

		hessian.data()[6] = vx * -dBTdc11_x - beta * dvxdc11_x;
		hessian.data()[7] = vy * -dBTdc11_x - beta * dvydc11_x;
		hessian.data()[8] = vz * -dBTdc11_x - beta * dvzdc11_x;

		hessian.data()[9] = vx * dBTdc11_x + (beta - 1.0) * dvxdc11_x;
		hessian.data()[10] = vy * dBTdc11_x + (beta - 1.0) * dvydc11_x;
		hessian.data()[11] = vz * dBTdc11_x + (beta - 1.0) * dvzdc11_x;
		// f / c11y
		hessian.data()[12] = vx * dALdc11_y + alpha * dvxdc11_y;
		hessian.data()[13] = vy * dALdc11_y + alpha * dvydc11_y;
		hessian.data()[14] = vz * dALdc11_y + alpha * dvzdc11_y;

		hessian.data()[15] = vx * -dALdc11_y + (1 - alpha) * dvxdc11_y;
		hessian.data()[16] = vy * -dALdc11_y + (1 - alpha) * dvydc11_y;
		hessian.data()[17] = vz * -dALdc11_y + (1 - alpha) * dvzdc11_y;

		hessian.data()[18] = vx * -dBTdc11_y - beta * dvxdc11_y;
		hessian.data()[19] = vy * -dBTdc11_y - beta * dvydc11_y;
		hessian.data()[20] = vz * -dBTdc11_y - beta * dvzdc11_y;

		hessian.data()[21] = vx * dBTdc11_y + (beta - 1.0) * dvxdc11_y;
		hessian.data()[22] = vy * dBTdc11_y + (beta - 1.0) * dvydc11_y;
		hessian.data()[23] = vz * dBTdc11_y + (beta - 1.0) * dvzdc11_y;
		// f / c11z
		hessian.data()[24] = vx * dALdc11_z + alpha * dvxdc11_z;
		hessian.data()[25] = vy * dALdc11_z + alpha * dvydc11_z;
		hessian.data()[26] = vz * dALdc11_z + alpha * dvzdc11_z;

		hessian.data()[27] = vx * -dALdc11_z + (1 - alpha) * dvxdc11_z;
		hessian.data()[28] = vy * -dALdc11_z + (1 - alpha) * dvydc11_z;
		hessian.data()[29] = vz * -dALdc11_z + (1 - alpha) * dvzdc11_z;

		hessian.data()[30] = vx * -dBTdc11_z - beta * dvxdc11_z;
		hessian.data()[31] = vy * -dBTdc11_z - beta * dvydc11_z;
		hessian.data()[32] = vz * -dBTdc11_z - beta * dvzdc11_z;

		hessian.data()[33] = vx * dBTdc11_z + (beta - 1.0) * dvxdc11_z;
		hessian.data()[34] = vy * dBTdc11_z + (beta - 1.0) * dvydc11_z;
		hessian.data()[35] = vz * dBTdc11_z + (beta - 1.0) * dvzdc11_z;

		// f / c12x
		hessian.data()[36] = vx * dALdc12_x + alpha * dvxdc12_x;
		hessian.data()[37] = vy * dALdc12_x + alpha * dvydc12_x;
		hessian.data()[38] = vz * dALdc12_x + alpha * dvzdc12_x;

		hessian.data()[39] = vx * -dALdc12_x + (1 - alpha) * dvxdc12_x;
		hessian.data()[40] = vy * -dALdc12_x + (1 - alpha) * dvydc12_x;
		hessian.data()[41] = vz * -dALdc12_x + (1 - alpha) * dvzdc12_x;

		hessian.data()[42] = vx * -dBTdc12_x - beta * dvxdc12_x;
		hessian.data()[43] = vy * -dBTdc12_x - beta * dvydc12_x;
		hessian.data()[44] = vz * -dBTdc12_x - beta * dvzdc12_x;

		hessian.data()[45] = vx * dBTdc12_x + (beta - 1.0) * dvxdc12_x;
		hessian.data()[46] = vy * dBTdc12_x + (beta - 1.0) * dvydc12_x;
		hessian.data()[47] = vz * dBTdc12_x + (beta - 1.0) * dvzdc12_x;
		// f / c12y
		hessian.data()[48] = vx * dALdc12_y + alpha * dvxdc12_y;
		hessian.data()[49] = vy * dALdc12_y + alpha * dvydc12_y;
		hessian.data()[50] = vz * dALdc12_y + alpha * dvzdc12_y;

		hessian.data()[51] = vx * -dALdc12_y + (1 - alpha) * dvxdc12_y;
		hessian.data()[52] = vy * -dALdc12_y + (1 - alpha) * dvydc12_y;
		hessian.data()[53] = vz * -dALdc12_y + (1 - alpha) * dvzdc12_y;

		hessian.data()[54] = vx * -dBTdc12_y - beta * dvxdc12_y;
		hessian.data()[55] = vy * -dBTdc12_y - beta * dvydc12_y;
		hessian.data()[56] = vz * -dBTdc12_y - beta * dvzdc12_y;

		hessian.data()[57] = vx * dBTdc12_y + (beta - 1.0) * dvxdc12_y;
		hessian.data()[58] = vy * dBTdc12_y + (beta - 1.0) * dvydc12_y;
		hessian.data()[59] = vz * dBTdc12_y + (beta - 1.0) * dvzdc12_y;
		// f / c12z
		hessian.data()[60] = vx * dALdc12_z + alpha * dvxdc12_z;
		hessian.data()[61] = vy * dALdc12_z + alpha * dvydc12_z;
		hessian.data()[62] = vz * dALdc12_z + alpha * dvzdc12_z;

		hessian.data()[63] = vx * -dALdc12_z + (1 - alpha) * dvxdc12_z;
		hessian.data()[64] = vy * -dALdc12_z + (1 - alpha) * dvydc12_z;
		hessian.data()[65] = vz * -dALdc12_z + (1 - alpha) * dvzdc12_z;

		hessian.data()[66] = vx * -dBTdc12_z - beta * dvxdc12_z;
		hessian.data()[67] = vy * -dBTdc12_z - beta * dvydc12_z;
		hessian.data()[68] = vz * -dBTdc12_z - beta * dvzdc12_z;

		hessian.data()[69] = vx * dBTdc12_z + (beta - 1.0) * dvxdc12_z;
		hessian.data()[70] = vy * dBTdc12_z + (beta - 1.0) * dvydc12_z;
		hessian.data()[71] = vz * dBTdc12_z + (beta - 1.0) * dvzdc12_z;

		// f / c21x
		hessian.data()[72] = vx * dALdc21_x + alpha * dvxdc21_x;
		hessian.data()[73] = vy * dALdc21_x + alpha * dvydc21_x;
		hessian.data()[74] = vz * dALdc21_x + alpha * dvzdc21_x;

		hessian.data()[75] = vx * -dALdc21_x + (1 - alpha) * dvxdc21_x;
		hessian.data()[76] = vy * -dALdc21_x + (1 - alpha) * dvydc21_x;
		hessian.data()[77] = vz * -dALdc21_x + (1 - alpha) * dvzdc21_x;

		hessian.data()[78] = vx * -dBTdc21_x - beta * dvxdc21_x;
		hessian.data()[79] = vy * -dBTdc21_x - beta * dvydc21_x;
		hessian.data()[80] = vz * -dBTdc21_x - beta * dvzdc21_x;

		hessian.data()[81] = vx * dBTdc21_x + (beta - 1.0) * dvxdc21_x;
		hessian.data()[82] = vy * dBTdc21_x + (beta - 1.0) * dvydc21_x;
		hessian.data()[83] = vz * dBTdc21_x + (beta - 1.0) * dvzdc21_x;
		// f / c21y
		hessian.data()[84] = vx * dALdc21_y + alpha * dvxdc21_y;
		hessian.data()[85] = vy * dALdc21_y + alpha * dvydc21_y;
		hessian.data()[86] = vz * dALdc21_y + alpha * dvzdc21_y;

		hessian.data()[87] = vx * -dALdc21_y + (1 - alpha) * dvxdc21_y;
		hessian.data()[88] = vy * -dALdc21_y + (1 - alpha) * dvydc21_y;
		hessian.data()[89] = vz * -dALdc21_y + (1 - alpha) * dvzdc21_y;

		hessian.data()[90] = vx * -dBTdc21_y - beta * dvxdc21_y;
		hessian.data()[91] = vy * -dBTdc21_y - beta * dvydc21_y;
		hessian.data()[92] = vz * -dBTdc21_y - beta * dvzdc21_y;

		hessian.data()[93] = vx * dBTdc21_y + (beta - 1.0) * dvxdc21_y;
		hessian.data()[94] = vy * dBTdc21_y + (beta - 1.0) * dvydc21_y;
		hessian.data()[95] = vz * dBTdc21_y + (beta - 1.0) * dvzdc21_y;
		// f / c21z
		hessian.data()[96] = vx * dALdc21_z + alpha * dvxdc21_z;
		hessian.data()[97] = vy * dALdc21_z + alpha * dvydc21_z;
		hessian.data()[98] = vz * dALdc21_z + alpha * dvzdc21_z;

		hessian.data()[99] = vx * -dALdc21_z + (1 - alpha) * dvxdc21_z;
		hessian.data()[100] = vy * -dALdc21_z + (1 - alpha) * dvydc21_z;
		hessian.data()[101] = vz * -dALdc21_z + (1 - alpha) * dvzdc21_z;

		hessian.data()[102] = vx * -dBTdc21_z - beta * dvxdc21_z;
		hessian.data()[103] = vy * -dBTdc21_z - beta * dvydc21_z;
		hessian.data()[104] = vz * -dBTdc21_z - beta * dvzdc21_z;

		hessian.data()[105] = vx * dBTdc21_z + (beta - 1.0) * dvxdc21_z;
		hessian.data()[106] = vy * dBTdc21_z + (beta - 1.0) * dvydc21_z;
		hessian.data()[107] = vz * dBTdc21_z + (beta - 1.0) * dvzdc21_z;

		// f / c22x
		hessian.data()[108] = vx * dALdc22_x + alpha * dvxdc22_x;
		hessian.data()[109] = vy * dALdc22_x + alpha * dvydc22_x;
		hessian.data()[110] = vz * dALdc22_x + alpha * dvzdc22_x;

		hessian.data()[111] = vx * -dALdc22_x + (1 - alpha) * dvxdc22_x;
		hessian.data()[112] = vy * -dALdc22_x + (1 - alpha) * dvydc22_x;
		hessian.data()[113] = vz * -dALdc22_x + (1 - alpha) * dvzdc22_x;

		hessian.data()[114] = vx * -dBTdc22_x - beta * dvxdc22_x;
		hessian.data()[115] = vy * -dBTdc22_x - beta * dvydc22_x;
		hessian.data()[116] = vz * -dBTdc22_x - beta * dvzdc22_x;

		hessian.data()[117] = vx * dBTdc22_x + (beta - 1.0) * dvxdc22_x;
		hessian.data()[118] = vy * dBTdc22_x + (beta - 1.0) * dvydc22_x;
		hessian.data()[119] = vz * dBTdc22_x + (beta - 1.0) * dvzdc22_x;
		// f / c22y
		hessian.data()[120] = vx * dALdc22_y + alpha * dvxdc22_y;
		hessian.data()[121] = vy * dALdc22_y + alpha * dvydc22_y;
		hessian.data()[122] = vz * dALdc22_y + alpha * dvzdc22_y;

		hessian.data()[123] = vx * -dALdc22_y + (1 - alpha) * dvxdc22_y;
		hessian.data()[124] = vy * -dALdc22_y + (1 - alpha) * dvydc22_y;
		hessian.data()[125] = vz * -dALdc22_y + (1 - alpha) * dvzdc22_y;

		hessian.data()[126] = vx * -dBTdc22_y - beta * dvxdc22_y;
		hessian.data()[127] = vy * -dBTdc22_y - beta * dvydc22_y;
		hessian.data()[128] = vz * -dBTdc22_y - beta * dvzdc22_y;

		hessian.data()[129] = vx * dBTdc22_y + (beta - 1.0) * dvxdc22_y;
		hessian.data()[130] = vy * dBTdc22_y + (beta - 1.0) * dvydc22_y;
		hessian.data()[131] = vz * dBTdc22_y + (beta - 1.0) * dvzdc22_y;
		// f / c22z
		hessian.data()[132] = vx * dALdc22_z + alpha * dvxdc22_z;
		hessian.data()[133] = vy * dALdc22_z + alpha * dvydc22_z;
		hessian.data()[134] = vz * dALdc22_z + alpha * dvzdc22_z;

		hessian.data()[135] = vx * -dALdc22_z + (1 - alpha) * dvxdc22_z;
		hessian.data()[136] = vy * -dALdc22_z + (1 - alpha) * dvydc22_z;
		hessian.data()[137] = vz * -dALdc22_z + (1 - alpha) * dvzdc22_z;

		hessian.data()[138] = vx * -dBTdc22_z - beta * dvxdc22_z;
		hessian.data()[139] = vy * -dBTdc22_z - beta * dvydc22_z;
		hessian.data()[140] = vz * -dBTdc22_z - beta * dvzdc22_z;

		hessian.data()[141] = vx * dBTdc22_z + (beta - 1.0) * dvxdc22_z;
		hessian.data()[142] = vy * dBTdc22_z + (beta - 1.0) * dvydc22_z;
		hessian.data()[143] = vz * dBTdc22_z + (beta - 1.0) * dvzdc22_z;
	}

	scalar MipcSlabSphereConstraint::computeDistance()
	{
		sC1 = spheres[0].center - spheres[2].center;
		sC2 = spheres[1].center - spheres[2].center;
		sC3 = spheres[2].center - spheres[3].center;

		sR1 = spheres[0].radius - spheres[2].radius;
		sR2 = spheres[1].radius - spheres[2].radius;
		sR3 = spheres[2].radius + spheres[3].radius;

		A = sC1.dot(sC1) - sR1 * sR1;
		B = 2.0 * (sC1.dot(sC2) - sR1 * sR2);
		C = sC2.dot(sC2) - sR2 * sR2;
		D = 2.0 * (sC1.dot(sC3) - sR1 * sR3);
		E = 2.0 * (sC2.dot(sC3) - sR2 * sR3);
		F = sC3.dot(sC3) - sR3 * sR3;

		delta = 4 * A * C - B * B;

		scalar temp_alpha, temp_beta;
		scalar temp_dist;
		distanceMode = TWO_ENDPOINTS;
		alpha = 0.0, beta = 0.0;
		distance = valueOfQuadraticSurface2D(alpha, beta, A, B, C, D, E, F);

		temp_alpha = 1.0;
		temp_beta = 0.0;
		temp_dist = valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
		if (distance > temp_dist)
		{
			distance = temp_dist;
			alpha = temp_alpha;
			beta = temp_beta;
		}

		temp_alpha = 0.0;
		temp_beta = 1.0;
		temp_dist = valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
		if (distance > temp_dist)
		{
			distance = temp_dist;
			alpha = temp_alpha;
			beta = temp_beta;
		}

		temp_alpha = 0.0;
		temp_beta = -E / (2.0 * C);
		if (temp_beta > 0.0 && temp_beta < 1.0)
		{
			temp_dist = valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
			if (distance > temp_dist)
			{
				distance = temp_dist;
				alpha = temp_alpha;
				beta = temp_beta;
				distanceMode = ALPHA_ZERO;
			}
		}

		temp_alpha = -D / (2.0 * A);
		temp_beta = 0.0;
		if (temp_alpha > 0.0 && temp_alpha < 1.0)
		{
			temp_dist = valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
			if (distance > temp_dist)
			{
				distance = temp_dist;
				alpha = temp_alpha;
				beta = temp_beta;
				distanceMode = BETA_ZERO;
			}
		}

		temp_alpha = 0.5 * (2.0 * C + E - B - D) / (A - B + C);
		temp_beta = 1.0 - temp_alpha;
		if (temp_alpha > 0.0 && temp_alpha < 1.0)
		{
			temp_dist = valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
			if (distance > temp_dist)
			{
				distance = temp_dist;
				alpha = temp_alpha;
				beta = temp_beta;
				distanceMode = ALPHA_BETA_ONE;
			}
		}

		// can be ignored
		if (delta != 0.0)
		{
			temp_alpha = (B * E - 2.0 * C * D) / delta;
			temp_beta = (B * D - 2.0 * A * E) / delta;
			if (temp_alpha > 0.0 && temp_alpha < 1.0 && temp_beta > 0.0 && temp_beta < 1.0 && temp_alpha + temp_beta < 1.0)
			{
				temp_dist = valueOfQuadraticSurface2D(temp_alpha, temp_beta, A, B, C, D, E, F);
				if (distance > temp_dist)
				{
					distance = temp_dist;
					alpha = temp_alpha;
					beta = temp_beta;
					distanceMode = ALPHA_BETA_ONE;
				}
			}
		}

		vec3 cp, cq;
		cp = alpha * spheres[0].center;
		cq = spheres[3].center;
		

		return distance;
	}
	
	void MipcSlabSphereConstraint::get_dist_gh(vec12 &g12, mat12 &h12)
	{
		vec12 distGrad;
		distGrad.resize(12);
		distGrad.setZero();
		mat12 distHessian;
		distHessian.resize(12, 12);
		distHessian.setZero();
		diff_F_x(distGrad);
		g12 = -distGrad;


		switch (distanceMode)
		{
		case TWO_ENDPOINTS:
			endPointsHessian(distHessian);
			break;
		case ALPHA_ZERO:
			alphaIsZeroHessian(distHessian);
			break;
		case BETA_ZERO:
			betaIsZeroHessian(distHessian);
			break;
		case ALPHA_BETA_ONE:
			alphaBetaPlusOneHessian(distHessian);
			break;
		case ALPHA_BETA:
			alphaBetaHessian(distHessian);
			break;
		default:
			break;
		};

		h12 = -distHessian;
		makePD(h12);
	}

	void MipcSlabSphereConstraint::diff_F_x(vec12 &diff)
	{
		vec3 v = 2.0 * (alpha * sC1 + beta * sC2 + sC3);

		diff[0] = (alpha * v.data()[0]);
		diff[1] = (alpha * v.data()[1]);
		diff[2] = (alpha * v.data()[2]);

		diff[3] = (beta * v.data()[0]);
		diff[4] = (beta * v.data()[1]);
		diff[5] = (beta * v.data()[2]);

		diff[6] = ((1.0 - alpha - beta) * v.data()[0]);
		diff[7] = ((1.0 - alpha - beta) * v.data()[1]);
		diff[8] = ((1.0 - alpha - beta) * v.data()[2]);

		diff[9] = (-v.data()[0]);
		diff[10] = (-v.data()[1]);
		diff[11] = (-v.data()[2]);
	}

	void MipcSlabSphereConstraint::endPointsHessian(mat12 &hessian)
	{
		// alpha and beta is constant
		scalar C1x = sC1.data()[0];
		scalar C1y = sC1.data()[1];
		scalar C1z = sC1.data()[2];
		scalar C2x = sC2.data()[0];
		scalar C2y = sC2.data()[1];
		scalar C2z = sC2.data()[2];
		scalar C3x = sC3.data()[0];
		scalar C3y = sC3.data()[1];
		scalar C3z = sC3.data()[2];

		scalar vx = 2.0 * (alpha * C1x + beta * C2x + C3x);
		scalar vy = 2.0 * (alpha * C1y + beta * C2y + C3y);
		scalar vz = 2.0 * (alpha * C1z + beta * C2z + C3z);

		scalar dvxdc11_x = 2.0 * alpha;
		scalar dvxdc11_y = 0;
		scalar dvxdc11_z = 0;

		scalar dvxdc12_x = 2.0 * beta;
		scalar dvxdc12_y = 0;
		scalar dvxdc12_z = 0;

		scalar dvxdc13_x = 2.0 * (1.0 - alpha - beta);
		scalar dvxdc13_y = 0;
		scalar dvxdc13_z = 0;

		scalar dvxdcp_x = 2.0 * -1.0;
		scalar dvxdcp_y = 0;
		scalar dvxdcp_z = 0;
		//
		scalar dvydc11_x = 0;
		scalar dvydc11_y = 2.0 * alpha;
		scalar dvydc11_z = 0;

		scalar dvydc12_x = 0;
		scalar dvydc12_y = 2.0 * beta;
		scalar dvydc12_z = 0;

		scalar dvydc13_x = 0;
		scalar dvydc13_y = 2.0 * (1.0 - alpha - beta);
		scalar dvydc13_z = 0;

		scalar dvydcp_x = 0;
		scalar dvydcp_y = 2.0 * -1.0;
		scalar dvydcp_z = 0;
		//
		scalar dvzdc11_x = 0;
		scalar dvzdc11_y = 0;
		scalar dvzdc11_z = 2.0 * alpha;

		scalar dvzdc12_x = 0;
		scalar dvzdc12_y = 0;
		scalar dvzdc12_z = 2.0 * beta;

		scalar dvzdc13_x = 0;
		scalar dvzdc13_y = 0;
		scalar dvzdc13_z = 2.0 * (1.0 - alpha - beta);

		scalar dvzdcp_x = 0;
		scalar dvzdcp_y = 0;
		scalar dvzdcp_z = 2.0 * -1.0;

		hessian.setZero();
		// f / c11x
		hessian.data()[0] = alpha * dvxdc11_x;
		hessian.data()[1] = alpha * dvydc11_x;
		hessian.data()[2] = alpha * dvzdc11_x;

		hessian.data()[3] = beta * dvxdc11_x;
		hessian.data()[4] = beta * dvydc11_x;
		hessian.data()[5] = beta * dvzdc11_x;

		hessian.data()[6] = (1.0 - alpha - beta) * dvxdc11_x;
		hessian.data()[7] = (1.0 - alpha - beta) * dvydc11_x;
		hessian.data()[8] = (1.0 - alpha - beta) * dvzdc11_x;

		hessian.data()[9] = -dvxdc11_x;
		hessian.data()[10] = -dvydc11_x;
		hessian.data()[11] = -dvzdc11_x;
		// f / c11y
		hessian.data()[12] = alpha * dvxdc11_y;
		hessian.data()[13] = alpha * dvydc11_y;
		hessian.data()[14] = alpha * dvzdc11_y;

		hessian.data()[15] = beta * dvxdc11_y;
		hessian.data()[16] = beta * dvydc11_y;
		hessian.data()[17] = beta * dvzdc11_y;

		hessian.data()[18] = (1.0 - alpha - beta) * dvxdc11_y;
		hessian.data()[19] = (1.0 - alpha - beta) * dvydc11_y;
		hessian.data()[20] = (1.0 - alpha - beta) * dvzdc11_y;

		hessian.data()[21] = -dvxdc11_y;
		hessian.data()[22] = -dvydc11_y;
		hessian.data()[23] = -dvzdc11_y;
		// f / c11z
		hessian.data()[24] = alpha * dvxdc11_z;
		hessian.data()[25] = alpha * dvydc11_z;
		hessian.data()[26] = alpha * dvzdc11_z;

		hessian.data()[27] = beta * dvxdc11_z;
		hessian.data()[28] = beta * dvydc11_z;
		hessian.data()[29] = beta * dvzdc11_z;

		hessian.data()[30] = (1.0 - alpha - beta) * dvxdc11_z;
		hessian.data()[31] = (1.0 - alpha - beta) * dvydc11_z;
		hessian.data()[32] = (1.0 - alpha - beta) * dvzdc11_z;

		hessian.data()[33] = -dvxdc11_z;
		hessian.data()[34] = -dvydc11_z;
		hessian.data()[35] = -dvzdc11_z;
		// f / c12x
		hessian.data()[36] = alpha * dvxdc12_x;
		hessian.data()[37] = alpha * dvydc12_x;
		hessian.data()[38] = alpha * dvzdc12_x;

		hessian.data()[39] = beta * dvxdc12_x;
		hessian.data()[40] = beta * dvydc12_x;
		hessian.data()[41] = beta * dvzdc12_x;

		hessian.data()[42] = (1.0 - alpha - beta) * dvxdc12_x;
		hessian.data()[43] = (1.0 - alpha - beta) * dvydc12_x;
		hessian.data()[44] = (1.0 - alpha - beta) * dvzdc12_x;

		hessian.data()[45] = -dvxdc12_x;
		hessian.data()[46] = -dvydc12_x;
		hessian.data()[47] = -dvzdc12_x;
		// f / c12y
		hessian.data()[48] = alpha * dvxdc12_y;
		hessian.data()[49] = alpha * dvydc12_y;
		hessian.data()[50] = alpha * dvzdc12_y;

		hessian.data()[51] = beta * dvxdc12_y;
		hessian.data()[52] = beta * dvydc12_y;
		hessian.data()[53] = beta * dvzdc12_y;

		hessian.data()[54] = (1.0 - alpha - beta) * dvxdc12_y;
		hessian.data()[55] = (1.0 - alpha - beta) * dvydc12_y;
		hessian.data()[56] = (1.0 - alpha - beta) * dvzdc12_y;

		hessian.data()[57] = -dvxdc12_y;
		hessian.data()[58] = -dvydc12_y;
		hessian.data()[59] = -dvzdc12_y;
		// f / c12z
		hessian.data()[60] = alpha * dvxdc12_z;
		hessian.data()[61] = alpha * dvydc12_z;
		hessian.data()[62] = alpha * dvzdc12_z;

		hessian.data()[63] = beta * dvxdc12_z;
		hessian.data()[64] = beta * dvydc12_z;
		hessian.data()[65] = beta * dvzdc12_z;

		hessian.data()[66] = (1.0 - alpha - beta) * dvxdc12_z;
		hessian.data()[67] = (1.0 - alpha - beta) * dvydc12_z;
		hessian.data()[68] = (1.0 - alpha - beta) * dvzdc12_z;

		hessian.data()[69] = -dvxdc12_z;
		hessian.data()[70] = -dvydc12_z;
		hessian.data()[71] = -dvzdc12_z;
		//	f / c13x
		hessian.data()[72] = alpha * dvxdc13_x;
		hessian.data()[73] = alpha * dvydc13_x;
		hessian.data()[74] = alpha * dvzdc13_x;

		hessian.data()[75] = beta * dvxdc13_x;
		hessian.data()[76] = beta * dvydc13_x;
		hessian.data()[77] = beta * dvzdc13_x;

		hessian.data()[78] = (1.0 - alpha - beta) * dvxdc13_x;
		hessian.data()[79] = (1.0 - alpha - beta) * dvydc13_x;
		hessian.data()[80] = (1.0 - alpha - beta) * dvzdc13_x;

		hessian.data()[81] = -dvxdc13_x;
		hessian.data()[82] = -dvydc13_x;
		hessian.data()[83] = -dvzdc13_x;
		//	f / c13y
		hessian.data()[84] = alpha * dvxdc13_y;
		hessian.data()[85] = alpha * dvydc13_y;
		hessian.data()[86] = alpha * dvzdc13_y;

		hessian.data()[87] = beta * dvxdc13_y;
		hessian.data()[88] = beta * dvydc13_y;
		hessian.data()[89] = beta * dvzdc13_y;

		hessian.data()[90] = (1.0 - alpha - beta) * dvxdc13_y;
		hessian.data()[91] = (1.0 - alpha - beta) * dvydc13_y;
		hessian.data()[92] = (1.0 - alpha - beta) * dvzdc13_y;

		hessian.data()[93] = -dvxdc13_y;
		hessian.data()[94] = -dvydc13_y;
		hessian.data()[95] = -dvzdc13_y;
		//	f / c13z
		hessian.data()[96] = alpha * dvxdc13_z;
		hessian.data()[97] = alpha * dvydc13_z;
		hessian.data()[98] = alpha * dvzdc13_z;

		hessian.data()[99] = beta * dvxdc13_z;
		hessian.data()[100] = beta * dvydc13_z;
		hessian.data()[101] = beta * dvzdc13_z;

		hessian.data()[102] = (1.0 - alpha - beta) * dvxdc13_z;
		hessian.data()[103] = (1.0 - alpha - beta) * dvydc13_z;
		hessian.data()[104] = (1.0 - alpha - beta) * dvzdc13_z;

		hessian.data()[105] = -dvxdc13_z;
		hessian.data()[106] = -dvydc13_z;
		hessian.data()[107] = -dvzdc13_z;
		//	f / cpx
		hessian.data()[108] = alpha * dvxdcp_x;
		hessian.data()[109] = alpha * dvydcp_x;
		hessian.data()[110] = alpha * dvzdcp_x;

		hessian.data()[111] = beta * dvxdcp_x;
		hessian.data()[112] = beta * dvydcp_x;
		hessian.data()[113] = beta * dvzdcp_x;

		hessian.data()[114] = (1.0 - alpha - beta) * dvxdcp_x;
		hessian.data()[115] = (1.0 - alpha - beta) * dvydcp_x;
		hessian.data()[116] = (1.0 - alpha - beta) * dvzdcp_x;

		hessian.data()[117] = -dvxdcp_x;
		hessian.data()[118] = -dvydcp_x;
		hessian.data()[119] = -dvzdcp_x;
		//	f / cpy
		hessian.data()[120] = alpha * dvxdcp_y;
		hessian.data()[121] = alpha * dvydcp_y;
		hessian.data()[122] = alpha * dvzdcp_y;

		hessian.data()[123] = beta * dvxdcp_y;
		hessian.data()[124] = beta * dvydcp_y;
		hessian.data()[125] = beta * dvzdcp_y;

		hessian.data()[126] = (1.0 - alpha - beta) * dvxdcp_y;
		hessian.data()[127] = (1.0 - alpha - beta) * dvydcp_y;
		hessian.data()[128] = (1.0 - alpha - beta) * dvzdcp_y;

		hessian.data()[129] = -dvxdcp_y;
		hessian.data()[130] = -dvydcp_y;
		hessian.data()[131] = -dvzdcp_y;
		//	f / cpz
		hessian.data()[132] = alpha * dvxdcp_z;
		hessian.data()[133] = alpha * dvydcp_z;
		hessian.data()[134] = alpha * dvzdcp_z;

		hessian.data()[135] = beta * dvxdcp_z;
		hessian.data()[136] = beta * dvydcp_z;
		hessian.data()[137] = beta * dvzdcp_z;

		hessian.data()[138] = (1.0 - alpha - beta) * dvxdcp_z;
		hessian.data()[139] = (1.0 - alpha - beta) * dvydcp_z;
		hessian.data()[140] = (1.0 - alpha - beta) * dvzdcp_z;

		hessian.data()[141] = -dvxdcp_z;
		hessian.data()[142] = -dvydcp_z;
		hessian.data()[143] = -dvzdcp_z;
	}

	void MipcSlabSphereConstraint::alphaIsZeroHessian(mat12 &hessian)
	{
		scalar dAdc11_x = 2.0 * sC1.data()[0];
		scalar dAdc11_y = 2.0 * sC1.data()[1];
		scalar dAdc11_z = 2.0 * sC1.data()[2];
		scalar dAdc13_x = -2.0 * sC1.data()[0];
		scalar dAdc13_y = -2.0 * sC1.data()[1];
		scalar dAdc13_z = -2.0 * sC1.data()[2];

		scalar dBdc11_x = 2.0 * sC2.data()[0];
		scalar dBdc11_y = 2.0 * sC2.data()[1];
		scalar dBdc11_z = 2.0 * sC2.data()[2];
		scalar dBdc12_x = 2.0 * sC1.data()[0];
		scalar dBdc12_y = 2.0 * sC1.data()[1];
		scalar dBdc12_z = 2.0 * sC1.data()[2];
		scalar dBdc13_x = -2.0 * (sC1.data()[0] + sC2.data()[0]);
		scalar dBdc13_y = -2.0 * (sC1.data()[1] + sC2.data()[1]);
		scalar dBdc13_z = -2.0 * (sC1.data()[2] + sC2.data()[2]);

		scalar dCdc12_x = 2.0 * sC2.data()[0];
		scalar dCdc12_y = 2.0 * sC2.data()[1];
		scalar dCdc12_z = 2.0 * sC2.data()[2];
		scalar dCdc13_x = -2.0 * sC2.data()[0];
		scalar dCdc13_y = -2.0 * sC2.data()[1];
		scalar dCdc13_z = -2.0 * sC2.data()[2];

		scalar dDdc11_x = 2.0 * sC3.data()[0];
		scalar dDdc11_y = 2.0 * sC3.data()[1];
		scalar dDdc11_z = 2.0 * sC3.data()[2];
		scalar dDdc13_x = 2.0 * (sC1.data()[0] - sC3.data()[0]);
		scalar dDdc13_y = 2.0 * (sC1.data()[1] - sC3.data()[1]);
		scalar dDdc13_z = 2.0 * (sC1.data()[2] - sC3.data()[2]);
		scalar dDdcp_x = -2.0 * sC1.data()[0];
		scalar dDdcp_y = -2.0 * sC1.data()[1];
		scalar dDdcp_z = -2.0 * sC1.data()[2];

		scalar dEdc12_x = 2.0 * sC3.data()[0];
		scalar dEdc12_y = 2.0 * sC3.data()[1];
		scalar dEdc12_z = 2.0 * sC3.data()[2];
		scalar dEdc13_x = 2.0 * (sC2.data()[0] - sC3.data()[0]);
		scalar dEdc13_y = 2.0 * (sC2.data()[1] - sC3.data()[1]);
		scalar dEdc13_z = 2.0 * (sC2.data()[2] - sC3.data()[2]);
		scalar dEdcp_x = -2.0 * sC2.data()[0];
		scalar dEdcp_y = -2.0 * sC2.data()[1];
		scalar dEdcp_z = -2.0 * sC2.data()[2];

		scalar dFdc13_x = 2.0 * sC3.data()[0];
		scalar dFdc13_y = 2.0 * sC3.data()[1];
		scalar dFdc13_z = 2.0 * sC3.data()[2];
		scalar dFdcp_x = -2.0 * sC3.data()[0];
		scalar dFdcp_y = -2.0 * sC3.data()[1];
		scalar dFdcp_z = -2.0 * sC3.data()[2];

		//
		scalar q1 = 0.5 * E / (C * C);
		scalar q2 = -0.5 / C;

		scalar dBTdc12_x = q1 * dCdc12_x + q2 * dEdc12_x;
		scalar dBTdc12_y = q1 * dCdc12_y + q2 * dEdc12_y;
		scalar dBTdc12_z = q1 * dCdc12_z + q2 * dEdc12_z;

		scalar dBTdc13_x = q1 * dCdc13_x + q2 * dEdc13_x;
		scalar dBTdc13_y = q1 * dCdc13_y + q2 * dEdc13_y;
		scalar dBTdc13_z = q1 * dCdc13_z + q2 * dEdc13_z;

		scalar dBTdcp_x = q2 * dEdcp_x;
		scalar dBTdcp_y = q2 * dEdcp_y;
		scalar dBTdcp_z = q2 * dEdcp_z;

		scalar C1x = sC1.data()[0];
		scalar C1y = sC1.data()[1];
		scalar C1z = sC1.data()[2];
		scalar C2x = sC2.data()[0];
		scalar C2y = sC2.data()[1];
		scalar C2z = sC2.data()[2];
		scalar C3x = sC3.data()[0];
		scalar C3y = sC3.data()[1];
		scalar C3z = sC3.data()[2];

		scalar vx = 2.0 * (beta * C2x + C3x);
		scalar vy = 2.0 * (beta * C2y + C3y);
		scalar vz = 2.0 * (beta * C2z + C3z);

		scalar dvxdc12_x = 2.0 * (C2x * dBTdc12_x + beta);
		scalar dvxdc12_y = 2.0 * (C2x * dBTdc12_y);
		scalar dvxdc12_z = 2.0 * (C2x * dBTdc12_z);

		scalar dvxdc13_x = 2.0 * (C2x * dBTdc13_x + 1 - beta);
		scalar dvxdc13_y = 2.0 * (C2x * dBTdc13_y);
		scalar dvxdc13_z = 2.0 * (C2x * dBTdc13_z);

		scalar dvxdcp_x = 2.0 * (C2x * dBTdcp_x - 1);
		scalar dvxdcp_y = 2.0 * (C2x * dBTdcp_y);
		scalar dvxdcp_z = 2.0 * (C2x * dBTdcp_z);
		//

		scalar dvydc12_x = 2.0 * (C2y * dBTdc12_x);
		scalar dvydc12_y = 2.0 * (C2y * dBTdc12_y + beta);
		scalar dvydc12_z = 2.0 * (C2y * dBTdc12_z);

		scalar dvydc13_x = 2.0 * (C2y * dBTdc13_x);
		scalar dvydc13_y = 2.0 * (C2y * dBTdc13_y + 1 - beta);
		scalar dvydc13_z = 2.0 * (C2y * dBTdc13_z);

		scalar dvydcp_x = 2.0 * (C2y * dBTdcp_x);
		scalar dvydcp_y = 2.0 * (C2y * dBTdcp_y - 1);
		scalar dvydcp_z = 2.0 * (C2y * dBTdcp_z);
		//

		scalar dvzdc12_x = 2.0 * (C2z * dBTdc12_x);
		scalar dvzdc12_y = 2.0 * (C2z * dBTdc12_y);
		scalar dvzdc12_z = 2.0 * (C2z * dBTdc12_z + beta);

		scalar dvzdc13_x = 2.0 * (C2z * dBTdc13_x);
		scalar dvzdc13_y = 2.0 * (C2z * dBTdc13_y);
		scalar dvzdc13_z = 2.0 * (C2z * dBTdc13_z + 1 - beta);

		scalar dvzdcp_x = 2.0 * (C2z * dBTdcp_x);
		scalar dvzdcp_y = 2.0 * (C2z * dBTdcp_y);
		scalar dvzdcp_z = 2.0 * (C2z * dBTdcp_z - 1);

		hessian.setZero();
		// f / c11x
		hessian.data()[0] = 0;
		hessian.data()[1] = 0;
		hessian.data()[2] = 0;

		hessian.data()[3] = 0;
		hessian.data()[4] = 0;
		hessian.data()[5] = 0;

		hessian.data()[6] = 0;
		hessian.data()[7] = 0;
		hessian.data()[8] = 0;

		hessian.data()[9] = 0;
		hessian.data()[10] = 0;
		hessian.data()[11] = 0;
		// f / c11y
		hessian.data()[12] = 0;
		hessian.data()[13] = 0;
		hessian.data()[14] = 0;

		hessian.data()[15] = 0;
		hessian.data()[16] = 0;
		hessian.data()[17] = 0;

		hessian.data()[18] = 0;
		hessian.data()[19] = 0;
		hessian.data()[20] = 0;

		hessian.data()[21] = 0;
		hessian.data()[22] = 0;
		hessian.data()[23] = 0;
		// f / c11z
		hessian.data()[24] = 0;
		hessian.data()[25] = 0;
		hessian.data()[26] = 0;

		hessian.data()[27] = 0;
		hessian.data()[28] = 0;
		hessian.data()[29] = 0;

		hessian.data()[30] = 0;
		hessian.data()[31] = 0;
		hessian.data()[32] = 0;

		hessian.data()[33] = 0;
		hessian.data()[34] = 0;
		hessian.data()[35] = 0;
		// f / c12x
		hessian.data()[36] = 0;
		hessian.data()[37] = 0;
		hessian.data()[38] = 0;

		hessian.data()[39] = vx * dBTdc12_x + beta * dvxdc12_x;
		hessian.data()[40] = vy * dBTdc12_x + beta * dvydc12_x; //
		hessian.data()[41] = vz * dBTdc12_x + beta * dvzdc12_x;

		hessian.data()[42] = vx * (-dBTdc12_x) + (1.0 - beta) * dvxdc12_x;
		hessian.data()[43] = vy * (-dBTdc12_x) + (1.0 - beta) * dvydc12_x;
		hessian.data()[44] = vz * (-dBTdc12_x) + (1.0 - beta) * dvzdc12_x;

		hessian.data()[45] = -dvxdc12_x;
		hessian.data()[46] = -dvydc12_x;
		hessian.data()[47] = -dvzdc12_x;
		// f / c12y
		hessian.data()[48] = 0;
		hessian.data()[49] = 0;
		hessian.data()[50] = 0;

		hessian.data()[51] = vx * dBTdc12_y + beta * dvxdc12_y; //
		hessian.data()[52] = vy * dBTdc12_y + beta * dvydc12_y;
		hessian.data()[53] = vz * dBTdc12_y + beta * dvzdc12_y;

		hessian.data()[54] = vx * (-dBTdc12_y) + (1.0 - beta) * dvxdc12_y;
		hessian.data()[55] = vy * (-dBTdc12_y) + (1.0 - beta) * dvydc12_y;
		hessian.data()[56] = vz * (-dBTdc12_y) + (1.0 - beta) * dvzdc12_y;

		hessian.data()[57] = -dvxdc12_y;
		hessian.data()[58] = -dvydc12_y;
		hessian.data()[59] = -dvzdc12_y;
		// f / c12z
		hessian.data()[60] = 0;
		hessian.data()[61] = 0;
		hessian.data()[62] = 0;

		hessian.data()[63] = vx * dBTdc12_z + beta * dvxdc12_z;
		hessian.data()[64] = vy * dBTdc12_z + beta * dvydc12_z;
		hessian.data()[65] = vz * dBTdc12_z + beta * dvzdc12_z;

		hessian.data()[66] = vx * (-dBTdc12_z) + (1.0 - beta) * dvxdc12_z;
		hessian.data()[67] = vy * (-dBTdc12_z) + (1.0 - beta) * dvydc12_z;
		hessian.data()[68] = vz * (-dBTdc12_z) + (1.0 - beta) * dvzdc12_z;

		hessian.data()[69] = -dvxdc12_z;
		hessian.data()[70] = -dvydc12_z;
		hessian.data()[71] = -dvzdc12_z;
		//	f / c13x
		hessian.data()[72] = 0;
		hessian.data()[73] = 0;
		hessian.data()[74] = 0;

		hessian.data()[75] = vx * dBTdc13_x + beta * dvxdc13_x;
		hessian.data()[76] = vy * dBTdc13_x + beta * dvydc13_x;
		hessian.data()[77] = vz * dBTdc13_x + beta * dvzdc13_x;

		hessian.data()[78] = vx * (-dBTdc13_x) + (1.0 - beta) * dvxdc13_x;
		hessian.data()[79] = vy * (-dBTdc13_x) + (1.0 - beta) * dvydc13_x;
		hessian.data()[80] = vz * (-dBTdc13_x) + (1.0 - beta) * dvzdc13_x;

		hessian.data()[81] = -dvxdc13_x;
		hessian.data()[82] = -dvydc13_x;
		hessian.data()[83] = -dvzdc13_x;
		//	f / c13y
		hessian.data()[84] = 0;
		hessian.data()[85] = 0;
		hessian.data()[86] = 0;

		hessian.data()[87] = vx * dBTdc13_y + beta * dvxdc13_y;
		hessian.data()[88] = vy * dBTdc13_y + beta * dvydc13_y;
		hessian.data()[89] = vz * dBTdc13_y + beta * dvzdc13_y;

		hessian.data()[90] = vx * (-dBTdc13_y) + (1.0 - beta) * dvxdc13_y;
		hessian.data()[91] = vy * (-dBTdc13_y) + (1.0 - beta) * dvydc13_y;
		hessian.data()[92] = vz * (-dBTdc13_y) + (1.0 - beta) * dvzdc13_y;

		hessian.data()[93] = -dvxdc13_y;
		hessian.data()[94] = -dvydc13_y;
		hessian.data()[95] = -dvzdc13_y;
		//	f / c13z
		hessian.data()[96] = 0;
		hessian.data()[97] = 0;
		hessian.data()[98] = 0;

		hessian.data()[99] = vx * dBTdc13_z + beta * dvxdc13_z;
		hessian.data()[100] = vy * dBTdc13_z + beta * dvydc13_z;
		hessian.data()[101] = vz * dBTdc13_z + beta * dvzdc13_z;

		hessian.data()[102] = vx * (-dBTdc13_z) + (1.0 - beta) * dvxdc13_z;
		hessian.data()[103] = vy * (-dBTdc13_z) + (1.0 - beta) * dvydc13_z;
		hessian.data()[104] = vz * (-dBTdc13_z) + (1.0 - beta) * dvzdc13_z;

		hessian.data()[105] = -dvxdc13_z;
		hessian.data()[106] = -dvydc13_z;
		hessian.data()[107] = -dvzdc13_z;
		//	f / cpx
		hessian.data()[108] = 0;
		hessian.data()[109] = 0;
		hessian.data()[110] = 0;

		hessian.data()[111] = vx * dBTdcp_x + beta * dvxdcp_x;
		hessian.data()[112] = vy * dBTdcp_x + beta * dvydcp_x;
		hessian.data()[113] = vz * dBTdcp_x + beta * dvzdcp_x;

		hessian.data()[114] = vx * (-dBTdcp_x) + (1.0 - beta) * dvxdcp_x;
		hessian.data()[115] = vy * (-dBTdcp_x) + (1.0 - beta) * dvydcp_x;
		hessian.data()[116] = vz * (-dBTdcp_x) + (1.0 - beta) * dvzdcp_x;

		hessian.data()[117] = -dvxdcp_x;
		hessian.data()[118] = -dvydcp_x;
		hessian.data()[119] = -dvzdcp_x;
		//	f / cpy
		hessian.data()[120] = 0;
		hessian.data()[121] = 0;
		hessian.data()[122] = 0;

		hessian.data()[123] = vx * dBTdcp_y + beta * dvxdcp_y;
		hessian.data()[124] = vy * dBTdcp_y + beta * dvydcp_y;
		hessian.data()[125] = vz * dBTdcp_y + beta * dvzdcp_y;

		hessian.data()[126] = vx * (-dBTdcp_y) + (1.0 - beta) * dvxdcp_y;
		hessian.data()[127] = vy * (-dBTdcp_y) + (1.0 - beta) * dvydcp_y;
		hessian.data()[128] = vz * (-dBTdcp_y) + (1.0 - beta) * dvzdcp_y;

		hessian.data()[129] = -dvxdcp_y;
		hessian.data()[130] = -dvydcp_y;
		hessian.data()[131] = -dvzdcp_y;
		//	f / cpz
		hessian.data()[132] = 0;
		hessian.data()[133] = 0;
		hessian.data()[134] = 0;

		hessian.data()[135] = vx * dBTdcp_z + beta * dvxdcp_z;
		hessian.data()[136] = vy * dBTdcp_z + beta * dvydcp_z;
		hessian.data()[137] = vz * dBTdcp_z + beta * dvzdcp_z;

		hessian.data()[138] = vx * (-dBTdcp_z) + (1.0 - beta) * dvxdcp_z;
		hessian.data()[139] = vy * (-dBTdcp_z) + (1.0 - beta) * dvydcp_z;
		hessian.data()[140] = vz * (-dBTdcp_z) + (1.0 - beta) * dvzdcp_z;

		hessian.data()[141] = -dvxdcp_z;
		hessian.data()[142] = -dvydcp_z;
		hessian.data()[143] = -dvzdcp_z;
	}

	void MipcSlabSphereConstraint::betaIsZeroHessian(mat12 &hessian)
	{
		scalar dAdc11_x = 2.0 * sC1.data()[0];
		scalar dAdc11_y = 2.0 * sC1.data()[1];
		scalar dAdc11_z = 2.0 * sC1.data()[2];
		scalar dAdc13_x = -2.0 * sC1.data()[0];
		scalar dAdc13_y = -2.0 * sC1.data()[1];
		scalar dAdc13_z = -2.0 * sC1.data()[2];

		scalar dBdc11_x = 2.0 * sC2.data()[0];
		scalar dBdc11_y = 2.0 * sC2.data()[1];
		scalar dBdc11_z = 2.0 * sC2.data()[2];
		scalar dBdc12_x = 2.0 * sC1.data()[0];
		scalar dBdc12_y = 2.0 * sC1.data()[1];
		scalar dBdc12_z = 2.0 * sC1.data()[2];
		scalar dBdc13_x = -2.0 * (sC1.data()[0] + sC2.data()[0]);
		scalar dBdc13_y = -2.0 * (sC1.data()[1] + sC2.data()[1]);
		scalar dBdc13_z = -2.0 * (sC1.data()[2] + sC2.data()[2]);

		scalar dCdc12_x = 2.0 * sC2.data()[0];
		scalar dCdc12_y = 2.0 * sC2.data()[1];
		scalar dCdc12_z = 2.0 * sC2.data()[2];
		scalar dCdc13_x = -2.0 * sC2.data()[0];
		scalar dCdc13_y = -2.0 * sC2.data()[1];
		scalar dCdc13_z = -2.0 * sC2.data()[2];

		scalar dDdc11_x = 2.0 * sC3.data()[0];
		scalar dDdc11_y = 2.0 * sC3.data()[1];
		scalar dDdc11_z = 2.0 * sC3.data()[2];
		scalar dDdc13_x = 2.0 * (sC1.data()[0] - sC3.data()[0]);
		scalar dDdc13_y = 2.0 * (sC1.data()[1] - sC3.data()[1]);
		scalar dDdc13_z = 2.0 * (sC1.data()[2] - sC3.data()[2]);
		scalar dDdcp_x = -2.0 * sC1.data()[0];
		scalar dDdcp_y = -2.0 * sC1.data()[1];
		scalar dDdcp_z = -2.0 * sC1.data()[2];

		scalar dEdc12_x = 2.0 * sC3.data()[0];
		scalar dEdc12_y = 2.0 * sC3.data()[1];
		scalar dEdc12_z = 2.0 * sC3.data()[2];
		scalar dEdc13_x = 2.0 * (sC2.data()[0] - sC3.data()[0]);
		scalar dEdc13_y = 2.0 * (sC2.data()[1] - sC3.data()[1]);
		scalar dEdc13_z = 2.0 * (sC2.data()[2] - sC3.data()[2]);
		scalar dEdcp_x = -2.0 * sC2.data()[0];
		scalar dEdcp_y = -2.0 * sC2.data()[1];
		scalar dEdcp_z = -2.0 * sC2.data()[2];

		scalar dFdc13_x = 2.0 * sC3.data()[0];
		scalar dFdc13_y = 2.0 * sC3.data()[1];
		scalar dFdc13_z = 2.0 * sC3.data()[2];
		scalar dFdcp_x = -2.0 * sC3.data()[0];
		scalar dFdcp_y = -2.0 * sC3.data()[1];
		scalar dFdcp_z = -2.0 * sC3.data()[2];

		scalar q1 = 0.5 * D / (A * A);
		scalar q2 = -0.5 / A;

		scalar dALdc11_x = q1 * dAdc11_x + q2 * dDdc11_x;
		scalar dALdc11_y = q1 * dAdc11_y + q2 * dDdc11_y;
		scalar dALdc11_z = q1 * dAdc11_z + q2 * dDdc11_z;

		scalar dALdc13_x = q1 * dAdc13_x + q2 * dDdc13_x;
		scalar dALdc13_y = q1 * dAdc13_y + q2 * dDdc13_y;
		scalar dALdc13_z = q1 * dAdc13_z + q2 * dDdc13_z;

		scalar dALdcp_x = q2 * dDdcp_x;
		scalar dALdcp_y = q2 * dDdcp_y;
		scalar dALdcp_z = q2 * dDdcp_z;

		scalar C1x = sC1.data()[0];
		scalar C1y = sC1.data()[1];
		scalar C1z = sC1.data()[2];
		scalar C2x = sC2.data()[0];
		scalar C2y = sC2.data()[1];
		scalar C2z = sC2.data()[2];
		scalar C3x = sC3.data()[0];
		scalar C3y = sC3.data()[1];
		scalar C3z = sC3.data()[2];

		scalar vx = 2.0 * (alpha * C1x + C3x);
		scalar vy = 2.0 * (alpha * C1y + C3y);
		scalar vz = 2.0 * (alpha * C1z + C3z);

		scalar dvxdc11_x = 2.0 * (C1x * dALdc11_x + alpha);
		scalar dvxdc11_y = 2.0 * (C1x * dALdc11_y);
		scalar dvxdc11_z = 2.0 * (C1x * dALdc11_z);

		scalar dvxdc13_x = 2.0 * (C1x * dALdc13_x + 1 - alpha);
		scalar dvxdc13_y = 2.0 * (C1x * dALdc13_y);
		scalar dvxdc13_z = 2.0 * (C1x * dALdc13_z);

		scalar dvxdcp_x = 2.0 * (C1x * dALdcp_x - 1);
		scalar dvxdcp_y = 2.0 * (C1x * dALdcp_y);
		scalar dvxdcp_z = 2.0 * (C1x * dALdcp_z);
		//
		scalar dvydc11_x = 2.0 * (C1y * dALdc11_x);
		scalar dvydc11_y = 2.0 * (C1y * dALdc11_y + alpha);
		scalar dvydc11_z = 2.0 * (C1y * dALdc11_z);

		scalar dvydc13_x = 2.0 * (C1y * dALdc13_x);
		scalar dvydc13_y = 2.0 * (C1y * dALdc13_y + 1 - alpha);
		scalar dvydc13_z = 2.0 * (C1y * dALdc13_z);

		scalar dvydcp_x = 2.0 * (C1y * dALdcp_x);
		scalar dvydcp_y = 2.0 * (C1y * dALdcp_y - 1);
		scalar dvydcp_z = 2.0 * (C1y * dALdcp_z);
		//
		scalar dvzdc11_x = 2.0 * (C1z * dALdc11_x);
		scalar dvzdc11_y = 2.0 * (C1z * dALdc11_y);
		scalar dvzdc11_z = 2.0 * (C1z * dALdc11_z + alpha);

		scalar dvzdc13_x = 2.0 * (C1z * dALdc13_x);
		scalar dvzdc13_y = 2.0 * (C1z * dALdc13_y);
		scalar dvzdc13_z = 2.0 * (C1z * dALdc13_z + 1 - alpha);

		scalar dvzdcp_x = 2.0 * (C1z * dALdcp_x);
		scalar dvzdcp_y = 2.0 * (C1z * dALdcp_y);
		scalar dvzdcp_z = 2.0 * (C1z * dALdcp_z - 1);

		hessian.setZero();
		// f / c11x
		hessian.data()[0] = Check_double_ZERO(vx * dALdc11_x + alpha * dvxdc11_x);
		hessian.data()[1] = Check_double_ZERO(vy * dALdc11_x + alpha * dvydc11_x);
		hessian.data()[2] = Check_double_ZERO(vz * dALdc11_x + alpha * dvzdc11_x);

		hessian.data()[3] = 0;
		hessian.data()[4] = 0;
		hessian.data()[5] = 0;

		hessian.data()[6] = Check_double_ZERO(vx * (-dALdc11_x) + (1.0 - alpha) * dvxdc11_x);
		hessian.data()[7] = Check_double_ZERO(vy * (-dALdc11_x) + (1.0 - alpha) * dvydc11_x);
		hessian.data()[8] = Check_double_ZERO(vz * (-dALdc11_x) + (1.0 - alpha) * dvzdc11_x);

		hessian.data()[9] = Check_double_ZERO(-dvxdc11_x);
		hessian.data()[10] = Check_double_ZERO(-dvydc11_x);
		hessian.data()[11] = Check_double_ZERO(-dvzdc11_x);
		// f / c11y
		hessian.data()[12] = Check_double_ZERO(vx * dALdc11_y + alpha * dvxdc11_y);
		hessian.data()[13] = Check_double_ZERO(vy * dALdc11_y + alpha * dvydc11_y);
		hessian.data()[14] = Check_double_ZERO(vz * dALdc11_y + alpha * dvzdc11_y);

		hessian.data()[15] = 0;
		hessian.data()[16] = 0;
		hessian.data()[17] = 0;

		hessian.data()[18] = Check_double_ZERO(vx * (-dALdc11_y) + (1.0 - alpha) * dvxdc11_y);
		hessian.data()[19] = Check_double_ZERO(vy * (-dALdc11_y) + (1.0 - alpha) * dvydc11_y);
		hessian.data()[20] = Check_double_ZERO(vz * (-dALdc11_y) + (1.0 - alpha) * dvzdc11_y);

		hessian.data()[21] = Check_double_ZERO(-dvxdc11_y);
		hessian.data()[22] = Check_double_ZERO(-dvydc11_y);
		hessian.data()[23] = Check_double_ZERO(-dvzdc11_y);
		// f / c11z
		hessian.data()[24] = Check_double_ZERO(vx * dALdc11_z + alpha * dvxdc11_z);
		hessian.data()[25] = Check_double_ZERO(vy * dALdc11_z + alpha * dvydc11_z);
		hessian.data()[26] = Check_double_ZERO(vz * dALdc11_z + alpha * dvzdc11_z);

		hessian.data()[27] = 0;
		hessian.data()[28] = 0;
		hessian.data()[29] = 0;

		hessian.data()[30] = Check_double_ZERO(vx * (-dALdc11_z) + (1.0 - alpha) * dvxdc11_z);
		hessian.data()[31] = Check_double_ZERO(vy * (-dALdc11_z) + (1.0 - alpha) * dvydc11_z);
		hessian.data()[32] = Check_double_ZERO(vz * (-dALdc11_z) + (1.0 - alpha) * dvzdc11_z);

		hessian.data()[33] = Check_double_ZERO(-dvxdc11_z);
		hessian.data()[34] = Check_double_ZERO(-dvydc11_z);
		hessian.data()[35] = Check_double_ZERO(-dvzdc11_z);
		// f / c12x
		hessian.data()[36] = 0;
		hessian.data()[37] = 0;
		hessian.data()[38] = 0;

		hessian.data()[39] = 0;
		hessian.data()[40] = 0;
		hessian.data()[41] = 0;

		hessian.data()[42] = 0;
		hessian.data()[43] = 0;
		hessian.data()[44] = 0;

		hessian.data()[45] = 0;
		hessian.data()[46] = -0;
		hessian.data()[47] = 0;
		// f / c12y
		hessian.data()[48] = 0;
		hessian.data()[49] = 0;
		hessian.data()[50] = 0;

		hessian.data()[51] = 0;
		hessian.data()[52] = 0;
		hessian.data()[53] = 0;

		hessian.data()[54] = 0;
		hessian.data()[55] = 0;
		hessian.data()[56] = 0;

		hessian.data()[57] = 0;
		hessian.data()[58] = 0;
		hessian.data()[59] = 0;
		// f / c12z
		hessian.data()[60] = 0;
		hessian.data()[61] = 0;
		hessian.data()[62] = 0;

		hessian.data()[63] = 0;
		hessian.data()[64] = 0;
		hessian.data()[65] = 0;

		hessian.data()[66] = 0;
		hessian.data()[67] = 0;
		hessian.data()[68] = 0;

		hessian.data()[69] = 0;
		hessian.data()[70] = 0;
		hessian.data()[71] = 0;
		//	f / c13x
		hessian.data()[72] = Check_double_ZERO(vx * dALdc13_x + alpha * dvxdc13_x);
		hessian.data()[73] = Check_double_ZERO(vy * dALdc13_x + alpha * dvydc13_x);
		hessian.data()[74] = Check_double_ZERO(vz * dALdc13_x + alpha * dvzdc13_x);

		hessian.data()[75] = 0;
		hessian.data()[76] = 0;
		hessian.data()[77] = 0;

		hessian.data()[78] = Check_double_ZERO(vx * (-dALdc13_x) + (1.0 - alpha) * dvxdc13_x);
		hessian.data()[79] = Check_double_ZERO(vy * (-dALdc13_x) + (1.0 - alpha) * dvydc13_x);
		hessian.data()[80] = Check_double_ZERO(vz * (-dALdc13_x) + (1.0 - alpha) * dvzdc13_x);

		hessian.data()[81] = Check_double_ZERO(-dvxdc13_x);
		hessian.data()[82] = Check_double_ZERO(-dvydc13_x);
		hessian.data()[83] = Check_double_ZERO(-dvzdc13_x);
		//	f / c13y
		hessian.data()[84] = Check_double_ZERO(vx * dALdc13_y + alpha * dvxdc13_y);
		hessian.data()[85] = Check_double_ZERO(vy * dALdc13_y + alpha * dvydc13_y);
		hessian.data()[86] = Check_double_ZERO(vz * dALdc13_y + alpha * dvzdc13_y);

		hessian.data()[87] = Check_double_ZERO(beta * dvxdc13_y);
		hessian.data()[88] = Check_double_ZERO(beta * dvydc13_y);
		hessian.data()[89] = Check_double_ZERO(beta * dvzdc13_y);

		hessian.data()[90] = Check_double_ZERO(vx * (-dALdc13_y) + (1.0 - alpha) * dvxdc13_y);
		hessian.data()[91] = Check_double_ZERO(vy * (-dALdc13_y) + (1.0 - alpha) * dvydc13_y);
		hessian.data()[92] = Check_double_ZERO(vz * (-dALdc13_y) + (1.0 - alpha) * dvzdc13_y);

		hessian.data()[93] = Check_double_ZERO(-dvxdc13_y);
		hessian.data()[94] = Check_double_ZERO(-dvydc13_y);
		hessian.data()[95] = Check_double_ZERO(-dvzdc13_y);
		//	f / c13z
		hessian.data()[96] = Check_double_ZERO(vx * dALdc13_z + alpha * dvxdc13_z);
		hessian.data()[97] = Check_double_ZERO(vy * dALdc13_z + alpha * dvydc13_z);
		hessian.data()[98] = Check_double_ZERO(vz * dALdc13_z + alpha * dvzdc13_z);

		hessian.data()[99] = Check_double_ZERO(beta * dvxdc13_z);
		hessian.data()[100] = Check_double_ZERO(beta * dvydc13_z);
		hessian.data()[101] = Check_double_ZERO(beta * dvzdc13_z);

		hessian.data()[102] = Check_double_ZERO(vx * (-dALdc13_z) + (1.0 - alpha) * dvxdc13_z);
		hessian.data()[103] = Check_double_ZERO(vy * (-dALdc13_z) + (1.0 - alpha) * dvydc13_z);
		hessian.data()[104] = Check_double_ZERO(vz * (-dALdc13_z) + (1.0 - alpha) * dvzdc13_z);

		hessian.data()[105] = Check_double_ZERO(-dvxdc13_z);
		hessian.data()[106] = Check_double_ZERO(-dvydc13_z);
		hessian.data()[107] = Check_double_ZERO(-dvzdc13_z);
		//	f / cpx
		hessian.data()[108] = Check_double_ZERO(vx * dALdcp_x + alpha * dvxdcp_x);
		hessian.data()[109] = Check_double_ZERO(vy * dALdcp_x + alpha * dvydcp_x);
		hessian.data()[110] = Check_double_ZERO(vz * dALdcp_x + alpha * dvzdcp_x);

		hessian.data()[111] = Check_double_ZERO(beta * dvxdcp_x);
		hessian.data()[112] = Check_double_ZERO(beta * dvydcp_x);
		hessian.data()[113] = Check_double_ZERO(beta * dvzdcp_x);

		hessian.data()[114] = Check_double_ZERO(vx * (-dALdcp_x) + (1.0 - alpha) * dvxdcp_x);
		hessian.data()[115] = Check_double_ZERO(vy * (-dALdcp_x) + (1.0 - alpha) * dvydcp_x);
		hessian.data()[116] = Check_double_ZERO(vz * (-dALdcp_x) + (1.0 - alpha) * dvzdcp_x);

		hessian.data()[117] = Check_double_ZERO(-dvxdcp_x);
		hessian.data()[118] = Check_double_ZERO(-dvydcp_x);
		hessian.data()[119] = Check_double_ZERO(-dvzdcp_x);
		//	f / cpy
		hessian.data()[120] = Check_double_ZERO(vx * dALdcp_y + alpha * dvxdcp_y);
		hessian.data()[121] = Check_double_ZERO(vy * dALdcp_y + alpha * dvydcp_y);
		hessian.data()[122] = Check_double_ZERO(vz * dALdcp_y + alpha * dvzdcp_y);

		hessian.data()[123] = Check_double_ZERO(beta * dvxdcp_y);
		hessian.data()[124] = Check_double_ZERO(beta * dvydcp_y);
		hessian.data()[125] = Check_double_ZERO(beta * dvzdcp_y);

		hessian.data()[126] = Check_double_ZERO(vx * (-dALdcp_y) + (1.0 - alpha) * dvxdcp_y);
		hessian.data()[127] = Check_double_ZERO(vy * (-dALdcp_y) + (1.0 - alpha) * dvydcp_y);
		hessian.data()[128] = Check_double_ZERO(vz * (-dALdcp_y) + (1.0 - alpha) * dvzdcp_y);

		hessian.data()[129] = Check_double_ZERO(-dvxdcp_y);
		hessian.data()[130] = Check_double_ZERO(-dvydcp_y);
		hessian.data()[131] = Check_double_ZERO(-dvzdcp_y);
		//	f / cpz
		hessian.data()[132] = Check_double_ZERO(vx * dALdcp_z + alpha * dvxdcp_z);
		hessian.data()[133] = Check_double_ZERO(vy * dALdcp_z + alpha * dvydcp_z);
		hessian.data()[134] = Check_double_ZERO(vz * dALdcp_z + alpha * dvzdcp_z);

		hessian.data()[135] = Check_double_ZERO(beta * dvxdcp_z);
		hessian.data()[136] = Check_double_ZERO(beta * dvydcp_z);
		hessian.data()[137] = Check_double_ZERO(beta * dvzdcp_z);

		hessian.data()[138] = Check_double_ZERO(vx * (-dALdcp_z) + (1.0 - alpha) * dvxdcp_z);
		hessian.data()[139] = Check_double_ZERO(vy * (-dALdcp_z) + (1.0 - alpha) * dvydcp_z);
		hessian.data()[140] = Check_double_ZERO(vz * (-dALdcp_z) + (1.0 - alpha) * dvzdcp_z);

		hessian.data()[141] = Check_double_ZERO(-dvxdcp_z);
		hessian.data()[142] = Check_double_ZERO(-dvydcp_z);
		hessian.data()[143] = Check_double_ZERO(-dvzdcp_z);
	}

	void MipcSlabSphereConstraint::alphaBetaPlusOneHessian(mat12 &hessian)
	{
		scalar dAdc11_x = 2.0 * sC1.data()[0];
		scalar dAdc11_y = 2.0 * sC1.data()[1];
		scalar dAdc11_z = 2.0 * sC1.data()[2];
		scalar dAdc13_x = -2.0 * sC1.data()[0];
		scalar dAdc13_y = -2.0 * sC1.data()[1];
		scalar dAdc13_z = -2.0 * sC1.data()[2];

		scalar dBdc11_x = 2.0 * sC2.data()[0];
		scalar dBdc11_y = 2.0 * sC2.data()[1];
		scalar dBdc11_z = 2.0 * sC2.data()[2];
		scalar dBdc12_x = 2.0 * sC1.data()[0];
		scalar dBdc12_y = 2.0 * sC1.data()[1];
		scalar dBdc12_z = 2.0 * sC1.data()[2];
		scalar dBdc13_x = -2.0 * (sC1.data()[0] + sC2.data()[0]);
		scalar dBdc13_y = -2.0 * (sC1.data()[1] + sC2.data()[1]);
		scalar dBdc13_z = -2.0 * (sC1.data()[2] + sC2.data()[2]);

		scalar dCdc12_x = 2.0 * sC2.data()[0];
		scalar dCdc12_y = 2.0 * sC2.data()[1];
		scalar dCdc12_z = 2.0 * sC2.data()[2];
		scalar dCdc13_x = -2.0 * sC2.data()[0];
		scalar dCdc13_y = -2.0 * sC2.data()[1];
		scalar dCdc13_z = -2.0 * sC2.data()[2];

		scalar dDdc11_x = 2.0 * sC3.data()[0];
		scalar dDdc11_y = 2.0 * sC3.data()[1];
		scalar dDdc11_z = 2.0 * sC3.data()[2];
		scalar dDdc13_x = 2.0 * (sC1.data()[0] - sC3.data()[0]);
		scalar dDdc13_y = 2.0 * (sC1.data()[1] - sC3.data()[1]);
		scalar dDdc13_z = 2.0 * (sC1.data()[2] - sC3.data()[2]);
		scalar dDdcp_x = -2.0 * sC1.data()[0];
		scalar dDdcp_y = -2.0 * sC1.data()[1];
		scalar dDdcp_z = -2.0 * sC1.data()[2];

		scalar dEdc12_x = 2.0 * sC3.data()[0];
		scalar dEdc12_y = 2.0 * sC3.data()[1];
		scalar dEdc12_z = 2.0 * sC3.data()[2];
		scalar dEdc13_x = 2.0 * (sC2.data()[0] - sC3.data()[0]);
		scalar dEdc13_y = 2.0 * (sC2.data()[1] - sC3.data()[1]);
		scalar dEdc13_z = 2.0 * (sC2.data()[2] - sC3.data()[2]);
		scalar dEdcp_x = -2.0 * sC2.data()[0];
		scalar dEdcp_y = -2.0 * sC2.data()[1];
		scalar dEdcp_z = -2.0 * sC2.data()[2];

		scalar dFdc13_x = 2.0 * sC3.data()[0];
		scalar dFdc13_y = 2.0 * sC3.data()[1];
		scalar dFdc13_z = 2.0 * sC3.data()[2];
		scalar dFdcp_x = -2.0 * sC3.data()[0];
		scalar dFdcp_y = -2.0 * sC3.data()[1];
		scalar dFdcp_z = -2.0 * sC3.data()[2];

		//
		scalar q1 = 0.5 / (A + C - B);
		scalar q2 = alpha / (A + C - B);

		scalar dALdc11_x = q1 * (-(dBdc11_x + dDdc11_x)) - q2 * (dAdc11_x - dBdc11_x);
		scalar dALdc11_y = q1 * (-(dBdc11_y + dDdc11_y)) - q2 * (dAdc11_y - dBdc11_y);
		scalar dALdc11_z = q1 * (-(dBdc11_z + dDdc11_z)) - q2 * (dAdc11_z - dBdc11_z);

		scalar dALdc12_x = q1 * ((2.0 * dCdc12_x + dEdc12_x) - (dBdc12_x)) - q2 * (dCdc12_x - dBdc12_x);
		scalar dALdc12_y = q1 * ((2.0 * dCdc12_y + dEdc12_y) - (dBdc12_y)) - q2 * (dCdc12_y - dBdc12_y);
		scalar dALdc12_z = q1 * ((2.0 * dCdc12_z + dEdc12_z) - (dBdc12_z)) - q2 * (dCdc12_z - dBdc12_z);

		scalar dALdc13_x = 0;
		scalar dALdc13_y = 0;
		scalar dALdc13_z = 0;

		scalar dALdcp_x = q1 * (dEdcp_x - dDdcp_x);
		scalar dALdcp_y = q1 * (dEdcp_y - dDdcp_y);
		scalar dALdcp_z = q1 * (dEdcp_z - dDdcp_z);

		scalar dBTdc11_x = -dALdc11_x;
		scalar dBTdc11_y = -dALdc11_y;
		scalar dBTdc11_z = -dALdc11_z;

		scalar dBTdc12_x = -dALdc12_x;
		scalar dBTdc12_y = -dALdc12_y;
		scalar dBTdc12_z = -dALdc12_z;

		scalar dBTdc13_x = 0;
		scalar dBTdc13_y = 0;
		scalar dBTdc13_z = 0;

		scalar dBTdcp_x = -dALdcp_x;
		scalar dBTdcp_y = -dALdcp_y;
		scalar dBTdcp_z = -dALdcp_z;

		scalar C1x = sC1.data()[0];
		scalar C1y = sC1.data()[1];
		scalar C1z = sC1.data()[2];
		scalar C2x = sC2.data()[0];
		scalar C2y = sC2.data()[1];
		scalar C2z = sC2.data()[2];
		scalar C3x = sC3.data()[0];
		scalar C3y = sC3.data()[1];
		scalar C3z = sC3.data()[2];

		scalar vx = 2.0 * (alpha * C1x + beta * C2x + C3x);
		scalar vy = 2.0 * (alpha * C1y + beta * C2y + C3y);
		scalar vz = 2.0 * (alpha * C1z + beta * C2z + C3z);

		scalar dvxdc11_x = 2.0 * (C1x * dALdc11_x + C2x * dBTdc11_x + alpha);
		scalar dvxdc11_y = 2.0 * (C1x * dALdc11_y + C2x * dBTdc11_y);
		scalar dvxdc11_z = 2.0 * (C1x * dALdc11_z + C2x * dBTdc11_z);

		scalar dvxdc12_x = 2.0 * (C1x * dALdc12_x + C2x * dBTdc12_x + beta);
		scalar dvxdc12_y = 2.0 * (C1x * dALdc12_y + C2x * dBTdc12_y);
		scalar dvxdc12_z = 2.0 * (C1x * dALdc12_z + C2x * dBTdc12_z);

		scalar dvxdc13_x = 0;
		scalar dvxdc13_y = 0;
		scalar dvxdc13_z = 0;

		scalar dvxdcp_x = 2.0 * (C1x * dALdcp_x + C2x * dBTdcp_x - 1);
		scalar dvxdcp_y = 2.0 * (C1x * dALdcp_y + C2x * dBTdcp_y);
		scalar dvxdcp_z = 2.0 * (C1x * dALdcp_z + C2x * dBTdcp_z);
		//
		scalar dvydc11_x = 2.0 * (C1y * dALdc11_x + C2y * dBTdc11_x);
		scalar dvydc11_y = 2.0 * (C1y * dALdc11_y + C2y * dBTdc11_y + alpha);
		scalar dvydc11_z = 2.0 * (C1y * dALdc11_z + C2y * dBTdc11_z);

		scalar dvydc12_x = 2.0 * (C1y * dALdc12_x + C2y * dBTdc12_x);
		scalar dvydc12_y = 2.0 * (C1y * dALdc12_y + C2y * dBTdc12_y + beta);
		scalar dvydc12_z = 2.0 * (C1y * dALdc12_z + C2y * dBTdc12_z);

		scalar dvydc13_x = 0;
		scalar dvydc13_y = 0;
		scalar dvydc13_z = 0;

		scalar dvydcp_x = 2.0 * (C1y * dALdcp_x + C2y * dBTdcp_x);
		scalar dvydcp_y = 2.0 * (C1y * dALdcp_y + C2y * dBTdcp_y - 1);
		scalar dvydcp_z = 2.0 * (C1y * dALdcp_z + C2y * dBTdcp_z);
		//
		scalar dvzdc11_x = 2.0 * (C1z * dALdc11_x + C2z * dBTdc11_x);
		scalar dvzdc11_y = 2.0 * (C1z * dALdc11_y + C2z * dBTdc11_y);
		scalar dvzdc11_z = 2.0 * (C1z * dALdc11_z + C2z * dBTdc11_z + alpha);

		scalar dvzdc12_x = 2.0 * (C1z * dALdc12_x + C2z * dBTdc12_x);
		scalar dvzdc12_y = 2.0 * (C1z * dALdc12_y + C2z * dBTdc12_y);
		scalar dvzdc12_z = 2.0 * (C1z * dALdc12_z + C2z * dBTdc12_z + beta);

		scalar dvzdc13_x = 0;
		scalar dvzdc13_y = 0;
		scalar dvzdc13_z = 0;

		scalar dvzdcp_x = 2.0 * (C1z * dALdcp_x + C2z * dBTdcp_x);
		scalar dvzdcp_y = 2.0 * (C1z * dALdcp_y + C2z * dBTdcp_y);
		scalar dvzdcp_z = 2.0 * (C1z * dALdcp_z + C2z * dBTdcp_z - 1);

		hessian.setZero();
		// f / c11x
		hessian.data()[0] = Check_double_ZERO(vx * dALdc11_x + alpha * dvxdc11_x);
		hessian.data()[1] = Check_double_ZERO(vy * dALdc11_x + alpha * dvydc11_x);
		hessian.data()[2] = Check_double_ZERO(vz * dALdc11_x + alpha * dvzdc11_x);

		hessian.data()[3] = Check_double_ZERO(vx * dBTdc11_x + beta * dvxdc11_x);
		hessian.data()[4] = Check_double_ZERO(vy * dBTdc11_x + beta * dvydc11_x);
		hessian.data()[5] = Check_double_ZERO(vz * dBTdc11_x + beta * dvzdc11_x);

		hessian.data()[6] = Check_double_ZERO(vx * (-dALdc11_x - dBTdc11_x) + (1.0 - alpha - beta) * dvxdc11_x);
		hessian.data()[7] = Check_double_ZERO(vy * (-dALdc11_x - dBTdc11_x) + (1.0 - alpha - beta) * dvydc11_x);
		hessian.data()[8] = Check_double_ZERO(vz * (-dALdc11_x - dBTdc11_x) + (1.0 - alpha - beta) * dvzdc11_x);

		hessian.data()[9] = Check_double_ZERO(-dvxdc11_x);
		hessian.data()[10] = Check_double_ZERO(-dvydc11_x);
		hessian.data()[11] = Check_double_ZERO(-dvzdc11_x);
		// f / c11y
		hessian.data()[12] = Check_double_ZERO(vx * dALdc11_y + alpha * dvxdc11_y);
		hessian.data()[13] = Check_double_ZERO(vy * dALdc11_y + alpha * dvydc11_y);
		hessian.data()[14] = Check_double_ZERO(vz * dALdc11_y + alpha * dvzdc11_y);

		hessian.data()[15] = Check_double_ZERO(vx * dBTdc11_y + beta * dvxdc11_y);
		hessian.data()[16] = Check_double_ZERO(vy * dBTdc11_y + beta * dvydc11_y);
		hessian.data()[17] = Check_double_ZERO(vz * dBTdc11_y + beta * dvzdc11_y);

		hessian.data()[18] = Check_double_ZERO(vx * (-dALdc11_y - dBTdc11_y) + (1.0 - alpha - beta) * dvxdc11_y);
		hessian.data()[19] = Check_double_ZERO(vy * (-dALdc11_y - dBTdc11_y) + (1.0 - alpha - beta) * dvydc11_y);
		hessian.data()[20] = Check_double_ZERO(vz * (-dALdc11_y - dBTdc11_y) + (1.0 - alpha - beta) * dvzdc11_y);

		hessian.data()[21] = Check_double_ZERO(-dvxdc11_y);
		hessian.data()[22] = Check_double_ZERO(-dvydc11_y);
		hessian.data()[23] = Check_double_ZERO(-dvzdc11_y);
		// f / c11z
		hessian.data()[24] = Check_double_ZERO(vx * dALdc11_z + alpha * dvxdc11_z);
		hessian.data()[25] = Check_double_ZERO(vy * dALdc11_z + alpha * dvydc11_z);
		hessian.data()[26] = Check_double_ZERO(vz * dALdc11_z + alpha * dvzdc11_z);

		hessian.data()[27] = Check_double_ZERO(vx * dBTdc11_z + beta * dvxdc11_z);
		hessian.data()[28] = Check_double_ZERO(vy * dBTdc11_z + beta * dvydc11_z);
		hessian.data()[29] = Check_double_ZERO(vz * dBTdc11_z + beta * dvzdc11_z);

		hessian.data()[30] = Check_double_ZERO(vx * (-dALdc11_z - dBTdc11_z) + (1.0 - alpha - beta) * dvxdc11_z);
		hessian.data()[31] = Check_double_ZERO(vy * (-dALdc11_z - dBTdc11_z) + (1.0 - alpha - beta) * dvydc11_z);
		hessian.data()[32] = Check_double_ZERO(vz * (-dALdc11_z - dBTdc11_z) + (1.0 - alpha - beta) * dvzdc11_z);

		hessian.data()[33] = Check_double_ZERO(-dvxdc11_z);
		hessian.data()[34] = Check_double_ZERO(-dvydc11_z);
		hessian.data()[35] = Check_double_ZERO(-dvzdc11_z);
		// f / c12x
		hessian.data()[36] = Check_double_ZERO(vx * dALdc12_x + alpha * dvxdc12_x);
		hessian.data()[37] = Check_double_ZERO(vy * dALdc12_x + alpha * dvydc12_x);
		hessian.data()[38] = Check_double_ZERO(vz * dALdc12_x + alpha * dvzdc12_x);

		hessian.data()[39] = Check_double_ZERO(vx * dBTdc12_x + beta * dvxdc12_x);
		hessian.data()[40] = Check_double_ZERO(vy * dBTdc12_x + beta * dvydc12_x);
		hessian.data()[41] = Check_double_ZERO(vz * dBTdc12_x + beta * dvzdc12_x);

		hessian.data()[42] = Check_double_ZERO(vx * (-dALdc12_x - dBTdc12_x) + (1.0 - alpha - beta) * dvxdc12_x);
		hessian.data()[43] = Check_double_ZERO(vy * (-dALdc12_x - dBTdc12_x) + (1.0 - alpha - beta) * dvydc12_x);
		hessian.data()[44] = Check_double_ZERO(vz * (-dALdc12_x - dBTdc12_x) + (1.0 - alpha - beta) * dvzdc12_x);

		hessian.data()[45] = Check_double_ZERO(-dvxdc12_x);
		hessian.data()[46] = Check_double_ZERO(-dvydc12_x);
		hessian.data()[47] = Check_double_ZERO(-dvzdc12_x);
		// f / c12y
		hessian.data()[48] = Check_double_ZERO(vx * dALdc12_y + alpha * dvxdc12_y);
		hessian.data()[49] = Check_double_ZERO(vy * dALdc12_y + alpha * dvydc12_y);
		hessian.data()[50] = Check_double_ZERO(vz * dALdc12_y + alpha * dvzdc12_y);

		hessian.data()[51] = Check_double_ZERO(vx * dBTdc12_y + beta * dvxdc12_y);
		hessian.data()[52] = Check_double_ZERO(vy * dBTdc12_y + beta * dvydc12_y);
		hessian.data()[53] = Check_double_ZERO(vz * dBTdc12_y + beta * dvzdc12_y);

		hessian.data()[54] = Check_double_ZERO(vx * (-dALdc12_y - dBTdc12_y) + (1.0 - alpha - beta) * dvxdc12_y);
		hessian.data()[55] = Check_double_ZERO(vy * (-dALdc12_y - dBTdc12_y) + (1.0 - alpha - beta) * dvydc12_y);
		hessian.data()[56] = Check_double_ZERO(vz * (-dALdc12_y - dBTdc12_y) + (1.0 - alpha - beta) * dvzdc12_y);

		hessian.data()[57] = Check_double_ZERO(-dvxdc12_y);
		hessian.data()[58] = Check_double_ZERO(-dvydc12_y);
		hessian.data()[59] = Check_double_ZERO(-dvzdc12_y);
		// f / c12z
		hessian.data()[60] = Check_double_ZERO(vx * dALdc12_z + alpha * dvxdc12_z);
		hessian.data()[61] = Check_double_ZERO(vy * dALdc12_z + alpha * dvydc12_z);
		hessian.data()[62] = Check_double_ZERO(vz * dALdc12_z + alpha * dvzdc12_z);

		hessian.data()[63] = Check_double_ZERO(vx * dBTdc12_z + beta * dvxdc12_z);
		hessian.data()[64] = Check_double_ZERO(vy * dBTdc12_z + beta * dvydc12_z);
		hessian.data()[65] = Check_double_ZERO(vz * dBTdc12_z + beta * dvzdc12_z);

		hessian.data()[66] = Check_double_ZERO(vx * (-dALdc12_z - dBTdc12_z) + (1.0 - alpha - beta) * dvxdc12_z);
		hessian.data()[67] = Check_double_ZERO(vy * (-dALdc12_z - dBTdc12_z) + (1.0 - alpha - beta) * dvydc12_z);
		hessian.data()[68] = Check_double_ZERO(vz * (-dALdc12_z - dBTdc12_z) + (1.0 - alpha - beta) * dvzdc12_z);

		hessian.data()[69] = Check_double_ZERO(-dvxdc12_z);
		hessian.data()[70] = Check_double_ZERO(-dvydc12_z);
		hessian.data()[71] = Check_double_ZERO(-dvzdc12_z);
		//	f / c13x
		hessian.data()[72] = Check_double_ZERO(vx * dALdc13_x + alpha * dvxdc13_x);
		hessian.data()[73] = Check_double_ZERO(vy * dALdc13_x + alpha * dvydc13_x);
		hessian.data()[74] = Check_double_ZERO(vz * dALdc13_x + alpha * dvzdc13_x);

		hessian.data()[75] = Check_double_ZERO(vx * dBTdc13_x + beta * dvxdc13_x);
		hessian.data()[76] = Check_double_ZERO(vy * dBTdc13_x + beta * dvydc13_x);
		hessian.data()[77] = Check_double_ZERO(vz * dBTdc13_x + beta * dvzdc13_x);

		hessian.data()[78] = Check_double_ZERO(vx * (-dALdc13_x - dBTdc13_x) + (1.0 - alpha - beta) * dvxdc13_x);
		hessian.data()[79] = Check_double_ZERO(vy * (-dALdc13_x - dBTdc13_x) + (1.0 - alpha - beta) * dvydc13_x);
		hessian.data()[80] = Check_double_ZERO(vz * (-dALdc13_x - dBTdc13_x) + (1.0 - alpha - beta) * dvzdc13_x);

		hessian.data()[81] = Check_double_ZERO(-dvxdc13_x);
		hessian.data()[82] = Check_double_ZERO(-dvydc13_x);
		hessian.data()[83] = Check_double_ZERO(-dvzdc13_x);
		//	f / c13y
		hessian.data()[84] = Check_double_ZERO(vx * dALdc13_y + alpha * dvxdc13_y);
		hessian.data()[85] = Check_double_ZERO(vy * dALdc13_y + alpha * dvydc13_y);
		hessian.data()[86] = Check_double_ZERO(vz * dALdc13_y + alpha * dvzdc13_y);

		hessian.data()[87] = Check_double_ZERO(vx * dBTdc13_y + beta * dvxdc13_y);
		hessian.data()[88] = Check_double_ZERO(vy * dBTdc13_y + beta * dvydc13_y);
		hessian.data()[89] = Check_double_ZERO(vz * dBTdc13_y + beta * dvzdc13_y);

		hessian.data()[90] = Check_double_ZERO(vx * (-dALdc13_y - dBTdc13_y) + (1.0 - alpha - beta) * dvxdc13_y);
		hessian.data()[91] = Check_double_ZERO(vy * (-dALdc13_y - dBTdc13_y) + (1.0 - alpha - beta) * dvydc13_y);
		hessian.data()[92] = Check_double_ZERO(vz * (-dALdc13_y - dBTdc13_y) + (1.0 - alpha - beta) * dvzdc13_y);

		hessian.data()[93] = Check_double_ZERO(-dvxdc13_y);
		hessian.data()[94] = Check_double_ZERO(-dvydc13_y);
		hessian.data()[95] = Check_double_ZERO(-dvzdc13_y);
		//	f / c13z
		hessian.data()[96] = Check_double_ZERO(vx * dALdc13_z + alpha * dvxdc13_z);
		hessian.data()[97] = Check_double_ZERO(vy * dALdc13_z + alpha * dvydc13_z);
		hessian.data()[98] = Check_double_ZERO(vz * dALdc13_z + alpha * dvzdc13_z);

		hessian.data()[99] = Check_double_ZERO(vx * dBTdc13_z + beta * dvxdc13_z);
		hessian.data()[100] = Check_double_ZERO(vy * dBTdc13_z + beta * dvydc13_z);
		hessian.data()[101] = Check_double_ZERO(vz * dBTdc13_z + beta * dvzdc13_z);

		hessian.data()[102] = Check_double_ZERO(vx * (-dALdc13_z - dBTdc13_z) + (1.0 - alpha - beta) * dvxdc13_z);
		hessian.data()[103] = Check_double_ZERO(vy * (-dALdc13_z - dBTdc13_z) + (1.0 - alpha - beta) * dvydc13_z);
		hessian.data()[104] = Check_double_ZERO(vz * (-dALdc13_z - dBTdc13_z) + (1.0 - alpha - beta) * dvzdc13_z);

		hessian.data()[105] = Check_double_ZERO(-dvxdc13_z);
		hessian.data()[106] = Check_double_ZERO(-dvydc13_z);
		hessian.data()[107] = Check_double_ZERO(-dvzdc13_z);
		//	f / cpx
		hessian.data()[108] = Check_double_ZERO(vx * dALdcp_x + alpha * dvxdcp_x);
		hessian.data()[109] = Check_double_ZERO(vy * dALdcp_x + alpha * dvydcp_x);
		hessian.data()[110] = Check_double_ZERO(vz * dALdcp_x + alpha * dvzdcp_x);

		hessian.data()[111] = Check_double_ZERO(vx * dBTdcp_x + beta * dvxdcp_x);
		hessian.data()[112] = Check_double_ZERO(vy * dBTdcp_x + beta * dvydcp_x);
		hessian.data()[113] = Check_double_ZERO(vz * dBTdcp_x + beta * dvzdcp_x);

		hessian.data()[114] = Check_double_ZERO(vx * (-dALdcp_x - dBTdcp_x) + (1.0 - alpha - beta) * dvxdcp_x);
		hessian.data()[115] = Check_double_ZERO(vy * (-dALdcp_x - dBTdcp_x) + (1.0 - alpha - beta) * dvydcp_x);
		hessian.data()[116] = Check_double_ZERO(vz * (-dALdcp_x - dBTdcp_x) + (1.0 - alpha - beta) * dvzdcp_x);

		hessian.data()[117] = Check_double_ZERO(-dvxdcp_x);
		hessian.data()[118] = Check_double_ZERO(-dvydcp_x);
		hessian.data()[119] = Check_double_ZERO(-dvzdcp_x);
		//	f / cpy
		hessian.data()[120] = Check_double_ZERO(vx * dALdcp_y + alpha * dvxdcp_y);
		hessian.data()[121] = Check_double_ZERO(vy * dALdcp_y + alpha * dvydcp_y);
		hessian.data()[122] = Check_double_ZERO(vz * dALdcp_y + alpha * dvzdcp_y);

		hessian.data()[123] = Check_double_ZERO(vx * dBTdcp_y + beta * dvxdcp_y);
		hessian.data()[124] = Check_double_ZERO(vy * dBTdcp_y + beta * dvydcp_y);
		hessian.data()[125] = Check_double_ZERO(vz * dBTdcp_y + beta * dvzdcp_y);

		hessian.data()[126] = Check_double_ZERO(vx * (-dALdcp_y - dBTdcp_y) + (1.0 - alpha - beta) * dvxdcp_y);
		hessian.data()[127] = Check_double_ZERO(vy * (-dALdcp_y - dBTdcp_y) + (1.0 - alpha - beta) * dvydcp_y);
		hessian.data()[128] = Check_double_ZERO(vz * (-dALdcp_y - dBTdcp_y) + (1.0 - alpha - beta) * dvzdcp_y);

		hessian.data()[129] = Check_double_ZERO(-dvxdcp_y);
		hessian.data()[130] = Check_double_ZERO(-dvydcp_y);
		hessian.data()[131] = Check_double_ZERO(-dvzdcp_y);
		//	f / cpz
		hessian.data()[132] = Check_double_ZERO(vx * dALdcp_z + alpha * dvxdcp_z);
		hessian.data()[133] = Check_double_ZERO(vy * dALdcp_z + alpha * dvydcp_z);
		hessian.data()[134] = Check_double_ZERO(vz * dALdcp_z + alpha * dvzdcp_z);

		hessian.data()[135] = Check_double_ZERO(vx * dBTdcp_z + beta * dvxdcp_z);
		hessian.data()[136] = Check_double_ZERO(vy * dBTdcp_z + beta * dvydcp_z);
		hessian.data()[137] = Check_double_ZERO(vz * dBTdcp_z + beta * dvzdcp_z);

		hessian.data()[138] = Check_double_ZERO(vx * (-dALdcp_z - dBTdcp_z) + (1.0 - alpha - beta) * dvxdcp_z);
		hessian.data()[139] = Check_double_ZERO(vy * (-dALdcp_z - dBTdcp_z) + (1.0 - alpha - beta) * dvydcp_z);
		hessian.data()[140] = Check_double_ZERO(vz * (-dALdcp_z - dBTdcp_z) + (1.0 - alpha - beta) * dvzdcp_z);

		hessian.data()[141] = Check_double_ZERO(-dvxdcp_z);
		hessian.data()[142] = Check_double_ZERO(-dvydcp_z);
		hessian.data()[143] = Check_double_ZERO(-dvzdcp_z);
	}

	void MipcSlabSphereConstraint::alphaBetaHessian(mat12 &hessian)
	{
		scalar dAdc11_x = 2.0 * sC1.data()[0];
		scalar dAdc11_y = 2.0 * sC1.data()[1];
		scalar dAdc11_z = 2.0 * sC1.data()[2];
		scalar dAdc13_x = -2.0 * sC1.data()[0];
		scalar dAdc13_y = -2.0 * sC1.data()[1];
		scalar dAdc13_z = -2.0 * sC1.data()[2];

		scalar dBdc11_x = 2.0 * sC2.data()[0];
		scalar dBdc11_y = 2.0 * sC2.data()[1];
		scalar dBdc11_z = 2.0 * sC2.data()[2];
		scalar dBdc12_x = 2.0 * sC1.data()[0];
		scalar dBdc12_y = 2.0 * sC1.data()[1];
		scalar dBdc12_z = 2.0 * sC1.data()[2];
		scalar dBdc13_x = -2.0 * (sC1.data()[0] + sC2.data()[0]);
		scalar dBdc13_y = -2.0 * (sC1.data()[1] + sC2.data()[1]);
		scalar dBdc13_z = -2.0 * (sC1.data()[2] + sC2.data()[2]);

		scalar dCdc12_x = 2.0 * sC2.data()[0];
		scalar dCdc12_y = 2.0 * sC2.data()[1];
		scalar dCdc12_z = 2.0 * sC2.data()[2];
		scalar dCdc13_x = -2.0 * sC2.data()[0];
		scalar dCdc13_y = -2.0 * sC2.data()[1];
		scalar dCdc13_z = -2.0 * sC2.data()[2];

		scalar dDdc11_x = 2.0 * sC3.data()[0];
		scalar dDdc11_y = 2.0 * sC3.data()[1];
		scalar dDdc11_z = 2.0 * sC3.data()[2];
		scalar dDdc13_x = 2.0 * (sC1.data()[0] - sC3.data()[0]);
		scalar dDdc13_y = 2.0 * (sC1.data()[1] - sC3.data()[1]);
		scalar dDdc13_z = 2.0 * (sC1.data()[2] - sC3.data()[2]);
		scalar dDdcp_x = -2.0 * sC1.data()[0];
		scalar dDdcp_y = -2.0 * sC1.data()[1];
		scalar dDdcp_z = -2.0 * sC1.data()[2];

		scalar dEdc12_x = 2.0 * sC3.data()[0];
		scalar dEdc12_y = 2.0 * sC3.data()[1];
		scalar dEdc12_z = 2.0 * sC3.data()[2];
		scalar dEdc13_x = 2.0 * (sC2.data()[0] - sC3.data()[0]);
		scalar dEdc13_y = 2.0 * (sC2.data()[1] - sC3.data()[1]);
		scalar dEdc13_z = 2.0 * (sC2.data()[2] - sC3.data()[2]);
		scalar dEdcp_x = -2.0 * sC2.data()[0];
		scalar dEdcp_y = -2.0 * sC2.data()[1];
		scalar dEdcp_z = -2.0 * sC2.data()[2];

		scalar dFdc13_x = 2.0 * sC3.data()[0];
		scalar dFdc13_y = 2.0 * sC3.data()[1];
		scalar dFdc13_z = 2.0 * sC3.data()[2];
		scalar dFdcp_x = -2.0 * sC3.data()[0];
		scalar dFdcp_y = -2.0 * sC3.data()[1];
		scalar dFdcp_z = -2.0 * sC3.data()[2];

		//
		scalar q1 = 1.0 / delta;
		scalar q2 = alpha / delta;
		scalar q3 = beta / delta;

		scalar dALdc11_x = q1 * (E * dBdc11_x - 2.0 * C * dDdc11_x) - q2 * (4.0 * C * dAdc11_x - 2.0 * B * dBdc11_x);
		scalar dALdc11_y = q1 * (E * dBdc11_y - 2.0 * C * dDdc11_y) - q2 * (4.0 * C * dAdc11_y - 2.0 * B * dBdc11_y);
		scalar dALdc11_z = q1 * (E * dBdc11_z - 2.0 * C * dDdc11_z) - q2 * (4.0 * C * dAdc11_z - 2.0 * B * dBdc11_z);

		scalar dALdc12_x = q1 * (E * dBdc12_x + B * dEdc12_x - 2.0 * D * dCdc12_x) - q2 * (4.0 * A * dCdc12_x - 2.0 * B * dBdc12_x);
		scalar dALdc12_y = q1 * (E * dBdc12_y + B * dEdc12_y - 2.0 * D * dCdc12_y) - q2 * (4.0 * A * dCdc12_y - 2.0 * B * dBdc12_y);
		scalar dALdc12_z = q1 * (E * dBdc12_z + B * dEdc12_z - 2.0 * D * dCdc12_z) - q2 * (4.0 * A * dCdc12_z - 2.0 * B * dBdc12_z);

		scalar dALdc13_x = q1 * (E * dBdc13_x + B * dEdc13_x - 2.0 * (C * dDdc13_x + D * dCdc13_x)) - q2 * (4.0 * (A * dCdc13_x + C * dAdc13_x) - 2.0 * B * dBdc13_x);
		scalar dALdc13_y = q1 * (E * dBdc13_y + B * dEdc13_y - 2.0 * (C * dDdc13_y + D * dCdc13_y)) - q2 * (4.0 * (A * dCdc13_y + C * dAdc13_y) - 2.0 * B * dBdc13_y);
		scalar dALdc13_z = q1 * (E * dBdc13_z + B * dEdc13_z - 2.0 * (C * dDdc13_z + D * dCdc13_z)) - q2 * (4.0 * (A * dCdc13_z + C * dAdc13_z) - 2.0 * B * dBdc13_z);

		scalar dALdcp_x = q1 * (B * dEdcp_x - 2.0 * C * dDdcp_x);
		scalar dALdcp_y = q1 * (B * dEdcp_y - 2.0 * C * dDdcp_y);
		scalar dALdcp_z = q1 * (B * dEdcp_z - 2.0 * C * dDdcp_z);

		scalar dBTdc11_x = q1 * (D * dBdc11_x + B * dDdc11_x - 2.0 * E * dAdc11_x) - q3 * (4.0 * C * dAdc11_x - 2.0 * B * dBdc11_x);
		scalar dBTdc11_y = q1 * (D * dBdc11_y + B * dDdc11_y - 2.0 * E * dAdc11_y) - q3 * (4.0 * C * dAdc11_y - 2.0 * B * dBdc11_y);
		scalar dBTdc11_z = q1 * (D * dBdc11_z + B * dDdc11_z - 2.0 * E * dAdc11_z) - q3 * (4.0 * C * dAdc11_z - 2.0 * B * dBdc11_z);

		scalar dBTdc12_x = q1 * (D * dBdc12_x - 2.0 * A * dEdc12_x) - q3 * (4.0 * A * dCdc12_x - 2.0 * B * dBdc12_x);
		scalar dBTdc12_y = q1 * (D * dBdc12_y - 2.0 * A * dEdc12_y) - q3 * (4.0 * A * dCdc12_y - 2.0 * B * dBdc12_y);
		scalar dBTdc12_z = q1 * (D * dBdc12_z - 2.0 * A * dEdc12_z) - q3 * (4.0 * A * dCdc12_z - 2.0 * B * dBdc12_z);

		scalar dBTdc13_x = q1 * (D * dBdc13_x + B * dDdc13_x - 2.0 * (A * dEdc13_x + E * dAdc13_x)) - q3 * (4.0 * (A * dCdc13_x + C * dAdc13_x) - 2.0 * B * dBdc13_x);
		scalar dBTdc13_y = q1 * (D * dBdc13_y + B * dDdc13_y - 2.0 * (A * dEdc13_y + E * dAdc13_y)) - q3 * (4.0 * (A * dCdc13_y + C * dAdc13_y) - 2.0 * B * dBdc13_y);
		scalar dBTdc13_z = q1 * (D * dBdc13_z + B * dDdc13_z - 2.0 * (A * dEdc13_z + E * dAdc13_z)) - q3 * (4.0 * (A * dCdc13_z + C * dAdc13_z) - 2.0 * B * dBdc13_z);

		scalar dBTdcp_x = q1 * (B * dDdcp_x - 2.0 * A * dEdcp_x);
		scalar dBTdcp_y = q1 * (B * dDdcp_y - 2.0 * A * dEdcp_y);
		scalar dBTdcp_z = q1 * (B * dDdcp_z - 2.0 * A * dEdcp_z);

		scalar C1x = sC1.data()[0];
		scalar C1y = sC1.data()[1];
		scalar C1z = sC1.data()[2];
		scalar C2x = sC2.data()[0];
		scalar C2y = sC2.data()[1];
		scalar C2z = sC2.data()[2];
		scalar C3x = sC3.data()[0];
		scalar C3y = sC3.data()[1];
		scalar C3z = sC3.data()[2];

		scalar vx = 2.0 * (alpha * C1x + beta * C2x + C3x);
		scalar vy = 2.0 * (alpha * C1y + beta * C2y + C3y);
		scalar vz = 2.0 * (alpha * C1z + beta * C2z + C3z);

		scalar dvxdc11_x = 2.0 * (C1x * dALdc11_x + C2x * dBTdc11_x + alpha);
		scalar dvxdc11_y = 2.0 * (C1x * dALdc11_y + C2x * dBTdc11_y);
		scalar dvxdc11_z = 2.0 * (C1x * dALdc11_z + C2x * dBTdc11_z);

		scalar dvxdc12_x = 2.0 * (C1x * dALdc12_x + C2x * dBTdc12_x + beta);
		scalar dvxdc12_y = 2.0 * (C1x * dALdc12_y + C2x * dBTdc12_y);
		scalar dvxdc12_z = 2.0 * (C1x * dALdc12_z + C2x * dBTdc12_z);

		scalar dvxdc13_x = 2.0 * (C1x * dALdc13_x + C2x * dBTdc13_x + 1 - alpha - beta);
		scalar dvxdc13_y = 2.0 * (C1x * dALdc13_y + C2x * dBTdc13_y);
		scalar dvxdc13_z = 2.0 * (C1x * dALdc13_z + C2x * dBTdc13_z);

		scalar dvxdcp_x = 2.0 * (C1x * dALdcp_x + C2x * dBTdcp_x - 1);
		scalar dvxdcp_y = 2.0 * (C1x * dALdcp_y + C2x * dBTdcp_y);
		scalar dvxdcp_z = 2.0 * (C1x * dALdcp_z + C2x * dBTdcp_z);
		//
		scalar dvydc11_x = 2.0 * (C1y * dALdc11_x + C2y * dBTdc11_x);
		scalar dvydc11_y = 2.0 * (C1y * dALdc11_y + C2y * dBTdc11_y + alpha);
		scalar dvydc11_z = 2.0 * (C1y * dALdc11_z + C2y * dBTdc11_z);

		scalar dvydc12_x = 2.0 * (C1y * dALdc12_x + C2y * dBTdc12_x);
		scalar dvydc12_y = 2.0 * (C1y * dALdc12_y + C2y * dBTdc12_y + beta);
		scalar dvydc12_z = 2.0 * (C1y * dALdc12_z + C2y * dBTdc12_z);

		scalar dvydc13_x = 2.0 * (C1y * dALdc13_x + C2y * dBTdc13_x);
		scalar dvydc13_y = 2.0 * (C1y * dALdc13_y + C2y * dBTdc13_y + 1 - alpha - beta);
		scalar dvydc13_z = 2.0 * (C1y * dALdc13_z + C2y * dBTdc13_z);

		scalar dvydcp_x = 2.0 * (C1y * dALdcp_x + C2y * dBTdcp_x);
		scalar dvydcp_y = 2.0 * (C1y * dALdcp_y + C2y * dBTdcp_y - 1);
		scalar dvydcp_z = 2.0 * (C1y * dALdcp_z + C2y * dBTdcp_z);
		//
		scalar dvzdc11_x = 2.0 * (C1z * dALdc11_x + C2z * dBTdc11_x);
		scalar dvzdc11_y = 2.0 * (C1z * dALdc11_y + C2z * dBTdc11_y);
		scalar dvzdc11_z = 2.0 * (C1z * dALdc11_z + C2z * dBTdc11_z + alpha);

		scalar dvzdc12_x = 2.0 * (C1z * dALdc12_x + C2z * dBTdc12_x);
		scalar dvzdc12_y = 2.0 * (C1z * dALdc12_y + C2z * dBTdc12_y);
		scalar dvzdc12_z = 2.0 * (C1z * dALdc12_z + C2z * dBTdc12_z + beta);

		scalar dvzdc13_x = 2.0 * (C1z * dALdc13_x + C2z * dBTdc13_x);
		scalar dvzdc13_y = 2.0 * (C1z * dALdc13_y + C2z * dBTdc13_y);
		scalar dvzdc13_z = 2.0 * (C1z * dALdc13_z + C2z * dBTdc13_z + 1 - alpha - beta);

		scalar dvzdcp_x = 2.0 * (C1z * dALdcp_x + C2z * dBTdcp_x);
		scalar dvzdcp_y = 2.0 * (C1z * dALdcp_y + C2z * dBTdcp_y);
		scalar dvzdcp_z = 2.0 * (C1z * dALdcp_z + C2z * dBTdcp_z - 1);

		hessian.setZero();
		// f / c11x
		hessian.data()[0] = Check_double_ZERO(vx * dALdc11_x + alpha * dvxdc11_x);
		hessian.data()[1] = Check_double_ZERO(vy * dALdc11_x + alpha * dvydc11_x);
		hessian.data()[2] = Check_double_ZERO(vz * dALdc11_x + alpha * dvzdc11_x);

		hessian.data()[3] = Check_double_ZERO(vx * dBTdc11_x + beta * dvxdc11_x);
		hessian.data()[4] = Check_double_ZERO(vy * dBTdc11_x + beta * dvydc11_x);
		hessian.data()[5] = Check_double_ZERO(vz * dBTdc11_x + beta * dvzdc11_x);

		hessian.data()[6] = Check_double_ZERO(vx * (-dALdc11_x - dBTdc11_x) + (1.0 - alpha - beta) * dvxdc11_x);
		hessian.data()[7] = Check_double_ZERO(vy * (-dALdc11_x - dBTdc11_x) + (1.0 - alpha - beta) * dvydc11_x);
		hessian.data()[8] = Check_double_ZERO(vz * (-dALdc11_x - dBTdc11_x) + (1.0 - alpha - beta) * dvzdc11_x);

		hessian.data()[9] = Check_double_ZERO(-dvxdc11_x);
		hessian.data()[10] = Check_double_ZERO(-dvydc11_x);
		hessian.data()[11] = Check_double_ZERO(-dvzdc11_x);
		// f / c11y
		hessian.data()[12] = Check_double_ZERO(vx * dALdc11_y + alpha * dvxdc11_y);
		hessian.data()[13] = Check_double_ZERO(vy * dALdc11_y + alpha * dvydc11_y);
		hessian.data()[14] = Check_double_ZERO(vz * dALdc11_y + alpha * dvzdc11_y);

		hessian.data()[15] = Check_double_ZERO(vx * dBTdc11_y + beta * dvxdc11_y);
		hessian.data()[16] = Check_double_ZERO(vy * dBTdc11_y + beta * dvydc11_y);
		hessian.data()[17] = Check_double_ZERO(vz * dBTdc11_y + beta * dvzdc11_y);

		hessian.data()[18] = Check_double_ZERO(vx * (-dALdc11_y - dBTdc11_y) + (1.0 - alpha - beta) * dvxdc11_y);
		hessian.data()[19] = Check_double_ZERO(vy * (-dALdc11_y - dBTdc11_y) + (1.0 - alpha - beta) * dvydc11_y);
		hessian.data()[20] = Check_double_ZERO(vz * (-dALdc11_y - dBTdc11_y) + (1.0 - alpha - beta) * dvzdc11_y);

		hessian.data()[21] = Check_double_ZERO(-dvxdc11_y);
		hessian.data()[22] = Check_double_ZERO(-dvydc11_y);
		hessian.data()[23] = Check_double_ZERO(-dvzdc11_y);
		// f / c11z
		hessian.data()[24] = Check_double_ZERO(vx * dALdc11_z + alpha * dvxdc11_z);
		hessian.data()[25] = Check_double_ZERO(vy * dALdc11_z + alpha * dvydc11_z);
		hessian.data()[26] = Check_double_ZERO(vz * dALdc11_z + alpha * dvzdc11_z);

		hessian.data()[27] = Check_double_ZERO(vx * dBTdc11_z + beta * dvxdc11_z);
		hessian.data()[28] = Check_double_ZERO(vy * dBTdc11_z + beta * dvydc11_z);
		hessian.data()[29] = Check_double_ZERO(vz * dBTdc11_z + beta * dvzdc11_z);

		hessian.data()[30] = Check_double_ZERO(vx * (-dALdc11_z - dBTdc11_z) + (1.0 - alpha - beta) * dvxdc11_z);
		hessian.data()[31] = Check_double_ZERO(vy * (-dALdc11_z - dBTdc11_z) + (1.0 - alpha - beta) * dvydc11_z);
		hessian.data()[32] = Check_double_ZERO(vz * (-dALdc11_z - dBTdc11_z) + (1.0 - alpha - beta) * dvzdc11_z);

		hessian.data()[33] = Check_double_ZERO(-dvxdc11_z);
		hessian.data()[34] = Check_double_ZERO(-dvydc11_z);
		hessian.data()[35] = Check_double_ZERO(-dvzdc11_z);
		// f / c12x
		hessian.data()[36] = Check_double_ZERO(vx * dALdc12_x + alpha * dvxdc12_x);
		hessian.data()[37] = Check_double_ZERO(vy * dALdc12_x + alpha * dvydc12_x);
		hessian.data()[38] = Check_double_ZERO(vz * dALdc12_x + alpha * dvzdc12_x);

		hessian.data()[39] = Check_double_ZERO(vx * dBTdc12_x + beta * dvxdc12_x);
		hessian.data()[40] = Check_double_ZERO(vy * dBTdc12_x + beta * dvydc12_x);
		hessian.data()[41] = Check_double_ZERO(vz * dBTdc12_x + beta * dvzdc12_x);

		hessian.data()[42] = Check_double_ZERO(vx * (-dALdc12_x - dBTdc12_x) + (1.0 - alpha - beta) * dvxdc12_x);
		hessian.data()[43] = Check_double_ZERO(vy * (-dALdc12_x - dBTdc12_x) + (1.0 - alpha - beta) * dvydc12_x);
		hessian.data()[44] = Check_double_ZERO(vz * (-dALdc12_x - dBTdc12_x) + (1.0 - alpha - beta) * dvzdc12_x);

		hessian.data()[45] = Check_double_ZERO(-dvxdc12_x);
		hessian.data()[46] = Check_double_ZERO(-dvydc12_x);
		hessian.data()[47] = Check_double_ZERO(-dvzdc12_x);
		// f / c12y
		hessian.data()[48] = Check_double_ZERO(vx * dALdc12_y + alpha * dvxdc12_y);
		hessian.data()[49] = Check_double_ZERO(vy * dALdc12_y + alpha * dvydc12_y);
		hessian.data()[50] = Check_double_ZERO(vz * dALdc12_y + alpha * dvzdc12_y);

		hessian.data()[51] = Check_double_ZERO(vx * dBTdc12_y + beta * dvxdc12_y);
		hessian.data()[52] = Check_double_ZERO(vy * dBTdc12_y + beta * dvydc12_y);
		hessian.data()[53] = Check_double_ZERO(vz * dBTdc12_y + beta * dvzdc12_y);

		hessian.data()[54] = Check_double_ZERO(vx * (-dALdc12_y - dBTdc12_y) + (1.0 - alpha - beta) * dvxdc12_y);
		hessian.data()[55] = Check_double_ZERO(vy * (-dALdc12_y - dBTdc12_y) + (1.0 - alpha - beta) * dvydc12_y);
		hessian.data()[56] = Check_double_ZERO(vz * (-dALdc12_y - dBTdc12_y) + (1.0 - alpha - beta) * dvzdc12_y);

		hessian.data()[57] = Check_double_ZERO(-dvxdc12_y);
		hessian.data()[58] = Check_double_ZERO(-dvydc12_y);
		hessian.data()[59] = Check_double_ZERO(-dvzdc12_y);
		// f / c12z
		hessian.data()[60] = Check_double_ZERO(vx * dALdc12_z + alpha * dvxdc12_z);
		hessian.data()[61] = Check_double_ZERO(vy * dALdc12_z + alpha * dvydc12_z);
		hessian.data()[62] = Check_double_ZERO(vz * dALdc12_z + alpha * dvzdc12_z);

		hessian.data()[63] = Check_double_ZERO(vx * dBTdc12_z + beta * dvxdc12_z);
		hessian.data()[64] = Check_double_ZERO(vy * dBTdc12_z + beta * dvydc12_z);
		hessian.data()[65] = Check_double_ZERO(vz * dBTdc12_z + beta * dvzdc12_z);

		hessian.data()[66] = Check_double_ZERO(vx * (-dALdc12_z - dBTdc12_z) + (1.0 - alpha - beta) * dvxdc12_z);
		hessian.data()[67] = Check_double_ZERO(vy * (-dALdc12_z - dBTdc12_z) + (1.0 - alpha - beta) * dvydc12_z);
		hessian.data()[68] = Check_double_ZERO(vz * (-dALdc12_z - dBTdc12_z) + (1.0 - alpha - beta) * dvzdc12_z);

		hessian.data()[69] = Check_double_ZERO(-dvxdc12_z);
		hessian.data()[70] = Check_double_ZERO(-dvydc12_z);
		hessian.data()[71] = Check_double_ZERO(-dvzdc12_z);
		//	f / c13x
		hessian.data()[72] = Check_double_ZERO(vx * dALdc13_x + alpha * dvxdc13_x);
		hessian.data()[73] = Check_double_ZERO(vy * dALdc13_x + alpha * dvydc13_x);
		hessian.data()[74] = Check_double_ZERO(vz * dALdc13_x + alpha * dvzdc13_x);

		hessian.data()[75] = Check_double_ZERO(vx * dBTdc13_x + beta * dvxdc13_x);
		hessian.data()[76] = Check_double_ZERO(vy * dBTdc13_x + beta * dvydc13_x);
		hessian.data()[77] = Check_double_ZERO(vz * dBTdc13_x + beta * dvzdc13_x);

		hessian.data()[78] = Check_double_ZERO(vx * (-dALdc13_x - dBTdc13_x) + (1.0 - alpha - beta) * dvxdc13_x);
		hessian.data()[79] = Check_double_ZERO(vy * (-dALdc13_x - dBTdc13_x) + (1.0 - alpha - beta) * dvydc13_x);
		hessian.data()[80] = Check_double_ZERO(vz * (-dALdc13_x - dBTdc13_x) + (1.0 - alpha - beta) * dvzdc13_x);

		hessian.data()[81] = Check_double_ZERO(-dvxdc13_x);
		hessian.data()[82] = Check_double_ZERO(-dvydc13_x);
		hessian.data()[83] = Check_double_ZERO(-dvzdc13_x);
		//	f / c13y
		hessian.data()[84] = Check_double_ZERO(vx * dALdc13_y + alpha * dvxdc13_y);
		hessian.data()[85] = Check_double_ZERO(vy * dALdc13_y + alpha * dvydc13_y);
		hessian.data()[86] = Check_double_ZERO(vz * dALdc13_y + alpha * dvzdc13_y);

		hessian.data()[87] = Check_double_ZERO(vx * dBTdc13_y + beta * dvxdc13_y);
		hessian.data()[88] = Check_double_ZERO(vy * dBTdc13_y + beta * dvydc13_y);
		hessian.data()[89] = Check_double_ZERO(vz * dBTdc13_y + beta * dvzdc13_y);

		hessian.data()[90] = Check_double_ZERO(vx * (-dALdc13_y - dBTdc13_y) + (1.0 - alpha - beta) * dvxdc13_y);
		hessian.data()[91] = Check_double_ZERO(vy * (-dALdc13_y - dBTdc13_y) + (1.0 - alpha - beta) * dvydc13_y);
		hessian.data()[92] = Check_double_ZERO(vz * (-dALdc13_y - dBTdc13_y) + (1.0 - alpha - beta) * dvzdc13_y);

		hessian.data()[93] = Check_double_ZERO(-dvxdc13_y);
		hessian.data()[94] = Check_double_ZERO(-dvydc13_y);
		hessian.data()[95] = Check_double_ZERO(-dvzdc13_y);
		//	f / c13z
		hessian.data()[96] = Check_double_ZERO(vx * dALdc13_z + alpha * dvxdc13_z);
		hessian.data()[97] = Check_double_ZERO(vy * dALdc13_z + alpha * dvydc13_z);
		hessian.data()[98] = Check_double_ZERO(vz * dALdc13_z + alpha * dvzdc13_z);

		hessian.data()[99] = Check_double_ZERO(vx * dBTdc13_z + beta * dvxdc13_z);
		hessian.data()[100] = Check_double_ZERO(vy * dBTdc13_z + beta * dvydc13_z);
		hessian.data()[101] = Check_double_ZERO(vz * dBTdc13_z + beta * dvzdc13_z);

		hessian.data()[102] = Check_double_ZERO(vx * (-dALdc13_z - dBTdc13_z) + (1.0 - alpha - beta) * dvxdc13_z);
		hessian.data()[103] = Check_double_ZERO(vy * (-dALdc13_z - dBTdc13_z) + (1.0 - alpha - beta) * dvydc13_z);
		hessian.data()[104] = Check_double_ZERO(vz * (-dALdc13_z - dBTdc13_z) + (1.0 - alpha - beta) * dvzdc13_z);

		hessian.data()[105] = Check_double_ZERO(-dvxdc13_z);
		hessian.data()[106] = Check_double_ZERO(-dvydc13_z);
		hessian.data()[107] = Check_double_ZERO(-dvzdc13_z);
		//	f / cpx
		hessian.data()[108] = Check_double_ZERO(vx * dALdcp_x + alpha * dvxdcp_x);
		hessian.data()[109] = Check_double_ZERO(vy * dALdcp_x + alpha * dvydcp_x);
		hessian.data()[110] = Check_double_ZERO(vz * dALdcp_x + alpha * dvzdcp_x);

		hessian.data()[111] = Check_double_ZERO(vx * dBTdcp_x + beta * dvxdcp_x);
		hessian.data()[112] = Check_double_ZERO(vy * dBTdcp_x + beta * dvydcp_x);
		hessian.data()[113] = Check_double_ZERO(vz * dBTdcp_x + beta * dvzdcp_x);

		hessian.data()[114] = Check_double_ZERO(vx * (-dALdcp_x - dBTdcp_x) + (1.0 - alpha - beta) * dvxdcp_x);
		hessian.data()[115] = Check_double_ZERO(vy * (-dALdcp_x - dBTdcp_x) + (1.0 - alpha - beta) * dvydcp_x);
		hessian.data()[116] = Check_double_ZERO(vz * (-dALdcp_x - dBTdcp_x) + (1.0 - alpha - beta) * dvzdcp_x);

		hessian.data()[117] = Check_double_ZERO(-dvxdcp_x);
		hessian.data()[118] = Check_double_ZERO(-dvydcp_x);
		hessian.data()[119] = Check_double_ZERO(-dvzdcp_x);
		//	f / cpy
		hessian.data()[120] = Check_double_ZERO(vx * dALdcp_y + alpha * dvxdcp_y);
		hessian.data()[121] = Check_double_ZERO(vy * dALdcp_y + alpha * dvydcp_y);
		hessian.data()[122] = Check_double_ZERO(vz * dALdcp_y + alpha * dvzdcp_y);

		hessian.data()[123] = Check_double_ZERO(vx * dBTdcp_y + beta * dvxdcp_y);
		hessian.data()[124] = Check_double_ZERO(vy * dBTdcp_y + beta * dvydcp_y);
		hessian.data()[125] = Check_double_ZERO(vz * dBTdcp_y + beta * dvzdcp_y);

		hessian.data()[126] = Check_double_ZERO(vx * (-dALdcp_y - dBTdcp_y) + (1.0 - alpha - beta) * dvxdcp_y);
		hessian.data()[127] = Check_double_ZERO(vy * (-dALdcp_y - dBTdcp_y) + (1.0 - alpha - beta) * dvydcp_y);
		hessian.data()[128] = Check_double_ZERO(vz * (-dALdcp_y - dBTdcp_y) + (1.0 - alpha - beta) * dvzdcp_y);

		hessian.data()[129] = Check_double_ZERO(-dvxdcp_y);
		hessian.data()[130] = Check_double_ZERO(-dvydcp_y);
		hessian.data()[131] = Check_double_ZERO(-dvzdcp_y);
		//	f / cpz
		hessian.data()[132] = Check_double_ZERO(vx * dALdcp_z + alpha * dvxdcp_z);
		hessian.data()[133] = Check_double_ZERO(vy * dALdcp_z + alpha * dvydcp_z);
		hessian.data()[134] = Check_double_ZERO(vz * dALdcp_z + alpha * dvzdcp_z);

		hessian.data()[135] = Check_double_ZERO(vx * dBTdcp_z + beta * dvxdcp_z);
		hessian.data()[136] = Check_double_ZERO(vy * dBTdcp_z + beta * dvydcp_z);
		hessian.data()[137] = Check_double_ZERO(vz * dBTdcp_z + beta * dvzdcp_z);

		hessian.data()[138] = Check_double_ZERO(vx * (-dALdcp_z - dBTdcp_z) + (1.0 - alpha - beta) * dvxdcp_z);
		hessian.data()[139] = Check_double_ZERO(vy * (-dALdcp_z - dBTdcp_z) + (1.0 - alpha - beta) * dvydcp_z);
		hessian.data()[140] = Check_double_ZERO(vz * (-dALdcp_z - dBTdcp_z) + (1.0 - alpha - beta) * dvzdcp_z);

		hessian.data()[141] = Check_double_ZERO(-dvxdcp_z);
		hessian.data()[142] = Check_double_ZERO(-dvydcp_z);
		hessian.data()[143] = Check_double_ZERO(-dvzdcp_z);
	}

}
