#ifndef PBC_ENERGY_H
#define PBC_ENERGY_H

#include "../VecMatDef.h"

void computeStrainMatchingEnergy(double stiffness, const Eigen::Matrix<double,3,1> & epsilon, const Eigen::Matrix<double,4,2> & x, const Eigen::Matrix<double,4,2> & X, double& energy);
void computeStrainMatchingEnergyGradient(double stiffness, const Eigen::Matrix<double,3,1> & epsilon, const Eigen::Matrix<double,4,2> & x, const Eigen::Matrix<double,4,2> & X, Eigen::Matrix<double, 8, 1>& energygradient);
void computeStrainMatchingEnergyHessian(double stiffness, const Eigen::Matrix<double,3,1> & epsilon, const Eigen::Matrix<double,4,2> & x, const Eigen::Matrix<double,4,2> & X, Eigen::Matrix<double, 8, 8>& energyhessian);

void computed2ndPKdmany(const Eigen::Matrix<double,2,1> & xi, const Eigen::Matrix<double,2,1> & xj, const Eigen::Matrix<double,2,1> & xk, const Eigen::Matrix<double,2,1> & xl, const Eigen::Matrix<double,2,1> & Xi, 
	const Eigen::Matrix<double,2,1> & Xj, const Eigen::Matrix<double,2,1> & Xk, const Eigen::Matrix<double,2,1> & Xl, const Eigen::Matrix<double,2,1> & f0_unscaled, const Eigen::Matrix<double,2,1> & f1_unscaled, 
	double sign, double thickness, Eigen::Matrix<double, 3, 4>& dSdF, Eigen::Matrix<double, 4, 8>& dFdx, Eigen::Matrix<double, 3, 4>& dSdf, 
	Eigen::Matrix<double, 4, 8>& dndx, Eigen::Matrix<double, 3, 4>& dSdn);
	
void computedfdELocal(double stiffness, const Eigen::Matrix<double,3,1> & epsilon, const Eigen::Matrix<double,4,2> & x, const Eigen::Matrix<double,4,2> & X, Eigen::Matrix<double, 8, 3>& dfdE);
#endif