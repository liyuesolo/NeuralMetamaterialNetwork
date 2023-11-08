#ifndef FEM_ENERGY_H
#define FEM_ENERGY_H

#include "../VecMatDef.h"

void computeLinear2DNeoHookeanEnergy(double E, double nu, const Eigen::Matrix<double,3,2> & x, const Eigen::Matrix<double,3,2> & X, double& energy);
void computeLinear2DNeoHookeanEnergyGradient(double E, double nu, const Eigen::Matrix<double,3,2> & x, const Eigen::Matrix<double,3,2> & X, Eigen::Matrix<double, 6, 1>& energygradient);
void computeLinear2DNeoHookeanEnergyHessian(double E, double nu, const Eigen::Matrix<double,3,2> & x, const Eigen::Matrix<double,3,2> & X, Eigen::Matrix<double, 6, 6>& energyhessian);
void computeLinear2DNeoHookeandfdX(double E, double nu, const Eigen::Matrix<double,3,2> & x, const Eigen::Matrix<double,3,2> & X, Eigen::Matrix<double, 6, 6>& dfdX);

void computeQuadratic2DNeoHookeanEnergy(double E, double nu, const Eigen::Matrix<double,6,2> & x, const Eigen::Matrix<double,6,2> & X, double& energy);
void computeQuadratic2DNeoHookeanEnergyGradient(double E, double nu, const Eigen::Matrix<double,6,2> & x, const Eigen::Matrix<double,6,2> & X, Eigen::Matrix<double, 12, 1>& energygradient);
void computeQuadratic2DNeoHookeanEnergyHessian(double E, double nu, const Eigen::Matrix<double,6,2> & x, const Eigen::Matrix<double,6,2> & X, Eigen::Matrix<double, 12, 12>& energyhessian);
void computeQuadratic2DNeoHookeandfdX(double E, double nu, const Eigen::Matrix<double,6,2> & x, const Eigen::Matrix<double,6,2> & X, Eigen::Matrix<double, 12, 12>& dfdX);

void computeNHEnergyFromGreenStrain(double E, double nu, const Eigen::Matrix<double,4,1> & Green_strain, double& energy);
void computeNHEnergyFromGreenStrainGradient(double E, double nu, const Eigen::Matrix<double,4,1> & Green_strain, Eigen::Matrix<double, 4, 1>& energygradient);
void computeNHEnergyFromGreenStrainHessian(double E, double nu, const Eigen::Matrix<double,4,1> & Green_strain, Eigen::Matrix<double, 4, 4>& energyhessian);

void neoHookeanVoigtDerivative(double lambda, double mu, const Eigen::Matrix<double,3,1> & E, Eigen::Matrix<double, 3, 1>& dPsidE, Eigen::Matrix<double, 3, 3>& d2PsidE2);
void computed2PsidE2(double lambda, double mu, const Eigen::Matrix<double,3,1> & epsilon, double& energy, Eigen::Matrix<double, 3, 1>& dPsidE, 
	Eigen::Matrix<double, 3, 3>& d2PsidE2);

void compute2DLinearMooneyRivilinEnergy(const Eigen::Matrix<double,3,2> & x, const Eigen::Matrix<double,3,2> & X, double C10, double C01, double C11, 
	double lambda, double& energy);
void compute2DLinearMooneyRivilinEnergyGradient(const Eigen::Matrix<double,3,2> & x, const Eigen::Matrix<double,3,2> & X, double C10, double C01, double C11, 
	double lambda, Eigen::Matrix<double, 6, 1>& energygradient);
void compute2DLinearMooneyRivilinEnergyHessian(const Eigen::Matrix<double,3,2> & x, const Eigen::Matrix<double,3,2> & X, double C10, double C01, double C11, 
	double lambda, Eigen::Matrix<double, 6, 6>& energyhessian);


void compute2DQuadraticMooneyRivilinEnergy(const Eigen::Matrix<double,6,2> & x, const Eigen::Matrix<double,6,2> & X, double C10, double C01, double C11, 
	double lambda, double& energy);
void compute2DQuadraticMooneyRivilinEnergyGradient(const Eigen::Matrix<double,6,2> & x, const Eigen::Matrix<double,6,2> & X, double C10, double C01, double C11, 
	double lambda, Eigen::Matrix<double, 12, 1>& energygradient);
void compute2DQuadraticMooneyRivilinEnergyHessian(const Eigen::Matrix<double,6,2> & x, const Eigen::Matrix<double,6,2> & X, double C10, double C01, double C11, 
	double lambda, Eigen::Matrix<double, 12, 12>& energyhessian);
#endif