#ifndef UTIL_H
#define UTIL_H

#include <fstream>
#include <iostream>

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>

#include "VecMatDef.h"



void loadMeshFromVTKFile(const std::string& filename, Eigen::MatrixXd& V, Eigen::MatrixXi& F);

void loadMeshFromVTKFile3D(const std::string& filename, Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXi& T);

void loadQuadraticTriangleMeshFromVTKFile(const std::string& filename, 
    Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXi& V_quad);

void loadPBCDataFromMSHFile(const std::string& filename, 
    std::vector<std::vector<Vector<int ,2>>>& pbc_pairs);
    // std::vector<Vector<int, 3>>& pbc_pairs);

Matrix<T, 2, 2> rotMat(T angle);

T angleToXaxis(Vector<T, 2>& vec);
T angleToYaxis(Vector<T, 2>& vec);


#endif