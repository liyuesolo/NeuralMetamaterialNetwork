#ifndef FEM_SOLVER_H
#define FEM_SOLVER_H

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <unordered_map>
#include <complex>
#include <iomanip>

#include "VecMatDef.h"

#include "../include/Util.h"

enum PBCType
{
    PBC_XY, PBC_X, PBC_None
};
class FEMSolver
{
public:
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using MatrixXb = Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;
    using TV = Vector<T, 2>;
    using IV = Vector<int, 2>;
    using IV3 = Vector<int, 3>;
    using IV4 = Vector<int, 4>;
    using TV3 = Vector<T, 3>;
    using TM = Matrix<T, 2, 2>;
    using TM3 = Matrix<T, 3, 3>;

    using StiffnessMatrix = Eigen::SparseMatrix<T>;

    using Entry = Eigen::Triplet<T>;

    using QuadEleNodes = Matrix<T, 6, 2>;
    using QuadEleIdx = Vector<int, 6>;

    using EleNodes = Matrix<T, 3, 2>;
    using EleIdx = Vector<int, 3>;

    using Face = Vector<int, 3>;
    using Edge = Vector<int, 2>;

public:

    T E = 2.6 * 1e3; // N/cm2 -> TPU 95A 26M Pa
    T nu = 0.48;
    T lambda;

    bool use_mooney_rivlin = false;
    T C10, C01, D1, C20, C02, C11;

    int dim = 2;
    bool use_reduced = true;
    VectorXT u;
    VectorXT f;
    VectorXT deformed, undeformed;
    VectorXT reduced_dof;
    VectorXi indices, surface_indices;

    bool use_quadratic_triangle = false;

    // PBC
    PBCType pbc_type = PBC_None;
    std::string pbc_translation_file = "";
    bool add_pbc = false;
    T pbc_strain_w = 1e6;
    bool add_pbc_strain = false;
    TV t1, t2;
    std::vector<std::vector<IV>> pbc_pairs;
    T uniaxial_strain = 1.0;
    T uniaxial_strain_ortho = 1.0;
    bool biaxial = false;
    T strain_theta = 0.0;
    bool prescribe_strain_tensor = false;
    TV3 target_strain = TV3(1, 1, 0); // epsilon_xx epsilon_yy epsilon_xy
    Vector<int, 4> pbc_corners = Vector<int, 4>::Constant(-1);
    T scale = 1.0;

    std::unordered_map<int, T> dirichlet_data;
    std::unordered_map<int, T> penalty_pairs;

    int num_nodes = 0;   
    int num_ele = 0;

    bool project_block_PD = false;
    bool verbose = false;
    bool run_diff_test = false;

    T newton_tol = 1e-6;
    int max_newton_iter = 1000;

    bool use_ipc = false;
    bool add_friction = false;
    bool unilateral_qubic = false;
    T penalty_weight = 1e6;
    T y_bar = 0.0;
    T thickness = 1.0;
    

    // IPC
    T max_barrier_weight = 1e8;
    T friction_mu = 0.5;
    T epsv_times_h = 1e-5;
    int num_ipc_vtx = 0;
    T barrier_distance = 1e-5;
    T barrier_weight = 1.0;
    T ipc_min_dis = 1e-6;
    Eigen::MatrixXd ipc_vertices;
    Eigen::MatrixXd ipc_vertices_2x2;
    Eigen::MatrixXi ipc_edges;
    Eigen::MatrixXi ipc_edges_2x2;
    Eigen::MatrixXi ipc_faces;
    VectorXT coarse_to_fine;
    std::unordered_map<int, int> fine_to_coarse;
    StiffnessMatrix jacobian;
    StiffnessMatrix jac_full2reduced;
    std::vector<bool> is_pbc_vtx;
    std::vector<bool> is_interior_vtx;
    int translation_dof_offset = 0;
    MatrixXb is_boundary_edge; 


    template <class OP>
    void iterateDirichletDoF(const OP& f) {
        for (auto dirichlet: dirichlet_data){
            f(dirichlet.first, dirichlet.second);
        } 
    }

    template <class OP>
    void iterateBCPenaltyPairs(const OP& f)
    {
        for (auto pair : penalty_pairs)
        {
            f(pair.first, pair.second);
        }
    }

    template <typename OP>
    void iterateElementsSerial(const OP& f)
    {
        for (int i = 0; i < num_ele; i++)
        {
            EleIdx tet_idx = indices.segment<3>(i * 3);
            EleNodes ele_deformed = getEleNodesDeformed(tet_idx);
            EleNodes ele_undeformed = getEleNodesUndeformed(tet_idx);
            f(ele_deformed, ele_undeformed, tet_idx, i);
        }
    }

    template <typename OP>
    void iterateElementsParallel(const OP& f)
    {
        tbb::parallel_for(0, num_ele, [&](int i)
        {
            EleIdx ele_idx = indices.segment<3>(i * 3);
            EleNodes ele_deformed = getEleNodesDeformed(ele_idx);
            EleNodes ele_undeformed = getEleNodesUndeformed(ele_idx);
            f(ele_deformed, ele_undeformed, ele_idx, i);
        });
    }

    /*
Triangle:               Triangle6:          

v
^                                           
|                                           
2                       2                   
|`\                     |`\                 
|  `\                   |  `\               
|    `\                 4    `3             
|      `\               |      `\           
|        `\             |        `\         
0----------1 --> u      0-----5----1        

*/

    template <typename OP>
    void iterateQuadElementsSerial(const OP& f)
    {
        for (int i = 0; i < num_ele; i++)
        {
            QuadEleIdx ele_idx = indices.segment<6>(i * 6);
            QuadEleIdx ele_idx_reorder = ele_idx;
            ele_idx_reorder[3] = ele_idx[4];
            ele_idx_reorder[4] = ele_idx[5];
            ele_idx_reorder[5] = ele_idx[3];
            QuadEleNodes ele_deformed = getQuadEleNodesDeformed(ele_idx_reorder);
            QuadEleNodes ele_undeformed = getQuadEleNodesUndeformed(ele_idx_reorder);

            f(ele_deformed, ele_undeformed, ele_idx_reorder, i);
        }
    }

    template <typename OP>
    void iterateQuadElementsParallel(const OP& f)
    {
        tbb::parallel_for(0, num_ele, [&](int i)
        {
            QuadEleIdx ele_idx = indices.segment<6>(i * 6);
            QuadEleIdx ele_idx_reorder = ele_idx;
            ele_idx_reorder[3] = ele_idx[4];
            ele_idx_reorder[4] = ele_idx[5];
            ele_idx_reorder[5] = ele_idx[3];
            QuadEleNodes ele_deformed = getQuadEleNodesDeformed(ele_idx_reorder);
            QuadEleNodes ele_undeformed = getQuadEleNodesUndeformed(ele_idx_reorder);
            f(ele_deformed, ele_undeformed, ele_idx_reorder, i);
        });
    }

private:
    template<int dim0, int dim1>
    void clipMatrixMin(Matrix<T, dim0, dim1>& mat, T epsilon = 1e-12)
    {
        for (int i = 0; i < dim0; i++)
            for (int j = 0; j < dim1; j++)
                if (std::abs(mat(i, j)) < epsilon)
                    mat(i, j) = epsilon;
    }

    template<int size>
    bool isHessianBlockPD(const Matrix<T, size, size> & symMtr)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, size, size>> eigenSolver(symMtr);
        // sorted from the smallest to the largest
        if (eigenSolver.eigenvalues()[0] >= 0.0) 
            return true;
        else
            return false;
        
    }

    template<int size>
    VectorXT computeHessianBlockEigenValues(const Matrix<T, size, size> & symMtr)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, size, size>> eigenSolver(symMtr);
        return eigenSolver.eigenvalues();
    }

    template <int size>
    void projectBlockPD(Eigen::Matrix<T, size, size>& symMtr)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, size, size>> eigenSolver(symMtr);
        if (eigenSolver.eigenvalues()[0] >= 0.0) {
            return;
        }
        Eigen::DiagonalMatrix<T, size> D(eigenSolver.eigenvalues());
        int rows = ((size == Eigen::Dynamic) ? symMtr.rows() : size);
        for (int i = 0; i < rows; i++) {
            if (D.diagonal()[i] < 0.0) {
                D.diagonal()[i] = 0.0;
            }
            else {
                break;
            }
        }
        symMtr = eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
    }

    template<int size>
    void addForceEntry(VectorXT& residual, 
        const VectorXi& vtx_idx, 
        const Vector<T, size>& gradent)
    {
        if (vtx_idx.size() * 2 != size)
            std::cout << "wrong gradient block size in addForceEntry" << std::endl;

        for (int i = 0; i < vtx_idx.size(); i++)
            residual.segment<2>(vtx_idx[i] * 2) += gradent.template segment<2>(i * 2);
    }

    template<int dim>
    void getSubVector(const VectorXT& _vector, 
        const VectorXi& vtx_idx, 
        Vector<T, dim>& sub_vec)
    {
        if (vtx_idx.rows() * 2 != dim)
            std::cout << "wrong gradient block size in getSubVector" << std::endl;

        sub_vec = Vector<T, dim>::Zero(vtx_idx.rows() * 2);
        for (int i = 0; i < vtx_idx.rows(); i++)
        {
            sub_vec.template segment<2>(i * 2) = _vector.segment<2>(vtx_idx[i] * 2);
        }
    }

    template<int size>
    void addHessianEntry(
        std::vector<Entry>& triplets,
        const VectorXi& vtx_idx, 
        const Matrix<T, size, size>& hessian)
    {
        if (vtx_idx.size() * 2 != size)
            std::cout << "wrong hessian block size" << std::endl;

        for (int i = 0; i < vtx_idx.size(); i++)
        {
            int dof_i = vtx_idx[i];
            for (int j = 0; j < vtx_idx.size(); j++)
            {
                int dof_j = vtx_idx[j];
                for (int k = 0; k < 2; k++)
                    for (int l = 0; l < 2; l++)
                    {
                        if (std::abs(hessian(i * 2 + k, j * 2 + l)) > 1e-8)
                            triplets.push_back(Entry(dof_i * 2 + k, dof_j * 2 + l, hessian(i * 2 + k, j * 2 + l)));                
                    }
            }
        }
    }

    template<int size>
    void getHessianSubBlock(
        const StiffnessMatrix& hessian,
        const VectorXi& vtx_idx, 
        Matrix<T, size, size>& sub_hessian)
    {
        for (int i = 0; i < vtx_idx.size(); i++)
        {
            int dof_i = vtx_idx[i];
            for (int j = 0; j < vtx_idx.size(); j++)
            {
                int dof_j = vtx_idx[j];
                for (int k = 0; k < 2; k++)
                    for (int l = 0; l < 2; l++)
                    {
                        sub_hessian(i * 2 + k, j * 2 + l) = hessian.coeff(dof_i * 2 + k, dof_j * 2 + l);
                    }
            }
        }
    }

    inline T getSmallestPositiveRealQuadRoot(T a, T b, T c, T tol)
    {
        // return negative value if no positive real root is found
        using std::abs;
        using std::sqrt;
        T t;
        if (abs(a) <= tol) {
            if (abs(b) <= tol) // f(x) = c > 0 for all x
                t = -1;
            else
                t = -c / b;
        }
        else {
            T desc = b * b - 4 * a * c;
            if (desc > 0) {
                t = (-b - sqrt(desc)) / (2 * a);
                if (t < 0)
                    t = (-b + sqrt(desc)) / (2 * a);
            }
            else // desv<0 ==> imag
                t = -1;
        }
        return t;
    }

    inline T getSmallestPositiveRealCubicRoot(T a, T b, T c, T d, T tol = 1e-10)
    {
        // return negative value if no positive real root is found
        using std::abs;
        using std::complex;
        using std::pow;
        using std::sqrt;
        T t = -1;
        if (abs(a) <= tol)
            t = getSmallestPositiveRealQuadRoot(b, c, d, tol);
        else {
            complex<T> i(0, 1);
            complex<T> delta0(b * b - 3 * a * c, 0);
            complex<T> delta1(2 * b * b * b - 9 * a * b * c + 27 * a * a * d, 0);
            complex<T> C = pow((delta1 + sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0, 1.0 / 3.0);
            if (abs(C) < tol)
                C = pow((delta1 - sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0, 1.0 / 3.0);
            complex<T> u2 = (-1.0 + sqrt(3.0) * i) / 2.0;
            complex<T> u3 = (-1.0 - sqrt(3.0) * i) / 2.0;
            complex<T> t1 = (b + C + delta0 / C) / (-3.0 * a);
            complex<T> t2 = (b + u2 * C + delta0 / (u2 * C)) / (-3.0 * a);
            complex<T> t3 = (b + u3 * C + delta0 / (u3 * C)) / (-3.0 * a);
            if ((abs(imag(t1)) < tol) && (real(t1) > 0))
                t = real(t1);
            if ((abs(imag(t2)) < tol) && (real(t2) > 0) && ((real(t2) < t) || (t < 0)))
                t = real(t2);
            if ((abs(imag(t3)) < tol) && (real(t3) > 0) && ((real(t3) < t) || (t < 0)))
                t = real(t3);
        }
        return t;
    }

    std::vector<Entry> entriesFromSparseMatrix(const StiffnessMatrix& A)
    {
        std::vector<Entry> triplets;
        triplets.reserve(A.nonZeros());
        for (int k=0; k < A.outerSize(); ++k)
            for (StiffnessMatrix::InnerIterator it(A,k); it; ++it)
                triplets.push_back(Entry(it.row(), it.col(), it.value()));
        return triplets;
    }

    EleNodes getEleNodesDeformed(const EleIdx& nodal_indices)
    {
        EleNodes ele_x;
        for (int i = 0; i < 3; i++)
        {
            ele_x.row(i) = deformed.segment<2>(nodal_indices[i]*dim);
        }
        return ele_x;
    }

    EleNodes getEleNodesUndeformed(const EleIdx& nodal_indices)
    {
        EleNodes ele_x;
        for (int i = 0; i < 3; i++)
        {
            ele_x.row(i) = undeformed.segment<2>(nodal_indices[i]*dim);
        }
        return ele_x;
    }

    QuadEleNodes getQuadEleNodesDeformed(const QuadEleIdx& nodal_indices)
    {
        QuadEleNodes ele_x;
        for (int i = 0; i < 6; i++)
        {
            ele_x.row(i) = deformed.segment<2>(nodal_indices[i]*dim);
        }
        return ele_x;
    }

    QuadEleNodes getQuadEleNodesUndeformed(const QuadEleIdx& nodal_indices)
    {
        QuadEleNodes ele_x;
        for (int i = 0; i < 6; i++)
        {
            ele_x.row(i) = undeformed.segment<2>(nodal_indices[i]*dim);
        }
        return ele_x;
    }

    void getPBCPairsAxisDirection(std::vector<int>& side0, std::vector<int>& side1, int direction);
    void rotate(T angle)
    {
        TM R = rotMat(angle);
        tbb::parallel_for(0, num_nodes, [&](int i)
        {
            TV xi = undeformed.segment<2>(i * 2);
            undeformed.segment<2>(i * 2) = R * xi;
        });
        deformed = undeformed;
    }

    
public:

    void constructReducedJacobian();
    void fullDoFFromReduced(VectorXT& full_dof, const VectorXT& reduced);

    // DerivativeTest.cpp
    void checkTotalGradient(bool perturb);
    void checkTotalGradientScale(bool perturb);
    void checkTotalHessianScale(bool perturb);
    void checkTotalHessian(bool perturb);
    void checkdfdXScale(bool perturb);
    void checkdfdX(bool perturb);

    // Elasticity.cpp
    T computeTotalArea();
    T computeInversionFreeStepsize(const VectorXT& _u, const VectorXT& du);
    void addElastsicPotential(T& energy);
    void addElasticForceEntries(VectorXT& residual);
    void addElasticHessianEntries(std::vector<Entry>& entries, bool project_PD = false);
    void addElasticdfdXEntries(std::vector<Entry>& entries);
    void computeFirstPiola(VectorXT& PKStress);
    void computePrincipleStress(VectorXT& principle_stress);

    // PBC.cpp
    void computeDirectionStiffnessAnalytical(int n_samples, T strain_mag, VectorXT& stiffness_values);
    void computeDirectionStiffnessFiniteDifference(int n_samples, T strain_mag, VectorXT& stiffness_values);
    
    void computeHomogenizationElasticityTensor(T strain_dir, T strain_mag, 
        Matrix<T, 3, 3>& elasticity_tensor);
    void computeHomogenizationElasticityTensor(const TV3& strain_voigt, 
        Matrix<T, 3, 3>& elasticity_tensor);
    void computeHomogenizationElasticityTensorSA(T strain_dir, T strain_mag, 
        Matrix<T, 3, 3>& elasticity_tensor);
    void diffTestdxdE(const TV3& strain_voigt);
    void diffTestdfdE(const TV3& strain_voigt);
    void diffTestdxdEScale(const TV3& strain_voigt);
    void diffTestdfdEScale(const TV3& strain_voigt);
    void computedfdE(const TV3& strain_voigt, MatrixXT& dfdE);
    void computedxdE(const TV3& strain_voigt, MatrixXT& dxdE);
    void computeHomogenizationData(TM& secondPK_stress, TM& Green_strain, T& energy_density, int pair_idx = 0);
    void computeHomogenizationDataCauchy(TM& cauchy_stress, TM& cauchy_strain, T& energy_density);
    void computeHomogenizedStressStrain(TM& sigma, TM& epsilon);
    void computeHomogenizedStressStrain(TM& sigma, TM& Cauchy_strain, TM& Green_strain);
    void computeMarcoBoundaryIndices();
    void getMarcoBoundaryData(Matrix<T, 4, 2>& x, Matrix<T, 4, 2>& X, IV4& bd_indices);
    void getPBCPairData(int pair_idx, Matrix<T, 4, 2>& x, Matrix<T, 4, 2>& X, IV4& bd_indices);

    void addPBCPairInX();
    bool addPBCPairsXY();
    void getPBCPairs3D(std::vector<std::pair<TV3, TV3>>& pairs);
    void addPBCEnergy(T& energy);
    void addPBCForceEntries(VectorXT& residual);
    void addPBCHessianEntries(std::vector<Entry>& entries, bool project_PD = false);

    // IPC.cpp
    void updateIPCVertices(const VectorXT& _u);
    T computeCollisionFreeStepsize(const VectorXT& _u, const VectorXT& du);
    void computeIPCRestData();
    void updateBarrierInfo(bool first_step);
    void addIPCEnergy(T& energy);
    void addIPCForceEntries(VectorXT& residual);
    void addIPCHessianEntries(std::vector<Entry>& entries, 
        bool project_PD = false);
    void constructPeriodicContactPatch(
        const MatrixXT& ipc_vertices_unit, 
        MatrixXT& ipc_vertices_2x2, const VectorXT& position);

    // BoundaryCondition.cpp
    void addForceBox(const TV& min_corner, const TV& max_corner, const TV& force);
    void addDirichletBox(const TV& min_corner, const TV& max_corner, const TV& displacement);
    void addDirichletBoxY(const TV& min_corner, const TV& max_corner, const TV& displacement);
    void addDirichletBoxX(const TV& min_corner, const TV& max_corner, T dx);
    void addPenaltyPairsBox(const TV& min_corner, const TV& max_corner, const TV& displacement);
    void addPenaltyPairsDisk(const TV& center, T radius, const TV& displacement);
    void addPenaltyPairsBoxXY(const TV& min_corner, const TV& max_corner, const TV& displacement);
    void addPenaltyPairsBoxX(const TV& min_corner, const TV& max_corner, T dx);

    // Penalty.cpp
    void savePenaltyForces(const std::string& filename);
    void addBCPenaltyEnergy(T w, T& energy);
    void addBCPenaltyForceEntries(T w, VectorXT& residual);
    void addBCPenaltyHessianEntries(T w, std::vector<Entry>& entries);
    void addUnilateralQubicPenaltyEnergy(T w, T& energy);
    void addUnilateralQubicPenaltyForceEntries(T w, VectorXT& residuals);
    void addUnilateralQubicPenaltyHessianEntries(T w, std::vector<Entry>& entries);

    // FEMSolver.cpp
    T computeTotalEnergy(const VectorXT& _u);
    void buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K);
    void builddfdX(const VectorXT& _u, StiffnessMatrix& dfdX);
    T computeResidual(const VectorXT& _u, VectorXT& residual);
    T lineSearchNewton(VectorXT& _u,  VectorXT& residual);
    bool staticSolve();
    bool staticSolveStep(int step);
    bool linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du);
    void projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data);
    void reset();
    void checkHessianPD(bool save_txt = false);
    
    // Scene.cpp
    void loadOBJ(const std::string& filename, bool rest_shape = false);
    void saveIPCMesh(const std::string& filename);
    void saveToOBJ(const std::string& filename, bool rest_shape = false);
    void computeBoundingBox(TV& min_corner, TV& max_corner, bool rest_shape = false);

    FEMSolver() {}
    ~FEMSolver() {}

};

#endif
