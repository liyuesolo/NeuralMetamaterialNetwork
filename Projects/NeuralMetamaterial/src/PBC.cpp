#include <Eigen/CholmodSupport>
#include "../include/FEMSolver.h"
#include "../include/autodiff/PBCEnergy.h"
#include "../include/autodiff/FEMEnergy.h"
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

bool FEMSolver::addPBCPairsXY()
{
    // std::cout << pbc_translation_file << std::endl;
    std::ifstream in(pbc_translation_file);
    TV t1, t2;
    in >> t1[0] >> t1[1] >> t2[0] >> t2[1];
    in.close();

    // std::cout << std::acos(t1.normalized().dot(TV(0.0, 1.0))) << std::endl;

    // set t1 to be closer to X-axis
    if (std::acos(std::abs(t1.normalized().dot(TV(0.0, 1.0)))) < M_PI * 0.25)
    {
        TV tmp = t1;
        t1 = t2; t2 = tmp;
    }

    // std::cout << t1.normalized().transpose() << " " << t2.normalized().transpose() << std::endl;

    // 2,3 quadrant
    if (t1.normalized().dot(TV(1.0, 0.0)) < 0.0)
    {   
        t1 *= -1.0;
    }
    // 3, 4 quadrant
    if (t2.normalized().dot(TV(0.0, 1.0)) < 0.0)
    {   
        t2 *= -1.0;
    }

    // rotate the structure to have one translation vector align with X
    // then gether the pairs in Y

    std::vector<int> dir0_side0, dir0_side1, dir1_side0, dir1_side1;
    T alpha = angleToXaxis(t1);
    rotate(-alpha);
    getPBCPairsAxisDirection(dir0_side0, dir0_side1, 1);
    rotate(alpha);
    alpha = angleToYaxis(t2);
    rotate(-alpha);
    getPBCPairsAxisDirection(dir1_side0, dir1_side1, 0);
    rotate(alpha);

    bool same_num_nodes_dir0 = dir0_side0.size() == dir0_side1.size();
    bool same_num_nodes_dir1 = dir1_side0.size() == dir1_side1.size();
    if (same_num_nodes_dir0 != same_num_nodes_dir1)
    {
        std::cout << same_num_nodes_dir0 << " " << same_num_nodes_dir1 << std::endl;
        return false;
    }
        
    
    pbc_pairs = {std::vector<IV>(), std::vector<IV>()};
    // std::cout <<  dir0_side0.size() << " " << dir0_side1.size()
    //     << " " << dir1_side0.size() << " " << dir1_side1.size() << std::endl;

    if (dir0_side0[0] == dir1_side0[0])
    {
        for (int i = 0; i < std::min(dir0_side0.size(), dir0_side1.size()); i++)
        {
            pbc_pairs[0].push_back(IV(dir0_side0[i], dir0_side1[i]));
        }
        for (int i = 0; i < std::min(dir1_side0.size(), dir1_side1.size()); i++)
        {
            pbc_pairs[1].push_back(IV(dir1_side0[i], dir1_side1[i]));
        }
    }
    else if (dir0_side0[0] == dir1_side1[0])
    {
        for (int i = 0; i < std::min(dir0_side0.size(), dir0_side1.size()); i++)
        {
            pbc_pairs[0].push_back(IV(dir0_side0[i], dir0_side1[i]));
        }
        for (int i = 0; i < std::min(dir1_side0.size(), dir1_side1.size()); i++)
        {
            pbc_pairs[1].push_back(IV(dir1_side1[i], dir1_side0[i]));
        }
    }
    else if (dir0_side1[0] == dir1_side0[0])
    {
        for (int i = 0; i < std::min(dir0_side0.size(), dir0_side1.size()); i++)
        {
            pbc_pairs[0].push_back(IV(dir0_side1[i], dir0_side0[i]));
        }
        for (int i = 0; i < std::min(dir1_side0.size(), dir1_side1.size()); i++)
        {
            pbc_pairs[1].push_back(IV(dir1_side0[i], dir1_side1[i]));
        }
    }
    else if (dir0_side1[0] == dir1_side1[0])
    {
        for (int i = 0; i < std::min(dir0_side0.size(), dir0_side1.size()); i++)
        {
            pbc_pairs[0].push_back(IV(dir0_side1[i], dir0_side0[i]));
        }
        for (int i = 0; i < std::min(dir1_side0.size(), dir1_side1.size()); i++)
        {
            pbc_pairs[1].push_back(IV(dir1_side1[i], dir1_side0[i]));
        }
    }
    else
    {
        for (int i = 0; i < std::min(dir0_side0.size(), dir0_side1.size()); i++)
        {
            pbc_pairs[0].push_back(IV(dir0_side0[i], dir0_side1[i]));
        }
        for (int i = 0; i < std::min(dir1_side0.size(), dir1_side1.size()); i++)
        {
            pbc_pairs[1].push_back(IV(dir1_side0[i], dir1_side1[i]));
        }
    }
    
    return same_num_nodes_dir0 && same_num_nodes_dir1;
}

void FEMSolver::getPBCPairsAxisDirection(std::vector<int>& side0, 
    std::vector<int>& side1, int direction)
{
    T thres_hold = 1e-4;
    int ortho = !direction;
    // std::cout << "dir " << direction << " " << ortho << std::endl;
    side0.clear(); side1.clear();
    TV min_corner, max_corner;
    computeBoundingBox(min_corner, max_corner);
    // std::cout << max_corner.transpose() << std::endl;
    // std::cout << min_corner.transpose() << std::endl;
    // std::getchar();
    for (int i = 0; i < num_nodes; i++)
    {
        TV xi = undeformed.segment<2>(i * 2);
        // std::cout << xi[direction] << " " << min_corner[direction] << std::endl;
        // std::getchar();
        if (std::abs(xi[direction] - min_corner[direction]) < thres_hold)
        {
            side0.push_back(i);
            // std::cout << "good" << std::endl;
        }
        if (std::abs(xi[direction] - max_corner[direction]) < thres_hold)
        {
            side1.push_back(i);
            // std::cout << "good too" << std::endl;
        }
    }
    
    std::sort(side0.begin(), side0.end(), [&](const int a, const int b){
        TV xa = undeformed.segment<2>(a * 2);
        TV xb = undeformed.segment<2>(b * 2);
        return xa[ortho] < xb[ortho];
    });
    std::sort(side1.begin(), side1.end(), [&](const int a, const int b){
        TV xa = undeformed.segment<2>(a * 2);
        TV xb = undeformed.segment<2>(b * 2);
        return xa[ortho] < xb[ortho];
    });
}

void FEMSolver::addPBCPairInX()
{
    TV min_corner, max_corner;
    computeBoundingBox(min_corner, max_corner);
    T dx = max_corner[0] - min_corner[0];
    std::vector<int> left_nodes, right_nodes;
    for (int i = 0; i < num_nodes; i++)
    {
        TV xi = undeformed.segment<2>(i * 2);
        if (std::abs(xi[0] - min_corner[0]) < 1e-6)
            left_nodes.push_back(i);
        if (std::abs(xi[0] - max_corner[0]) < 1e-6)
            right_nodes.push_back(i);
    }
    std::sort(left_nodes.begin(), left_nodes.end(), [&](const int a, const int b){
        TV xa = undeformed.segment<2>(a * 2);
        TV xb = undeformed.segment<2>(b * 2);
        return xa[1] < xb[1];
    });
    std::sort(right_nodes.begin(), right_nodes.end(), [&](const int a, const int b){
        TV xa = undeformed.segment<2>(a * 2);
        TV xb = undeformed.segment<2>(b * 2);
        return xa[1] < xb[1];
    });

    if (left_nodes.size() != right_nodes.size())
    {
        std::cout << left_nodes.size() << " " << right_nodes.size() << std::endl;
    }

    pbc_pairs = {std::vector<IV>(), std::vector<IV>()};
    // pbc_pairs.resize(2, std::vector<IV>());    
    
    for (int i = 0; i < left_nodes.size(); i++)
    // for (int i = 0; i < 2; i++)
    {
        TV x_left = undeformed.segment<2>(left_nodes[i] * 2);
        TV x_right = undeformed.segment<2>(right_nodes[i] * 2);
        // std::cout << "pairs in x " <<  left_nodes[i] << " " << right_nodes[i] << std::endl;
        // std::cout << (x_right - x_left).normalized().dot(TV(1, 0)) << std::endl;
        // std::cout << x_right.transpose() << " " << x_left.transpose() << std::endl;
        pbc_pairs[0].push_back(IV(left_nodes[i], right_nodes[i]));
    }
    
    std::cout << "#pbc pairs " << pbc_pairs[0].size() + pbc_pairs[1].size() << std::endl;
}

void FEMSolver::getPBCPairs3D(std::vector<std::pair<TV3, TV3>>& pairs)
{
    pairs.resize(0);
    for (int dir = 0; dir < 2; dir++)
        for (IV& pbc_pair : pbc_pairs[dir])
        {
            int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
            TV Xi = deformed.segment<2>(idx0 * 2);
            TV Xj = deformed.segment<2>(idx1 * 2);
            // std::cout << idx0 << " " << idx1 << " " << Xi.transpose() << " " << Xj.transpose() << std::endl;
            pairs.push_back(std::make_pair(TV3(Xi[0], Xi[1], 0.0), TV3(Xj[0], Xj[1], 0.0)));
        }
}

void FEMSolver::addPBCEnergy(T& energy)
{
    T energy_pbc = 0.0;
    TV strain_dir = TV(std::cos(strain_theta), std::sin(strain_theta));
    TV ortho_dir = TV(-std::sin(strain_theta), std::cos(strain_theta));
    auto addPBCEnergyDirection = [&](int dir)
    {
        for (auto pbc_pair : pbc_pairs[dir])
        {
            // strain term
            int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
            TV Xi = undeformed.segment<2>(idx0 * 2);
            TV Xj = undeformed.segment<2>(idx1 * 2);
            TV xi = deformed.segment<2>(idx0 * 2);
            TV xj = deformed.segment<2>(idx1 * 2);

            T Dij = (Xj - Xi).dot(strain_dir);
            T dij_target = Dij * uniaxial_strain;
            T dij = (xj - xi).dot(strain_dir);
            
            if (add_pbc_strain && !prescribe_strain_tensor)
                energy_pbc += 0.5 * pbc_strain_w * (dij - dij_target) * (dij - dij_target);
            if (add_pbc_strain && !prescribe_strain_tensor && biaxial)
            {
                Dij = (Xj - Xi).dot(ortho_dir);
                dij_target = Dij * uniaxial_strain_ortho;
                dij = (xj - xi).dot(ortho_dir);
                energy_pbc += 0.5 * pbc_strain_w * (dij - dij_target) * (dij - dij_target);
            }
        }
    };

    addPBCEnergyDirection(0);
    addPBCEnergyDirection(1);

    if (add_pbc_strain && prescribe_strain_tensor)
    {
        int n_pairs = std::min(pbc_pairs[0].size(), pbc_pairs[1].size());
        VectorXT strain_matching_energies(n_pairs);
        tbb::parallel_for(0, n_pairs, [&](int i)
        {
            Matrix<T, 4, 2> x, X;
            IV4 bd_indices;
            getPBCPairData(i, x, X, bd_indices);
            computeStrainMatchingEnergy(pbc_strain_w, target_strain, x, X, 
                strain_matching_energies[i]);
        });
        energy_pbc += strain_matching_energies.sum();
        // Matrix<T, 4, 2> x, X;
        // IV4 bd_indices;
        // getMarcoBoundaryData(x, X, bd_indices);
        // T strain_matching_term;
        // computeStrainMatchingEnergy(pbc_strain_w, target_strain, x, X, 
        //     strain_matching_term);
        // energy_pbc += strain_matching_term;
    }
    energy += energy_pbc;
}

void FEMSolver::addPBCForceEntries(VectorXT& residual)
{
    VectorXT pbc_force = residual;
    pbc_force.setZero();

    TV strain_dir = TV(std::cos(strain_theta), std::sin(strain_theta));
    TV ortho_dir = TV(-std::sin(strain_theta), std::cos(strain_theta));

    auto addPBCForceDirection = [&](int dir)
    {
        for (auto pbc_pair : pbc_pairs[dir])
        {
            int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
            
            TV Xi = undeformed.segment<2>(idx0 * 2);
            TV Xj = undeformed.segment<2>(idx1 * 2);
            TV xi = deformed.segment<2>(idx0 * 2);
            TV xj = deformed.segment<2>(idx1 * 2);

            T Dij = (Xj - Xi).dot(strain_dir);
            T dij_target = Dij * uniaxial_strain;
            T dij = (xj - xi).dot(strain_dir);
            if (add_pbc_strain && !prescribe_strain_tensor)
            {
                pbc_force.segment<2>(idx0 * 2) += pbc_strain_w * strain_dir * (dij - dij_target);
                pbc_force.segment<2>(idx1 * 2) -= pbc_strain_w * strain_dir * (dij - dij_target);
            }
            if (add_pbc_strain && !prescribe_strain_tensor && biaxial)
            {
                Dij = (Xj - Xi).dot(ortho_dir);
                dij_target = Dij * uniaxial_strain_ortho;
                dij = (xj - xi).dot(ortho_dir);
                pbc_force.segment<2>(idx0 * 2) += pbc_strain_w * ortho_dir * (dij - dij_target);
                pbc_force.segment<2>(idx1 * 2) -= pbc_strain_w * ortho_dir * (dij - dij_target);
            }
        }
    };

    addPBCForceDirection(0);
    addPBCForceDirection(1);
    if (add_pbc_strain && prescribe_strain_tensor)
    {
        int n_pairs = std::min(pbc_pairs[0].size(), pbc_pairs[1].size());
        tbb::parallel_for(0, n_pairs, [&](int i)
        {
            Matrix<T, 4, 2> x, X;
            IV4 bd_indices;
            getPBCPairData(i, x, X, bd_indices);
            Vector<T, 8> dedx;
            computeStrainMatchingEnergyGradient(pbc_strain_w, target_strain, x, X, 
                dedx);
            addForceEntry<8>(pbc_force, bd_indices, -dedx);
        });
        // Matrix<T, 4, 2> x, X;
        // IV4 bd_indices;
        // getMarcoBoundaryData(x, X, bd_indices);

        // Vector<T, 8> dedx;
        // computeStrainMatchingEnergyGradient(pbc_strain_w, target_strain, x, X, 
        //     dedx);
        // addForceEntry<8>(pbc_force, bd_indices, -dedx);
    }
    residual += pbc_force;
}

void FEMSolver::addPBCHessianEntries(std::vector<Entry>& entries, bool project_PD)
{
    TV strain_dir = TV(std::cos(strain_theta), std::sin(strain_theta));
    TV ortho_dir = TV(-std::sin(strain_theta), std::cos(strain_theta));
    auto addPBCHessianDirection = [&](int dir)
    {
        for (auto pbc_pair : pbc_pairs[dir])
        {
            int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
            TV Xi = undeformed.segment<2>(idx0 * 2);
            TV Xj = undeformed.segment<2>(idx1 * 2);
            TV xi = deformed.segment<2>(idx0 * 2);
            TV xj = deformed.segment<2>(idx1 * 2);

            T Dij = (Xj - Xi).dot(strain_dir);
            T dij_target = Dij * uniaxial_strain;
            T dij = (xj - xi).dot(strain_dir);

            TM Hessian = strain_dir * strain_dir.transpose();
            
            if (add_pbc_strain && !prescribe_strain_tensor)
            {
                for(int i = 0; i < dim; i++)
                {
                    for(int j = 0; j < dim; j++)
                    {
                        entries.push_back(Entry(idx0 * 2 + i, idx0 * 2 + j, pbc_strain_w * Hessian(i, j)));
                        entries.push_back(Entry(idx0 * 2 + i, idx1 * 2 + j, -pbc_strain_w * Hessian(i, j)));
                        entries.push_back(Entry(idx1 * 2 + i, idx0 * 2 + j, -pbc_strain_w * Hessian(i, j)));
                        entries.push_back(Entry(idx1 * 2 + i, idx1 * 2 + j, pbc_strain_w * Hessian(i, j)));
                    }
                }
            }
            if (add_pbc_strain && !prescribe_strain_tensor && biaxial)
            {
                Hessian = ortho_dir * ortho_dir.transpose();
                for(int i = 0; i < dim; i++)
                {
                    for(int j = 0; j < dim; j++)
                    {
                        entries.push_back(Entry(idx0 * 2 + i, idx0 * 2 + j, pbc_strain_w * Hessian(i, j)));
                        entries.push_back(Entry(idx0 * 2 + i, idx1 * 2 + j, -pbc_strain_w * Hessian(i, j)));
                        entries.push_back(Entry(idx1 * 2 + i, idx0 * 2 + j, -pbc_strain_w * Hessian(i, j)));
                        entries.push_back(Entry(idx1 * 2 + i, idx1 * 2 + j, pbc_strain_w * Hessian(i, j)));
                    }
                }
            }

        }
    };

    addPBCHessianDirection(0);
    addPBCHessianDirection(1);
    if (add_pbc_strain && prescribe_strain_tensor)
    {
        // Matrix<T, 4, 2> x, X;
        // IV4 bd_indices;
        // getMarcoBoundaryData(x, X, bd_indices);
        // Matrix<T, 8, 8> d2edx2;
        // computeStrainMatchingEnergyHessian(pbc_strain_w, target_strain, x, X, 
        //     d2edx2);
        // addHessianEntry<8>(entries, bd_indices, d2edx2);
        int n_pairs = std::min(pbc_pairs[0].size(), pbc_pairs[1].size());
        std::vector<Matrix<T, 8, 8>> sub_hessians(n_pairs);
        std::vector<IV4> indices(n_pairs);
        tbb::parallel_for(0, n_pairs, [&](int i)
        {
            Matrix<T, 4, 2> x, X;
            IV4 bd_indices;
            getPBCPairData(i, x, X, bd_indices);
            Matrix<T, 8, 8> d2edx2;
            computeStrainMatchingEnergyHessian(pbc_strain_w, target_strain, x, X, 
                d2edx2);
            sub_hessians[i] = d2edx2;
            indices[i] = bd_indices;
        });
        for (int i = 0; i < n_pairs; i++)
            addHessianEntry<8>(entries, indices[i], sub_hessians[i]);
    }
}

void FEMSolver::computeMarcoBoundaryIndices()
{
    pbc_corners << pbc_pairs[0][0][0], 
        pbc_pairs[0][pbc_pairs[0].size() - 1][0],
        pbc_pairs[0][pbc_pairs[0].size() - 1][1],
        pbc_pairs[0][0][1];
    
    // std::cout << pbc_corners << std::endl;
}

void FEMSolver::getPBCPairData(int pair_idx, Matrix<T, 4, 2>& x, Matrix<T, 4, 2>& X, IV4& bd_indices)
{
    TV xi = deformed.segment<2>(pbc_pairs[0][pair_idx][0] * 2);
    TV xj = deformed.segment<2>(pbc_pairs[0][pair_idx][1] * 2);
    TV xk = deformed.segment<2>(pbc_pairs[1][pair_idx][0] * 2);
    TV xl = deformed.segment<2>(pbc_pairs[1][pair_idx][1] * 2);
    x.row(0) = xi; x.row(1) = xj; x.row(2) = xk; x.row(3) = xl;

    TV Xi = undeformed.segment<2>(pbc_pairs[0][pair_idx][0] * 2);
    TV Xj = undeformed.segment<2>(pbc_pairs[0][pair_idx][1] * 2);
    TV Xk = undeformed.segment<2>(pbc_pairs[1][pair_idx][0] * 2);
    TV Xl = undeformed.segment<2>(pbc_pairs[1][pair_idx][1] * 2);
    X.row(0) = Xi; X.row(1) = Xj; X.row(2) = Xk; X.row(3) = Xl;

    bd_indices << pbc_pairs[0][pair_idx][0], pbc_pairs[0][pair_idx][1], pbc_pairs[1][pair_idx][0], pbc_pairs[1][pair_idx][1];
}

void FEMSolver::getMarcoBoundaryData(Matrix<T, 4, 2>& x, Matrix<T, 4, 2>& X, IV4& bd_indices)
{
    TV xi = deformed.segment<2>(pbc_pairs[0][0][0] * 2);
    TV xj = deformed.segment<2>(pbc_pairs[0][0][1] * 2);
    TV xk = deformed.segment<2>(pbc_pairs[1][0][0] * 2);
    TV xl = deformed.segment<2>(pbc_pairs[1][0][1] * 2);
    x.row(0) = xi; x.row(1) = xj; x.row(2) = xk; x.row(3) = xl;

    TV Xi = undeformed.segment<2>(pbc_pairs[0][0][0] * 2);
    TV Xj = undeformed.segment<2>(pbc_pairs[0][0][1] * 2);
    TV Xk = undeformed.segment<2>(pbc_pairs[1][0][0] * 2);
    TV Xl = undeformed.segment<2>(pbc_pairs[1][0][1] * 2);
    X.row(0) = Xi; X.row(1) = Xj; X.row(2) = Xk; X.row(3) = Xl;

    bd_indices << pbc_pairs[0][0][0], pbc_pairs[0][0][1], pbc_pairs[1][0][0], pbc_pairs[1][0][1];
}

void FEMSolver::computeHomogenizationDataCauchy(TM& cauchy_stress, TM& cauchy_strain, T& energy_density)
{
    TV xi = deformed.segment<2>(pbc_pairs[0][0][0] * 2);
    TV xj = deformed.segment<2>(pbc_pairs[0][0][1] * 2);

    TV xk = deformed.segment<2>(pbc_pairs[1][0][0] * 2);
    TV xl = deformed.segment<2>(pbc_pairs[1][0][1] * 2);


    TV Xi = undeformed.segment<2>(pbc_pairs[0][0][0] * 2);
    TV Xj = undeformed.segment<2>(pbc_pairs[0][0][1] * 2);
    TV Xk = undeformed.segment<2>(pbc_pairs[1][0][0] * 2);
    TV Xl = undeformed.segment<2>(pbc_pairs[1][0][1] * 2);


    TV tij = deformed.segment<2>(num_nodes*2), tkl = deformed.segment<2>(num_nodes*2 + 2);
    TV Tij = undeformed.segment<2>(num_nodes*2), Tkl = undeformed.segment<2>(num_nodes*2 + 2);

    VectorXT inner_force(num_nodes * 2 + 4);
    inner_force.setZero(); 
    addPBCForceEntries(inner_force);

    TV f0 = TV::Zero(), f1 = TV::Zero();
    // T l0 = (xj - xi).norm(), l1 = (xl - xk).norm();

    T l0 = tij.norm(), l1 = tkl.norm();

    for (auto pbc_pair : pbc_pairs[0])
    {
        f0 += inner_force.segment<2>(pbc_pair[1] * 2) / l1 / thickness;
    }
    for (auto pbc_pair : pbc_pairs[1])
    {
        f1 += inner_force.segment<2>(pbc_pair[0] * 2) / l0 / thickness;
    }

    TM R90 = TM::Zero();
    R90.row(0) = TV(0, -1);
    R90.row(1) = TV(1, 0);

    TM _X = TM::Zero(), _x = TM::Zero();
    _X.col(0) = Tij;
    _X.col(1) = Tkl;

    _x.col(0) = tij;
    _x.col(1) = tkl;

    TM F_macro = _x * _X.inverse();
    
    cauchy_strain = 0.5 * (F_macro.transpose() + F_macro) - TM::Identity();

     TV n1 = (R90 * tij.normalized()).normalized(), 
        n0 = (R90 * tkl.normalized()).normalized();
        
    // std::cout << "f0 " << f0.transpose() << " f1 " << f1.transpose() << std::endl;
    
    TM f_bc = TM::Zero(), n_bc = TM::Zero();
    
    f_bc.col(0) = f0; f_bc.col(1) = f1;
    
    n_bc.col(0) = n0; n_bc.col(1) = n1;

    cauchy_stress = f_bc * n_bc.inverse();

    T volume = std::abs(tij[0] * tkl[1] - tij[1] * tkl[0]) * thickness;
    T total_energy = computeTotalEnergy(u);
    energy_density = total_energy / volume;
}


void FEMSolver::computeDirectionStiffnessFiniteDifference(int n_samples, T strain_mag, VectorXT& stiffness_values)
{
    
    T dtheta = M_PI / T(n_samples);
    stiffness_values.resize(n_samples);
    for (int i = 0; i < n_samples; i++)
    {
        TM3 elasticity_tensor;
        T theta = dtheta * T(i);
        computeHomogenizationElasticityTensor(theta, strain_mag, elasticity_tensor);
        TV3 d_voigt = TV3(std::cos(theta) * std::cos(theta),
                            std::sin(theta) * std::sin(theta),
                            std::cos(theta) * std::sin(theta));
        stiffness_values[i] = 1.0 / (d_voigt.transpose() * (elasticity_tensor.inverse() * d_voigt));
    }
}

void FEMSolver::computeDirectionStiffnessAnalytical(int n_samples, T strain_mag, VectorXT& stiffness_values)
{
    T dtheta = M_PI / T(n_samples);
    stiffness_values.resize(n_samples);
    for (int i = 0; i < n_samples; i++)
    {
        TM3 elasticity_tensor;
        T theta = dtheta * T(i);
        computeHomogenizationElasticityTensorSA(theta, strain_mag, elasticity_tensor);
        // TM3 elasticity_tensor_FD;
        // computeHomogenizationElasticityTensor(theta, strain_mag, elasticity_tensor_FD);
        // std::cout << "elasticity tensor " << elasticity_tensor << std::endl << std::endl;
        // std::cout << "elasticity tensor FD " << elasticity_tensor_FD << std::endl << std::endl;
        // std::getchar();
        TV3 d_voigt = TV3(std::cos(theta) * std::cos(theta),
                            std::sin(theta) * std::sin(theta),
                            std::cos(theta) * std::sin(theta));
        stiffness_values[i] = 1.0 / (d_voigt.transpose() * (elasticity_tensor.inverse() * d_voigt));
    }
}
void FEMSolver::computeHomogenizationElasticityTensorSA(T strain_dir, T strain_mag, 
        Matrix<T, 3, 3>& elasticity_tensor)
{
    // std::cout << pbc_pairs[0].size() << " " << pbc_pairs[1].size() << " " << num_nodes << std::endl;
    // S = S(x(E))
    // dS/dE = dS/dx dx/dE
    prescribe_strain_tensor = false;
    elasticity_tensor.setZero();
    uniaxial_strain = strain_mag;
    strain_theta = strain_dir;
    biaxial = false;
    // pbc_strain_w = 1e6;
    staticSolve();
    // pbc_strain_w = 1e10;
    // staticSolve();
    TM stress_2ndPK, strain_Green;
    T energy_density;
    computeHomogenizationData(stress_2ndPK, strain_Green, energy_density);
    Vector<T, 3> strain_voigt;
    strain_voigt << strain_Green(0, 0), strain_Green(1, 1), 2.0 * strain_Green(0, 1);

    prescribe_strain_tensor = true;
    pbc_strain_w = 1e10;
    
    target_strain = strain_voigt;
    
    staticSolve();
    // ==========================

    TM stress_2ndPK_again, strain_Green_again;
    T energy_again;

    TV xi = deformed.segment<2>(pbc_pairs[0][0][0] * 2);
    TV xj = deformed.segment<2>(pbc_pairs[0][0][1] * 2);
    TV xk = deformed.segment<2>(pbc_pairs[1][0][0] * 2);
    TV xl = deformed.segment<2>(pbc_pairs[1][0][1] * 2);

    TV Xi = undeformed.segment<2>(pbc_pairs[0][0][0] * 2);
    TV Xj = undeformed.segment<2>(pbc_pairs[0][0][1] * 2);
    TV Xk = undeformed.segment<2>(pbc_pairs[1][0][0] * 2);
    TV Xl = undeformed.segment<2>(pbc_pairs[1][0][1] * 2);

    VectorXi pair_idx(4);
    pair_idx << pbc_pairs[0][0][0], pbc_pairs[0][0][1], pbc_pairs[1][0][0], pbc_pairs[1][0][1];

    T sign = 1.0;

    TV dx = Xj - Xi, dy = Xk - Xl;
    // std::cout << dx.norm() << " " << dy.norm() << " " << thickness << " " << E << " " << nu << std::endl;
    if ((Xi - Xl).norm() > 1e-6)
    {
        if ((Xk - Xj).norm() > 1e-6)
        {
            std::cout << "ALERT" << std::endl;
            std::cout << (Xi - Xl).norm() << " " << Xi.transpose() << " " << Xl.transpose() << " " << Xk.transpose() << " " << Xj.transpose() << std::endl;
            std::getchar();
        }
        else
        {
            sign = -1.0;
            dx = Xi - Xj;
            dy = Xl - Xk;
        }
    }

    VectorXT inner_force(num_nodes * 2);
    inner_force.setZero(); 
    addPBCForceEntries(inner_force);

    iterateDirichletDoF([&](int offset, T target)
    {
        inner_force[offset] = 0.0;
    });

    TV f0 = TV::Zero(), f1 = TV::Zero(), f0_unscaled = TV::Zero(), f1_unscaled = TV::Zero();
    
    MatrixXT dfunscaled_df(4, num_nodes * 2); dfunscaled_df.setZero();
    for (auto pbc_pair : pbc_pairs[0])
    {
        f0_unscaled += inner_force.segment<2>(pbc_pair[0] * 2);
        dfunscaled_df.block(0, pbc_pair[0] * 2, 2, 2).setIdentity();
    }
    for (auto pbc_pair : pbc_pairs[1])
    {
        f1_unscaled += inner_force.segment<2>(pbc_pair[1] * 2);
        dfunscaled_df.block(2, pbc_pair[1] * 2, 2, 2).setIdentity();
    }

    TM R90 = TM::Zero();
    R90.row(0) = TV(0, -1);
    R90.row(1) = TV(1, 0);

    TM _X = TM::Zero(), _x = TM::Zero();
    _X.col(0) = (Xj - Xi);
    _X.col(1) = (Xk - Xl);

    _x.col(0) = (xj - xi);
    _x.col(1) = (xk - xl);

    TM F_macro = _x * _X.inverse();

    std::cout << "deformation gradient macro " << F_macro << std::endl << std::endl;

    Matrix<T, 3, 4> dSdF, dSdf, dSdn; Matrix<T, 4, 8> dFdx, dndx; 
    computed2ndPKdmany(xi, xj, xk, xl, Xi, Xj, Xk, Xl, f0_unscaled, f1_unscaled, 
        sign, thickness, dSdF, dFdx, dSdf, dndx, dSdn);

    clipMatrixMin<3, 4>(dSdf); clipMatrixMin<3, 4>(dSdn);
    // d_internal_force / dE
    MatrixXT dfdE;
    computedfdE(strain_voigt, dfdE);

    elasticity_tensor = dSdf * (dfunscaled_df * dfdE);
    std::cout << elasticity_tensor << std::endl;
    // std::getchar();
    std::cout << "dSdF" << std::endl;
    std::cout << dSdF << std::endl << std::endl;

    std::cout << "dFdx" << std::endl;
    std::cout << dFdx << std::endl << std::endl;

    std::cout << "dSdf" << std::endl;
    std::cout << dSdf << std::endl << std::endl;

    std::cout << "dSdn" << std::endl;
    std::cout << dSdn << std::endl << std::endl;

    std::cout << "dndx" << std::endl;
    std::cout << dndx << std::endl << std::endl;

    // std::getchar();

    MatrixXT dFdx_full(4, num_nodes * 2), dndx_full(4, num_nodes * 2);
    dFdx_full.setZero(); dndx_full.setZero();
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            for (int d = 0; d < 2; d++)
            {
                dFdx_full(i, pair_idx[j] * 2 + d) = dFdx(i, j * 2 + d);
                dndx_full(i, pair_idx[j] * 2 + d) = dndx(i, j * 2 + d);
            }
        }
    }
    iterateDirichletDoF([&](int offset, T target)
    {
        dFdx_full.col(offset).setZero();
        dndx_full.col(offset).setZero();
    });
    StiffnessMatrix dfdx_pbc(num_nodes * 2, num_nodes * 2);
    std::vector<Entry> entries;
    addPBCHessianEntries(entries);
    dfdx_pbc.setFromTriplets(entries.begin(), entries.end());
    projectDirichletDoFMatrix(dfdx_pbc, dirichlet_data);
    // std::cout << "dFdx dndx full" << std::endl;
    MatrixXT dSdFdFdx = dSdF * dFdx_full;
    // std::cout << "dSdFdFdx" << std::endl;
    // std::cout << dSdFdFdx << std::endl;
    MatrixXT dSdsigma_dsigmadx = dSdf * (dfunscaled_df * -dfdx_pbc) + dSdn * dndx_full;
    // std::cout << "dSdsigma_dsigmadx" << std::endl;
    // std::cout << dSdsigma_dsigmadx << std::endl;
    
    MatrixXT dxdE(num_nodes * 2, 3);
    computedxdE(strain_voigt, dxdE);

    auto chain_rule_first_part = (dSdFdFdx + dSdsigma_dsigmadx) * dxdE;
    std::cout << "chain_rule_first_part " << chain_rule_first_part << std::endl << std::endl;
    elasticity_tensor += chain_rule_first_part;

    TM3 tensor_GT;
    TV3 stress_voigt;
    T mu = E / 2.0 / (1.0 + nu);
    T lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
    T E0;
    computed2PsidE2(lambda, mu, strain_voigt, E0, stress_voigt, tensor_GT);
    std::cout << "Psi GT " << E0 << std::endl;
    std::cout << "stress GT " << stress_voigt.transpose() << std::endl;
    std::cout << "tensor GT " << tensor_GT << std::endl << std::endl;
}


void FEMSolver::diffTestdfdEScale(const TV3& strain_voigt)
{
    target_strain = strain_voigt;
    prescribe_strain_tensor = true;
    pbc_strain_w = 1e6;
    staticSolve();
    pbc_strain_w = 1e10;
    staticSolve();
    VectorXT f0(num_nodes * 2), f1(num_nodes * 2);
    f0.setZero(); f1.setZero();
    computeResidual(u, f0);
    f0 *= -1.0;

    MatrixXT dfdE;
    computedfdE(strain_voigt, dfdE);
    
    int n_dof = 3;
    Vector<T, 3> dE;
    dE << 0.0015, 0.0008, 0.00012;

    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {
        target_strain += dE;
        computeResidual(u, f1);
        f1 *= -1.0;
        target_strain -= dE;
        T df_norm = (f0 + (dfdE * dE) - f1).norm();
        // std::cout << "df_norm " << df_norm << std::endl;
        if (i > 0)
        {
            std::cout << (previous/df_norm) << std::endl;
        }
        previous = df_norm;
        dE *= 0.5;
    }
}

void FEMSolver::diffTestdxdEScale(const TV3& strain_voigt)
{
    verbose = false;
    target_strain = strain_voigt;
    prescribe_strain_tensor = true;
    pbc_strain_w = 1e6;
    staticSolve();
    pbc_strain_w = 1e9;
    staticSolve();
    VectorXT f0 = deformed;
    MatrixXT dxdE;
    computedxdE(strain_voigt, dxdE);
    
    int n_dof = 3;
    Vector<T, 3> dE;
    dE << 0.00015, 0.00008, 0.00012;

    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {
        target_strain += dE;
        staticSolve();
        VectorXT f1 = deformed;
        target_strain -= dE;
        T df_norm = (f0 + (dxdE * dE) - f1).norm();
        // std::cout << "df_norm " << df_norm << std::endl;
        if (i > 0)
        {
            std::cout << (previous/df_norm) << std::endl;
        }
        previous = df_norm;
        dE *= 0.5;
    }
}

void FEMSolver::diffTestdxdE(const TV3& strain_voigt)
{
    auto cross2D = [&](const TV& vec1, const TV& vec2) ->T
    {
        return vec1[0] * vec2[1] - vec1[1] * vec2[0];
    };


    // newton_tol = 1e-8;
    verbose = false;
    target_strain = strain_voigt;
    prescribe_strain_tensor = false;
    pbc_strain_w = 1e6;
    staticSolve();
        
    Matrix<T, 4, 2> x, X;
    IV4 bd_indices;
    getPBCPairData(0, x, X, bd_indices);

    T cross_produc_sum = 0;
    TV center_X = 0.25 * (X.row(0) + X.row(1) + X.row(2) + X.row(3));
    TV center_x = 0.25 * (x.row(0) + x.row(1) + x.row(2) + x.row(3));
    for (int i = 0; i < 4; i++)
    {
        cross_produc_sum += cross2D((x.row(i).transpose() - center_x).normalized(), (X.row(i).transpose() - center_X).normalized());
    }
    // TV center = TV::Zero(), center_undeformed = TV::Zero();
    // for (int i = 0; i < num_nodes; i++)
    // {
    //     center_undeformed += undeformed.segment<2>(i * 2);
    //     center += deformed.segment<2>(i* 2);
    // }
    // center /= num_nodes;

    // for (int i = 0; i < num_nodes; i++)
    // {
    //     cross_produc_sum += cross2D(deformed.segment<2>(i * 2) - deformed.segment<2>(pbc_pairs[0][0][0] * 2), undeformed.segment<2>(i*2) - undeformed.segment<2>(pbc_pairs[0][0][0] * 2));
    // }
    
    
    std::cout << cross_produc_sum << std::endl;

    std::exit(0);
    pbc_strain_w = 1e10;
    staticSolve();
    VectorXT current_x = deformed;
    MatrixXT dxdE;
    computedxdE(strain_voigt, dxdE);

    VectorXT forward_x, backward_x;
    T epsilon = 1e-5;
    for (int i = 1; i < 3; i++)
    {
        target_strain[i] += epsilon;
        staticSolve();
        forward_x = deformed;
        target_strain[i] -= 2.0 * epsilon;
        staticSolve();
        backward_x = deformed;
        target_strain[i] += epsilon;
        
        for (int j = 0; j < num_nodes * 2; j++)
        {
            T fd = (forward_x[j] - backward_x[j]) / 2.0 / epsilon;
            if (std::abs(fd) < 1e-6 && std::abs(dxdE(j, i)) < 1e-6)
                continue;
            // if (std::abs(fd) < 1e-6)
            //     continue;
            std::cout << "dof " << j << " strain entry " << i << " fd " << fd << " symbolic: " << dxdE(j, i) << std::endl;
            std::getchar(); 
        }
        
    }
    std::cout << "diff test dxdE passed " << std::endl;
}

void FEMSolver::computedfdE(const TV3& strain_voigt, MatrixXT& dfdE)
{
    dfdE.resize(num_nodes * 2, 3); dfdE.setZero();
    int n_pairs = std::min(pbc_pairs[0].size(), pbc_pairs[1].size());
    tbb::parallel_for(0, n_pairs, [&](int i)
    {
        Matrix<T, 4, 2> x, X;
        IV4 bd_indices;
        getPBCPairData(i, x, X, bd_indices);
        // std::cout << bd_indices.transpose() << std::endl;
        // std::getchar();
        Matrix<T, 8, 3> dfdE_local;
        computedfdELocal(pbc_strain_w, strain_voigt, x, X, dfdE_local);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++)
                for (int d = 0; d < 2; d++)
                    dfdE(bd_indices[i] * 2 + d, j) += dfdE_local(i * 2 + d, j);
    }
    );
    iterateDirichletDoF([&](int offset, T target)
    {
        dfdE.row(offset).setZero();
    });
}

void FEMSolver::diffTestdfdE(const TV3& strain_voigt)
{
    verbose = false;
    target_strain = strain_voigt;
    prescribe_strain_tensor = true;
    pbc_strain_w = 1e6;
    staticSolve();
    pbc_strain_w = 1e10;
    staticSolve();

    MatrixXT dfdE;
    computedfdE(strain_voigt, dfdE);
    VectorXT forward_f(num_nodes * 2), backward_f(num_nodes * 2);
    forward_f.setZero(); backward_f.setZero();
    T epsilon = 1e-5;
    for (int i = 0; i < 3; i++)
    {
        target_strain[i] += epsilon;
        // staticSolve();
        computeResidual(u, forward_f);
        // addPBCForceEntries(forward_f);
        target_strain[i] -= 2.0 * epsilon;
        // staticSolve();
        computeResidual(u, backward_f);
        // addPBCForceEntries(backward_f);
        target_strain[i] += epsilon;
        
        for (int j = 0; j < num_nodes * 2; j++)
        {
            T fd = (forward_f[j] - backward_f[j]) / 2.0 / epsilon;
            if (std::abs(fd) < 1e-6 && std::abs(dfdE(j, i)) < 1e-6)
                continue;
            // if (std::abs( dfdE(j, i) - fd) < 1e-3 * std::abs(fd))
            //     continue;
            // if (std::abs(fd) < 1e-6)
            //     continue;
            std::cout << "dof " << j << " strain entry " << i << " fd " << fd << " symbolic: " << dfdE(j, i) << std::endl;
            std::getchar(); 
        }
        
    }
    std::cout << "diff test dfdE passed " << std::endl;
}

void FEMSolver::computedxdE(const TV3& strain_voigt, MatrixXT& dxdE)
{
    // staticSolve();
    MatrixXT dfdE;
    computedfdE(strain_voigt, dfdE);

    dxdE.resize(num_nodes * 2, 3); dxdE.setZero();
    StiffnessMatrix global_hessian(num_nodes * 2, num_nodes * 2);
    buildSystemMatrix(u, global_hessian);
    
    // MatrixXT H_dense = global_hessian;
    // std::cout << H_dense.maxCoeff() << " " << H_dense.minCoeff() << std::endl;
    StiffnessMatrix H(num_nodes * 2, num_nodes * 2);
    H.setIdentity(); H.diagonal().array() = 1e-8;
    // global_hessian += H;
    
    Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> solver;
    // Eigen::SparseLU<StiffnessMatrix> solver;
    solver.analyzePattern(global_hessian);
    solver.factorize(global_hessian);
    if (solver.info() == Eigen::NumericalIssue)
        std::cout << "!!!indefinite matrix!!!" << std::endl;

    int nmodes = 5;
    Spectra::SparseSymShiftSolve<T, Eigen::Lower> op(global_hessian);
        // T shift = indefinite ? -1e2 : -1e-4;
    T shift = -1e-4;
    Spectra::SymEigsShiftSolver<T, 
    Spectra::LARGEST_MAGN, 
    Spectra::SparseSymShiftSolve<T, Eigen::Lower> > 
        eigs(&op, nmodes, 2 * nmodes, shift);

    eigs.init();

    int nconv = eigs.compute();

    if (eigs.info() == Spectra::SUCCESSFUL)
    {
        Eigen::MatrixXd eigen_vectors = eigs.eigenvectors().real();
        Eigen::VectorXd eigen_values = eigs.eigenvalues().real();
        std::cout << eigen_values.transpose() << std::endl;
    }
    // std::cout << global_hessian.rows() << " " << global_hessian.cols() << " " << dfdE.rows() << " " << dfdE.cols() << std::endl;
    dxdE = solver.solve(dfdE);

    // std::cout << (global_hessian * dxdE - dfdE).norm() / dfdE.norm() << std::endl;

    iterateDirichletDoF([&](int offset, T target)
    {
        dxdE.row(offset).setZero();
    });
    // std::cout << dxdE << std::endl;
    // std::exit(0);
}

void FEMSolver::computeHomogenizationElasticityTensor(const TV3& strain_voigt, 
        Matrix<T, 3, 3>& elasticity_tensor)
{
    T E1, E2;
    elasticity_tensor.setZero();
    T epsilon = 1e-4;
    prescribe_strain_tensor = true;
    pbc_strain_w = 1e3;
    target_strain = strain_voigt;
    staticSolve();
    pbc_strain_w = 1e7;
    staticSolve();
    // pbc_strain_w = 1e7;
    TM stress_forward, stress_backward, dummy;
    // epsilon = 1e-4 * strain_voigt[0];
    // C(0, 0) 
    target_strain = TV3(strain_voigt[0] + epsilon, strain_voigt[1], strain_voigt[2]);
    
    staticSolve();
    computeHomogenizationData(stress_forward, dummy, E1);
    target_strain = TV3(strain_voigt[0] - epsilon, strain_voigt[1], strain_voigt[2]);
    staticSolve();
    computeHomogenizationData(stress_backward, dummy, E2);
    elasticity_tensor(0, 0) = (stress_forward(0, 0) - stress_backward(0, 0)) / epsilon / 2.0;
    elasticity_tensor(0, 1) = (stress_forward(1, 1) - stress_backward(1, 1)) / epsilon / 2.0;
    elasticity_tensor(2, 0) = elasticity_tensor(0, 2) = (stress_forward(0, 1) - stress_backward(0, 1)) / epsilon / 2.0;
    // std::cout << "stress forward " << stress_forward << std::endl << std::endl;
    // std::cout << "stress backward " << stress_forward << std::endl << std::endl;

    // C(1, 1), C(0, 1), C(1, 0) 
    // epsilon = 1e-3 * strain_voigt[1];
    target_strain = TV3(strain_voigt[0], strain_voigt[1] + epsilon, strain_voigt[2]);
    staticSolve();
    computeHomogenizationData(stress_forward, dummy, E1);
    target_strain = TV3(strain_voigt[0], strain_voigt[1] - epsilon, strain_voigt[2]);
    staticSolve();
    computeHomogenizationData(stress_backward, dummy, E2);
    elasticity_tensor(1, 0) = (stress_forward(0, 0) - stress_backward(0, 0)) / epsilon / 2.0;
    elasticity_tensor(1, 1) = (stress_forward(1, 1) - stress_backward(1, 1)) / epsilon / 2.0;
    elasticity_tensor(2, 1) = elasticity_tensor(1, 2) = (stress_forward(1, 0) - stress_backward(1, 0)) / epsilon / 2.0;

    // C(2, 2), C(0, 2), C(2, 0), C(1, 2), C(2, 1) 
    // epsilon = 1e-3 * strain_voigt[2];
    target_strain = TV3(strain_voigt[0], strain_voigt[1], strain_voigt[2] + epsilon);
    staticSolve();
    computeHomogenizationData(stress_forward, dummy, E1);
    target_strain = TV3(strain_voigt[0], strain_voigt[1], strain_voigt[2] -  epsilon);
    staticSolve();
    computeHomogenizationData(stress_backward, dummy, E2);

    // elasticity_tensor(2, 0) = (stress_forward(0, 0) - stress_backward(0, 0)) / epsilon / 2.0;
    // elasticity_tensor(2, 1) = (stress_forward(1, 1) - stress_backward(1, 1)) / epsilon / 2.0;
    elasticity_tensor(2, 2) = (stress_forward(0, 1) - stress_backward(0, 1)) / epsilon / 2.0;
}

void FEMSolver::computeHomogenizationElasticityTensor(
    T strain_dir, T strain_mag, Matrix<T, 3, 3>& elasticity_tensor)
{
    // pbc_strain_w = 1e6;
    // pbc_w = 1e7;

    prescribe_strain_tensor = false;
    elasticity_tensor.setZero();
    // reset();
    uniaxial_strain = strain_mag;
    strain_theta = strain_dir;
    biaxial = false;
    staticSolve();
    TM stress_2ndPK, strain_Green;
    T E0, E1, E2;
    computeHomogenizationData(stress_2ndPK, strain_Green, E0);
    std::cout << "Psi " << E0 << std::endl;
    std::cout << "Green strain " << strain_Green << std::endl << std::endl;
    std::cout << "2nd PK stress " << stress_2ndPK(0, 0) << " " << stress_2ndPK(1, 1) << " " << stress_2ndPK(0, 1) << std::endl << std::endl;
    T mu = E / 2.0 / (1.0 + nu);
    T lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
    TV3 strain_voigt; strain_voigt << strain_Green(0, 0), strain_Green(1, 1), 2.0 * strain_Green(0, 1);
    // TM3 tensor_GT;
    // TV3 stress_voigt;
    // computed2PsidE2(lambda, mu, strain_voigt, E0, stress_voigt, tensor_GT);
    // std::cout << "Psi GT " << E0 << std::endl;
    // std::cout << "stress GT " << stress_voigt.transpose() << std::endl;
    // std::cout << "tensor GT " << tensor_GT << std::endl << std::endl;
    // std::cout << "2nd PK stress " << stress_2ndPK << std::endl << std::endl;
    T epsilon = 1e-4;
    prescribe_strain_tensor = true;
    pbc_strain_w = 1e7;
    // pbc_strain_w = 1e7;
    TM stress_forward, stress_backward, dummy;
    // epsilon = 1e-4 * strain_voigt[0];
    // C(0, 0) 
    target_strain = TV3(strain_voigt[0] + epsilon, strain_voigt[1], strain_voigt[2]);
    staticSolve();
    computeHomogenizationData(stress_forward, dummy, E1);
    target_strain = TV3(strain_voigt[0] - epsilon, strain_voigt[1], strain_voigt[2]);
    staticSolve();
    computeHomogenizationData(stress_backward, dummy, E2);
    elasticity_tensor(0, 0) = (stress_forward(0, 0) - stress_backward(0, 0)) / epsilon / 2.0;
    elasticity_tensor(0, 1) = (stress_forward(1, 1) - stress_backward(1, 1)) / epsilon / 2.0;
    elasticity_tensor(2, 0) = elasticity_tensor(0, 2) = (stress_forward(0, 1) - stress_backward(0, 1)) / epsilon / 2.0;
    // std::cout << "stress forward " << stress_forward << std::endl << std::endl;
    // std::cout << "stress backward " << stress_forward << std::endl << std::endl;

    // C(1, 1), C(0, 1), C(1, 0) 
    // epsilon = 1e-3 * strain_voigt[1];
    target_strain = TV3(strain_voigt[0], strain_voigt[1] + epsilon, strain_voigt[2]);
    staticSolve();
    computeHomogenizationData(stress_forward, dummy, E1);
    target_strain = TV3(strain_voigt[0], strain_voigt[1] - epsilon, strain_voigt[2]);
    staticSolve();
    computeHomogenizationData(stress_backward, dummy, E2);
    elasticity_tensor(1, 0) = (stress_forward(0, 0) - stress_backward(0, 0)) / epsilon / 2.0;
    elasticity_tensor(1, 1) = (stress_forward(1, 1) - stress_backward(1, 1)) / epsilon / 2.0;
    elasticity_tensor(2, 1) = elasticity_tensor(1, 2) = (stress_forward(1, 0) - stress_backward(1, 0)) / epsilon / 2.0;

    // C(2, 2), C(0, 2), C(2, 0), C(1, 2), C(2, 1) 
    // epsilon = 1e-3 * strain_voigt[2];
    target_strain = TV3(strain_voigt[0], strain_voigt[1], strain_voigt[2] + epsilon);
    staticSolve();
    computeHomogenizationData(stress_forward, dummy, E1);
    target_strain = TV3(strain_voigt[0], strain_voigt[1], strain_voigt[2] -  epsilon);
    staticSolve();
    computeHomogenizationData(stress_backward, dummy, E2);

    // elasticity_tensor(2, 0) = (stress_forward(0, 0) - stress_backward(0, 0)) / epsilon / 2.0;
    // elasticity_tensor(2, 1) = (stress_forward(1, 1) - stress_backward(1, 1)) / epsilon / 2.0;
    elasticity_tensor(2, 2) = (stress_forward(0, 1) - stress_backward(0, 1)) / epsilon / 2.0;
    

}

void FEMSolver::computeHomogenizationData(TM& secondPK_stress, TM& Green_strain, T& energy_density, int pair_idx)
{
    
    TV xi = deformed.segment<2>(pbc_pairs[0][pair_idx][0] * 2);
    TV xj = deformed.segment<2>(pbc_pairs[0][pair_idx][1] * 2);

    TV xk = deformed.segment<2>(pbc_pairs[1][pair_idx][0] * 2);
    TV xl = deformed.segment<2>(pbc_pairs[1][pair_idx][1] * 2);


    TV Xi = undeformed.segment<2>(pbc_pairs[0][pair_idx][0] * 2);
    TV Xj = undeformed.segment<2>(pbc_pairs[0][pair_idx][1] * 2);
    TV Xk = undeformed.segment<2>(pbc_pairs[1][pair_idx][0] * 2);
    TV Xl = undeformed.segment<2>(pbc_pairs[1][pair_idx][1] * 2);


    TV tij = deformed.segment<2>(num_nodes*2), tkl = deformed.segment<2>(num_nodes*2 + 2);
    TV Tij = undeformed.segment<2>(num_nodes*2), Tkl = undeformed.segment<2>(num_nodes*2 + 2);

    VectorXT inner_force(num_nodes * 2 + 4);
    inner_force.setZero(); 
    addPBCForceEntries(inner_force);

    TV f0 = TV::Zero(), f1 = TV::Zero();
    // T l0 = (xj - xi).norm(), l1 = (xl - xk).norm();

    T l0 = tij.norm(), l1 = tkl.norm();

    for (auto pbc_pair : pbc_pairs[0])
    {
        f0 += inner_force.segment<2>(pbc_pair[1] * 2) / l1 / thickness;
    }
    for (auto pbc_pair : pbc_pairs[1])
    {
        f1 += inner_force.segment<2>(pbc_pair[0] * 2) / l0 / thickness;
    }

    TM R90 = TM::Zero();
    R90.row(0) = TV(0, -1);
    R90.row(1) = TV(1, 0);

    TM _X = TM::Zero(), _x = TM::Zero();
    _X.col(0) = Tij;
    _X.col(1) = Tkl;

    _x.col(0) = tij;
    _x.col(1) = tkl;

    TM F_macro = _x * _X.inverse();
    Green_strain = 0.5 * (F_macro.transpose() * F_macro - TM::Identity());
    TM cauchy_strain = 0.5 * (F_macro.transpose() + F_macro) - TM::Identity();

    // std::cout << "Cauchy strain " << 0.5 * (F_macro.transpose() + F_macro) - TM::Identity() << std::endl;
    // std::cout << F_macro << std::endl;

    TV n1 = (R90 * tij.normalized()).normalized(), 
        n0 = (R90 * tkl.normalized()).normalized();
        
    // std::cout << "f0 " << f0.transpose() << " f1 " << f1.transpose() << std::endl;
    
    TM f_bc = TM::Zero(), n_bc = TM::Zero();
    
    f_bc.col(0) = f0; f_bc.col(1) = f1;
    
    n_bc.col(0) = n0; n_bc.col(1) = n1;

    TM cauchy_stress = f_bc * n_bc.inverse();

    // std::cout << "Cauchy stress " << cauchy_stress << std::endl;

    // TV d = TV(std::cos(strain_theta), std::sin(strain_theta));
    // std::cout << "directional " << d.transpose() * cauchy_stress * d << std::endl;
    
    //https://engcourses-uofa.ca/books/introduction-to-solid-mechanics/stress/first-and-second-piola-kirchhoff-stress-tensors/
    TM F_inv = F_macro.inverse();
    secondPK_stress = F_macro.determinant() * F_inv * cauchy_stress.transpose() * F_inv.transpose();

    T volume = std::abs(tij[0] * tkl[1] - tij[1] * tkl[0]) * thickness;
    
    // std::cout << "volume " << volume << std::endl;
    
    T total_energy = computeTotalEnergy(u);
    // std::cout << "potential " << total_energy << std::endl;
    energy_density = total_energy / volume;
    // std::cout << l0 * thickness << " " << l1 * thickness << std::endl;
    // std::cout << volume << std::endl;
    // std::getchar();


    // Something to verify if the above holds for homogenous NeoHookean

    // Vector<T, 4> Green_strain_vector;
    // Green_strain_vector << Green_strain(0, 0), Green_strain(0, 1), Green_strain(1, 0), Green_strain(1, 1);
    // T energy_AD;
    // computeNHEnergyFromGreenStrain(E, nu, Green_strain_vector, energy_AD);
    // std::cout << "green strain " << std::endl;
    // std::cout << Green_strain << std::endl;
    // std::cout << "energy_density homo " << energy_density << " energy_density AD " << energy_AD << std::endl;
    // Vector<T, 4> secondPK_stress_vector;
    // computeNHEnergyFromGreenStrainGradient(E, nu, Green_strain_vector, secondPK_stress_vector);
    // TM secondPK_stress_AD;
    // secondPK_stress_AD << secondPK_stress_vector(0), secondPK_stress_vector(1), secondPK_stress_vector(2), secondPK_stress_vector(3);
    // std::cout << "second PK homo " << secondPK_stress << std::endl << "second PK AD " << secondPK_stress_AD << std::endl;
    // std::cout << "##############################" << std::endl;
    
    // std::getchar();
    // auto computedPsidE = [&](const Eigen::Matrix<double,2,2> & Green_strain, 
    //         double& energy, TM& dPsidE)
    // {
    //     T trace = Green_strain(0, 0) + Green_strain(1, 1);
    //     TM E2 = Green_strain * Green_strain;
    //     energy = 0.5 * trace * trace + E2(0, 0) + E2(1, 1);
    //     dPsidE = trace * TM::Identity() + 2.0 * Green_strain; 
    // };

    // auto difftest = [&]()
    // {
    //     T eps = 1e-6;
    //     TM rest;
    //     T E0, E1;
    //     computedPsidE(Green_strain, energy_density, rest);
    //     Green_strain(1, 0) += eps;
    //     computedPsidE(Green_strain, E0, secondPK_stress);
    //     Green_strain(1, 0) -= 2.0 * eps;
    //     computedPsidE(Green_strain, E1, secondPK_stress);
    //     Green_strain(1, 0) += eps;
    //     std::cout << "fd " << (E0 - E1) / 2.0/ eps << " analytic " << rest(1, 0) << std::endl;
    // };
    // computedPsidE(Green_strain, energy_density, secondPK_stress);
    // std::cout << "Green Strain" << std::endl;
    // std::cout << Green_strain << std::endl;
    // std::cout << "second PK Stress" << std::endl;
    // std::cout << secondPK_stress << std::endl;
    // difftest();
    // std::exit(0);
   
}

void FEMSolver::computeHomogenizedStressStrain(TM& sigma, TM& Cauchy_strain, TM& Green_strain)
{
    TV xi = deformed.segment<2>(pbc_pairs[0][0][0] * 2);
    TV xj = deformed.segment<2>(pbc_pairs[0][0][1] * 2);

    TV xk = deformed.segment<2>(pbc_pairs[1][0][0] * 2);
    TV xl = deformed.segment<2>(pbc_pairs[1][0][1] * 2);


    TV Xi = undeformed.segment<2>(pbc_pairs[0][0][0] * 2);
    TV Xj = undeformed.segment<2>(pbc_pairs[0][0][1] * 2);
    TV Xk = undeformed.segment<2>(pbc_pairs[1][0][0] * 2);
    TV Xl = undeformed.segment<2>(pbc_pairs[1][0][1] * 2);

    VectorXT inner_force(num_nodes * 2);
    inner_force.setZero(); 
    addPBCForceEntries(inner_force);
    
    // computeResidual(u, inner_force);
    TV f0 = TV::Zero(), f1 = TV::Zero();
    T l0 = (xj - xi).norm(), l1 = (xl - xk).norm();
    for (auto pbc_pair : pbc_pairs[0])
    {
        f0 += inner_force.segment<2>(pbc_pair[0] * 2) / l1;
    }
    for (auto pbc_pair : pbc_pairs[1])
    {
        f1 += inner_force.segment<2>(pbc_pair[1] * 2) / l0;
    }
    
    TM R90 = TM::Zero();
    R90.row(0) = TV(0, -1);
    R90.row(1) = TV(1, 0);

    TM _X = TM::Zero(), _x = TM::Zero();
    _X.col(0) = (Xi - Xj).template segment<2>(0);
    _X.col(1) = (Xk - Xl).template segment<2>(0);

    _x.col(0) = (xi - xj).template segment<2>(0);
    _x.col(1) = (xk - xl).template segment<2>(0);

    TM F_macro = _x * _X.inverse();
    Cauchy_strain = 0.5 * (F_macro.transpose() + F_macro) - TM::Identity();
    Green_strain = 0.5 * (F_macro.transpose() * F_macro - TM::Identity());
    

    TV n1 = (R90 * (xj - xi)).normalized(), 
        n0 = (R90 * (xl - xk)).normalized();

    TM f_bc = TM::Zero(), n_bc = TM::Zero();
    
    f_bc.col(0) = f0; f_bc.col(1) = f1;
    
    n_bc.col(0) = n0; n_bc.col(1) = n1;

    sigma = f_bc * n_bc.inverse();
}

void FEMSolver::computeHomogenizedStressStrain(TM& sigma, TM& epsilon)
{
    
    TV xi = deformed.segment<2>(pbc_pairs[0][0][0] * 2);
    TV xj = deformed.segment<2>(pbc_pairs[0][0][1] * 2);

    TV xk = deformed.segment<2>(pbc_pairs[1][0][0] * 2);
    TV xl = deformed.segment<2>(pbc_pairs[1][0][1] * 2);


    TV Xi = undeformed.segment<2>(pbc_pairs[0][0][0] * 2);
    TV Xj = undeformed.segment<2>(pbc_pairs[0][0][1] * 2);
    TV Xk = undeformed.segment<2>(pbc_pairs[1][0][0] * 2);
    TV Xl = undeformed.segment<2>(pbc_pairs[1][0][1] * 2);

    VectorXT inner_force(num_nodes * 2);
    inner_force.setZero(); 
    addPBCForceEntries(inner_force);
    
    // computeResidual(u, inner_force);
    TV f0 = TV::Zero(), f1 = TV::Zero();
    T l0 = (xj - xi).norm(), l1 = (xl - xk).norm();
    for (auto pbc_pair : pbc_pairs[0])
    {
        f0 += inner_force.segment<2>(pbc_pair[0] * 2) / l1;
    }
    for (auto pbc_pair : pbc_pairs[1])
    {
        f1 += inner_force.segment<2>(pbc_pair[1] * 2) / l0;
    }
    
    TM R90 = TM::Zero();
    R90.row(0) = TV(0, -1);
    R90.row(1) = TV(1, 0);

    TM _X = TM::Zero(), _x = TM::Zero();
    _X.col(0) = (Xi - Xj).template segment<2>(0);
    _X.col(1) = (Xk - Xl).template segment<2>(0);

    _x.col(0) = (xi - xj).template segment<2>(0);
    _x.col(1) = (xk - xl).template segment<2>(0);

    TM F_macro = _x * _X.inverse();
    // std::cout << "deformation graident" << std::endl;
    // std::cout << F_macro << std::endl;
    
    TM cauchy_strain = 0.5 * (F_macro.transpose() + F_macro) - TM::Identity();
    // std::cout << "cauchy_strain" << std::endl;
    // std::cout << cauchy_strain << std::endl;
    epsilon =  0.5 * (F_macro.transpose() + F_macro) - TM::Identity();

    // Matrix<T, 4, 2> x, X;
    // getMarcoBoundaryData(x, X);

    // T eps_xx = (xl[0] - xk[0]) / (Xl[0] - Xk[0]) - 1.0;
    // T eps_yy = (xj[1] - xi[1]) / (Xj[1] - Xi[1]) - 1.0;
    // T vdu = (xl[1] - xk[1] - (Xl[1] - Xk[1])) / ((Xl[0] - Xk[0]));
    // T udv = (xj[0] - xi[0] - (Xj[0] - Xi[0])) / (Xj[1] - Xi[1]);
    // T eps_xy = 0.5 * (vdu + udv);
    // Matrix<T, 3, 2> dNdb;
    //     dNdb << -1.0, -1.0, 
    //         1.0, 0.0,
    //         0.0, 1.0;
    // EleNodes x_undeformed, x_deformed;
    // x_undeformed << 0,0,1,0,0,1;
    // x_deformed << Xl[0], Xl[1], Xk[0], Xk[1], Xj[0], Xj[1];
    // // std::cout << x_deformed << std::endl;
    // // std::cout << Xl.transpose() << " " << Xk.transpose() << " " << Xj.transpose() << std::endl;
    // TM dXdb = x_undeformed.transpose() * dNdb;
    // TM dxdb = x_deformed.transpose() * dNdb;
    // TM F = dxdb * dXdb.inverse();
    
    // // xi = F.inverse() * xi; xj = F.inverse() * xj;xk = F.inverse() * xk;xl = F.inverse() * xl;
    // // Xi = F.inverse() * Xi; Xj = F.inverse() * Xj;Xk = F.inverse() * Xk;Xl = F.inverse() * Xl;
    // std::ofstream out("test_mesh.obj");
    // // out << "v " << xi.transpose() << " 0" << std::endl;
    // // out << "v " << xj.transpose() << " 0" << std::endl;
    // // out << "v " << xk.transpose() << " 0" << std::endl;
    // out << "v " << Xl.transpose() << " 0" << std::endl;
    // out << "v " << Xk.transpose() << " 0" << std::endl;
    // out << "v " << Xj.transpose() << " 0" << std::endl;

    // out << "v " << xl.transpose() << " 0" << std::endl;
    // out << "v " << xk.transpose() << " 0" << std::endl;
    // out << "v " << xj.transpose() << " 0" << std::endl;
    // // out << "v " << Xi.transpose() << " 0" << std::endl;
    // out << "f 1 2 3" << std::endl;
    // // out << "f 1 2 4" << std::endl;
    // out << "f 4 5 6" << std::endl;
    // out.close();
    // T eps_xx = (xl[0] - xk[0]) / (Xl[0] - Xk[0]) - 1.0;
    // T eps_yy = (xj[1] - xi[1]) / (Xj[1] - Xi[1]) - 1.0;
    // T v_du = (xl[1] - xk[1]) / ((Xl[0] - Xk[0]));
    // T u_dv = (xj[0] - xi[0]) / (Xj[1] - Xi[1]);
    // T eps_xy = 0.5 * (u_dv + v_du);
    // // T eps_xy = (xl[1] - xk[1]) / ((Xl[0] - Xk[0]));
    // // T eps_yx = (xj[0] - xi[0]) / (Xj[1] - Xi[1]);
    
    // epsilon << eps_xx, eps_xy, eps_xy, eps_yy;
    

    // std::cout << strain_macro << std::endl;
    
    TV n1 = (R90 * (xj - xi)).normalized(), 
        n0 = (R90 * (xl - xk)).normalized();

    // std::ofstream out("debug.obj");
    // int cnt = 0;
    // for (auto pbc_pair : pbc_pairs[1])
    // {
        
    //     int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
    //     TV Xi = undeformed.segment<2>(idx0 * 2);
    //     TV Xj = undeformed.segment<2>(idx1 * 2);
    //     out << "v " << Xj.transpose() << " 0" << std::endl;
    //     cnt ++;
    // }
    // for (int i = 0; i < cnt - 1;i++)
    //     out << "l " << i + 1 << " " << i + 2 << std::endl;
    // out << "v " << xi_ref.transpose() << " 0" << std::endl;
    // out << "v " << xj_ref.transpose() << " 0" << std::endl;
    // out << "v " << xk_ref.transpose() << " 0" << std::endl;
    // out << "v " << xl_ref.transpose() << " 0" << std::endl;
    // out << "v " << (xi_ref + n1).transpose() << " 0" << std::endl;
    // out << "v " << (xk_ref + n0).transpose() << " 0" << std::endl;
    // out << "l 1 2" << std::endl;
    // out << "l 1 5" << std::endl;
    // out << "l 3 4" << std::endl;
    // out << "l 3 6" << std::endl; 
    // out.close();
    TM f_bc = TM::Zero(), n_bc = TM::Zero();
    
    f_bc.col(0) = f0; f_bc.col(1) = f1;
    
    n_bc.col(0) = n0; n_bc.col(1) = n1;

    

    sigma = f_bc * n_bc.inverse();

    // TV strain_dir = TV(std::cos(strain_theta), std::sin(strain_theta));
    // std::cout << sigma * (R90 *strain_dir) << std::endl;
    // std::cout << std::endl;
    // std::cout << "stress macro" << std::endl;
    // std::cout << sigma << std::endl;
}