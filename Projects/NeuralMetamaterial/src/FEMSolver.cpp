#include "../include/FEMSolver.h"
#include "../include/Timer.h"
#include <Eigen/CholmodSupport>

#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>


T FEMSolver::computeTotalEnergy(const VectorXT& _u)
{
    T total_energy = 0.0;

    VectorXT projected = _u;

    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    if (use_reduced)
        projected = jac_full2reduced * projected;

    deformed = undeformed + projected;

    T e_NH = 0.0;
    addElastsicPotential(e_NH);
    total_energy += e_NH;

    if (add_pbc)
    {
        T e_pbc = 0.0;
        addPBCEnergy(e_pbc);
        total_energy += e_pbc;
    }
    
    if (use_ipc)
    {
        T ipc = 0.0;
        addIPCEnergy(ipc);
        total_energy += ipc;
    }

    if (unilateral_qubic)
    {
        T uni_qubic = 0.0;
        addUnilateralQubicPenaltyEnergy(penalty_weight, uni_qubic);
        total_energy += uni_qubic;
    }

    if (penalty_pairs.size())
    {
        T penalty = 0.0;
        addBCPenaltyEnergy(penalty_weight, penalty);
        total_energy += penalty;
    }
    

    return total_energy;
}


T FEMSolver::computeResidual(const VectorXT& _u, VectorXT& residual)
{
    
    VectorXT projected = _u;

    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    
    if (use_reduced)
        projected = jac_full2reduced * projected;
    
    deformed = undeformed + projected;

    residual.resize(deformed.rows());
    residual.setZero();
    
    VectorXT residual_backup = residual;

    addElasticForceEntries(residual);

    if (verbose)
    {
        std::cout << "elastic force " << (residual - residual_backup).norm() << std::endl;
        residual_backup = residual;
    }

    if (add_pbc)
    {
        addPBCForceEntries(residual);
        if (verbose)
        {
            std::cout << "pbc force " << (residual - residual_backup).norm() << std::endl;
            residual_backup = residual;
        }
    }

    if (use_ipc)
    {
        addIPCForceEntries(residual);
        if (verbose)
        {
            std::cout << "contact force " << (residual - residual_backup).norm() << std::endl;
            residual_backup = residual;
        }
    }

    if (unilateral_qubic)
    {
        addUnilateralQubicPenaltyForceEntries(penalty_weight, residual);
        if (verbose)
        {
            std::cout << "qubic penalty force " << (residual - residual_backup).norm() << std::endl;
            residual_backup = residual;
        }
    }

    if (penalty_pairs.size())
    {
        addBCPenaltyForceEntries(penalty_weight, residual);
        if (verbose)
        {
            std::cout << "penalty force " << (residual - residual_backup).norm() << std::endl;
            residual_backup = residual;
        }
    }
    
    residual = jac_full2reduced.transpose() * residual;
    
    if (!run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            residual[offset] = 0;
        });

    return residual.norm();
}

void FEMSolver::reset()
{
    deformed = undeformed;
    u.setZero();
    if (use_ipc)
    {
        computeIPCRestData();
        // ipc_vertices.resize(num_nodes, 2);
        // for (int i = 0; i < num_nodes; i++)
        //     ipc_vertices.row(i) = undeformed.segment<2>(i * 2);
    }
    
}

void FEMSolver::checkHessianPD(bool save_txt)
{
    bool backup = project_block_PD;
    project_block_PD = false;
    int nmodes = 10;
    int n_dof_sim = deformed.rows();
    StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
    buildSystemMatrix(u, d2edx2);
    project_block_PD = backup;
    bool use_Spectra = true;

    // Eigen::PardisoLLT<StiffnessMatrix, Eigen::Lower> solver;
    Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> solver;
    solver.analyzePattern(d2edx2); 
    // std::cout << "analyzePattern" << std::endl;
    solver.factorize(d2edx2);
    // std::cout << "factorize" << std::endl;
    bool indefinite = false;
    if (solver.info() == Eigen::NumericalIssue)
    {
        std::cout << "!!!indefinite matrix!!!" << std::endl;
        indefinite = true;
        
    }
    else
    {
        // std::cout << "indefinite" << std::endl;
    }
    
    if (use_Spectra)
    {
        
        Spectra::SparseSymShiftSolve<T, Eigen::Lower> op(d2edx2);
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
            if (save_txt)
            {
                std::ofstream out("eigen_vectors.txt");
                out << eigen_vectors.rows() << " " << eigen_vectors.cols() << std::endl;
                for (int i = 0; i < eigen_vectors.cols(); i++)
                    out << eigen_values[eigen_vectors.cols() - 1 - i] << " ";
                out << std::endl;
                for (int i = 0; i < eigen_vectors.rows(); i++)
                {
                    // for (int j = 0; j < eigen_vectors.cols(); j++)
                    for (int j = eigen_vectors.cols() - 1; j >-1 ; j--)
                        out << eigen_vectors(i, j) << " ";
                    out << std::endl;
                }       
                out << std::endl;
                out.close();
            }
        }
        else
        {
            std::cout << "Eigen decomposition failed" << std::endl;
        }
    }
}

void FEMSolver::builddfdX(const VectorXT& _u, StiffnessMatrix& dfdX)
{
    dfdX.resize(num_nodes * 2, num_nodes * 2);
    VectorXT projected = _u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    deformed = undeformed + projected;
    
    std::vector<Entry> entries;

    addElasticdfdXEntries(entries);

    dfdX.setFromTriplets(entries.begin(), entries.end());

    if (!run_diff_test)
        projectDirichletDoFMatrix(dfdX, dirichlet_data);
    
    dfdX.makeCompressed();
}

void FEMSolver::buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K)
{
    VectorXT projected = _u;

    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    if (use_reduced)
        projected = jac_full2reduced * projected;


    deformed = undeformed + projected;
    
    K.resize(deformed.rows(), deformed.rows());
    std::vector<Entry> entries;
    
    addElasticHessianEntries(entries, project_block_PD);
    if (add_pbc && add_pbc_strain)
        addPBCHessianEntries(entries, project_block_PD);
    if (use_ipc)
        addIPCHessianEntries(entries, project_block_PD);
    
    // if (unilateral_qubic)
    //     addUnilateralQubicPenaltyHessianEntries(penalty_weight, entries);

    if (penalty_pairs.size())
        addBCPenaltyHessianEntries(penalty_weight, entries);
    
    // K.reserve(entries.size());
    K.setFromTriplets(entries.begin(), entries.end());
    
    if (use_reduced)
        K = jac_full2reduced.transpose() * K * jac_full2reduced;
    

    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    
    // std::cout << "K " << K.rows() << " " << K.cols() << std::endl;
    K.makeCompressed();

}

void FEMSolver::projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data)
{
    for (auto iter : data)
    {
        A.row(iter.first) *= 0.0;
        A.col(iter.first) *= 0.0;
        A.coeffRef(iter.first, iter.first) = 1.0;
    }

}

bool FEMSolver::linearSolve(StiffnessMatrix& K, 
    VectorXT& residual, VectorXT& du)
{
    // std::cout << "Linear Solver" << std::endl;
    Timer t(true);
    Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> solver;
    // Eigen::PardisoLLT<StiffnessMatrix, Eigen::Lower> solver;
    T alpha = 1e-6;
    StiffnessMatrix H(K.rows(), K.cols());
    H.setIdentity(); H.diagonal().array() = 1e-10;
    K += H;
    solver.analyzePattern(K);
    // T time_analyze = t.elapsed_sec();
    // std::cout << "\t analyzePattern takes " << time_analyze << "s" << std::endl;
    
    int indefinite_count_reg_cnt = 0, invalid_search_dir_cnt = 0, invalid_residual_cnt = 0;

    for (int i = 0; i < 50; i++)
    {
        solver.factorize(K);
        // std::cout << "factorize" << std::endl;
        if (solver.info() == Eigen::NumericalIssue)
        {
            K.diagonal().array() += alpha;
            alpha *= 10;
            indefinite_count_reg_cnt++;
            continue;
        }

        du = solver.solve(residual);
        
        T dot_dx_g = du.normalized().dot(residual.normalized());

        int num_negative_eigen_values = 0;
        int num_zero_eigen_value = 0;

        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-6;
        if (!search_dir_correct_sign)
        {   
            invalid_search_dir_cnt++;
        }
        
        // bool solve_success = true;
        // bool solve_success = (K * du - residual).norm() / residual.norm() < 1e-6;
        bool solve_success = du.norm() < 1e3;
        
        if (!solve_success)
            invalid_residual_cnt++;
        // std::cout << "PD: " << positive_definte << " direction " 
        //     << search_dir_correct_sign << " solve " << solve_success << std::endl;

        if (positive_definte && search_dir_correct_sign && solve_success)
        {
            t.stop();
            if (verbose)
            {
                std::cout << "\t===== Linear Solve ===== " << std::endl;
                std::cout << "\tnnz: " << K.nonZeros() << std::endl;
                std::cout << "\t takes " << t.elapsed_sec() << "s" << std::endl;
                std::cout << "\t# regularization step " << i 
                    << " indefinite " << indefinite_count_reg_cnt 
                    << " invalid search dir " << invalid_search_dir_cnt
                    << " invalid solve " << invalid_residual_cnt << std::endl;
                std::cout << "\tdot(search, -gradient) " << dot_dx_g << std::endl;
                // std::cout << (K.selfadjointView<Eigen::Lower>() * du + UV * UV.transpose()*du - residual).norm() << std::endl;
                std::cout << "\t======================== " << std::endl;
            }
            return true;
        }
        else
        {
            K.diagonal().array() += alpha;
            alpha *= 10;
        }
    }
    return false;
}

T FEMSolver::lineSearchNewton(VectorXT& _u, VectorXT& residual)
{
    VectorXT du = residual;
    du.setZero();

    StiffnessMatrix K(residual.rows(), residual.rows());
    Timer ti(true);
    buildSystemMatrix(_u, K);
    // std::cout << "\tbuild system takes " <<  ti.elapsed_sec() << std::endl;
    // ti.restart();
    bool success = linearSolve(K, residual, du);
    // std::cout << "\tlinearSolve takes " <<  ti.elapsed_sec() << std::endl;
    // ti.restart();
    if (!success)
        return 1e16;
    T norm = du.norm();
    if (verbose)
        std::cout << "\t|du| " <<  du.norm() << std::endl;
    
    T alpha = computeInversionFreeStepsize(_u, du);
    if (verbose)
    {
        std::cout << "\t** step size **" << std::endl;
        std::cout << "\tafter tet inv step size: " << alpha << std::endl;
    }
    if (use_ipc)
    {
        T ipc_step_size = computeCollisionFreeStepsize(_u, du);
        alpha = std::min(alpha, ipc_step_size);
        if (verbose)
            std::cout << "\tafter ipc step size: " << alpha << std::endl;
    }
    if (verbose)
        std::cout << "\t**       **" << std::endl;

    // std::cout << "\tcompute step size takes " <<  ti.elapsed_sec() << std::endl;
    // ti.restart();

    T E0 = computeTotalEnergy(_u);
    int cnt = 0;
    while (true)
    {
        VectorXT u_ls = _u + alpha * du;
        T E1 = computeTotalEnergy(u_ls);
        if (E1 - E0 < 0 || cnt > 12)
        {
            // if (cnt > 15)
            //     std::cout << "cnt > 15" << std::endl;
            _u = u_ls;
            break;
        }
        alpha *= 0.5;
        cnt += 1;
    }
    if (verbose)
        std::cout << "#ls " << cnt << " alpha = " << alpha << std::endl;
    return norm;
}


bool FEMSolver::staticSolveStep(int step)
{
    Timer ti(true);
    if (step == 0)
    {
        // iterateDirichletDoF([&](int offset, T target)
        // {
        //     f[offset] = 0;
        // });
        // if (use_ipc) 
            // computeIPCRestData();
    }

    VectorXT residual(reduced_dof.rows());
    residual.setZero();

    if (use_ipc)
    {
        updateBarrierInfo(step == 0);
        // std::cout << "ipc barrier stiffness " << barrier_weight << std::endl;
        // std::getchar();
        updateIPCVertices(u);
    }

    T residual_norm = computeResidual(u, residual);
    // std::cout << "\tcomputeResidual takes " <<  ti.elapsed_sec() << std::endl;
    // ti.restart();
    // std::cout << residual.rows() << " " << u.rows() << " " << reduced_dof.rows() << std::endl;
    std::cout << "[NEWTON] iter " << step << "/" << max_newton_iter << ": residual_norm " << residual_norm << " tol: " << newton_tol << std::endl;

    T dq_norm = lineSearchNewton(u, residual);
    // std::cout << "\tlineSearchNewton takes " <<  ti.elapsed_sec() << std::endl;
    // ti.restart();

    if (residual_norm < newton_tol || step == max_newton_iter || dq_norm < 1e-12)
    {
        if (add_pbc_strain)
        {
            
            TM secondPK_stress, Green_strain;
            T psi;
            std::cout << std::setprecision(10) << std::endl;
            computeHomogenizationData(secondPK_stress, Green_strain, psi);
            std::cout << "strain " << Green_strain << std::endl << std::endl;
            std::cout << "stress " << secondPK_stress << std::endl << std::endl;
            // std::cout << std::setprecision(8) << "energy density " << psi << std::endl << std::endl;
            std::cout << "energy density " << psi << std::endl << std::endl;
            
        }
        return true;
    }
    

    
    // iterateDirichletDoF([&](int offset, T target)
    // {
    //     u[offset] = target;
    // });
    // deformed = undeformed + u;

    if(step == max_newton_iter || dq_norm > 1e10 || dq_norm < 1e-12)
    {
        // std::cout << "|u| " << dq_norm << std::endl;
        return true;
    }
    
    return false;

}

bool FEMSolver::staticSolve()
{
    int cnt = 0;
    T residual_norm = 1e10, du_norm = 1e10;

    iterateDirichletDoF([&](int offset, T target)
    {
        f[offset] = 0;
    });
    
    while (true)
    {
        
        VectorXT residual(reduced_dof.rows());
        residual.setZero();

        residual_norm = computeResidual(u, residual);
        if (use_ipc)
        {
            updateBarrierInfo(cnt == 0);
            updateIPCVertices(u);
            if (verbose)
                std::cout << "ipc barrier stiffness " << barrier_weight << std::endl;
        }
        
        if (verbose)
            std::cout << "iter " << cnt << "/" << max_newton_iter 
            << ": residual_norm " << residual.norm() << " tol: " << newton_tol << std::endl;
        
        if (residual_norm < newton_tol)
            break;
        
        du_norm = lineSearchNewton(u, residual);

        if(cnt == max_newton_iter || du_norm > 1e10 || du_norm < 1e-12)
            break;
        cnt++;
    }

    iterateDirichletDoF([&](int offset, T target)
    {
        u[offset] = target;
    });
    if (use_reduced)
        deformed = undeformed + jac_full2reduced * u;
    else
        deformed = undeformed + u;
    // if (verbose)
    // {
    //     T E_bc = 0.0, E_pbc = 0.0;
    //     addBCPenaltyEnergy(penalty_weight, E_bc);
    //     addPBCEnergy(E_pbc);
    //     std::cout << "penalty BC " << E_bc << " PBC " << E_pbc << std::endl;
    // }
    

    std::cout << "# of newton solve: " << cnt << " exited with |g|: " 
        << residual_norm << "|ddu|: " << du_norm  << std::endl;
    
    // VectorXT contact_force(reduced_dof.rows());
    // contact_force.setZero();
    // addIPCForceEntries(contact_force);
    // std::cout << "Contact force " << contact_force.norm() << std::endl;
    
    if ((cnt == max_newton_iter && residual_norm > 5e-6) || du_norm > 1e10 || residual_norm > 1)
        return false;
    return true;
    
}

void FEMSolver::constructReducedJacobian()
{
    int full_dof = deformed.rows() + 4;
                    
    bool bottom_left_corner_match = pbc_pairs[0][0][0] == pbc_pairs[1][0][0];
    bool top_left_corner_match = pbc_pairs[0][pbc_pairs[0].size()-1][0] == pbc_pairs[1][0][1];
    std::vector<int> duplicated_entries;
    if (bottom_left_corner_match && top_left_corner_match)
    {
        // std::cout << " match " << std::endl;
        for (int i = 0; i < (int)pbc_pairs[0].size(); i++)
            duplicated_entries.push_back(pbc_pairs[0][i][1]);
        for (int i = 1; i < (int)pbc_pairs[1].size()-1; i++)
            duplicated_entries.push_back(pbc_pairs[1][i][1]);
    }
    std::vector<bool> is_boundary_vtx(num_nodes, false);
    int check_cnt = 0;
    for (int dir = 0; dir < 2; dir++)
        for (IV& pbc_pair : pbc_pairs[dir])
        {
            int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
            is_boundary_vtx[idx0] = true;
            is_boundary_vtx[idx1] = true;
        }
    
    int n_reduced_dof = full_dof - duplicated_entries.size() * 2;
    
    reduced_dof.resize(n_reduced_dof);
    int translation_dof_start = n_reduced_dof - 4;
    jac_full2reduced.resize(full_dof, n_reduced_dof);
    

    deformed.conservativeResize(num_nodes * 2 + 4);
    undeformed.conservativeResize(num_nodes * 2 + 4);
    u.conservativeResize(num_nodes * 2 + 4);
    f.conservativeResize(num_nodes * 2 + 4);
    f.segment<4>(num_nodes * 2).setZero();

    deformed.segment<2>(full_dof-4) = 
        deformed.segment<2>(pbc_pairs[0][0][1] * 2) - 
        deformed.segment<2>(pbc_pairs[0][0][0] * 2);
    
    deformed.segment<2>(full_dof-2) = 
        deformed.segment<2>(pbc_pairs[1][0][1] * 2) - 
        deformed.segment<2>(pbc_pairs[1][0][0] * 2);
    
    undeformed = deformed;

    TV tij = deformed.segment<2>(full_dof-4);
    TV tkl = deformed.segment<2>(full_dof-2);
    
    std::vector<Entry> jac_entries;
    int valid_cnt = 0;
    for (int i = 0; i < num_nodes; i++)
    {
        if (!is_boundary_vtx[i])
        {
            reduced_dof.segment<2>(valid_cnt * 2) = deformed.segment<2>(i*2);
            jac_entries.push_back(Entry(i * 2 + 0, valid_cnt * 2 + 0, 1.0));
            jac_entries.push_back(Entry(i * 2 + 1, valid_cnt * 2 + 1, 1.0));
            valid_cnt++;
        }
    }

    for (int i = 0; i < pbc_pairs[0].size(); i++)
    {
        reduced_dof.segment<2>(valid_cnt * 2) = 
            deformed.segment<2>(pbc_pairs[0][i][0] * 2);
        
        if (i == 0)
        {
            dirichlet_data[valid_cnt * 2 + 0] = 0;
            dirichlet_data[valid_cnt * 2 + 1] = 0;
        }
        // std::cout << reduced_dof.segment<2>(valid_cnt * 2).transpose() << std::endl;

        jac_entries.push_back(Entry(pbc_pairs[0][i][0] * 2 + 0, valid_cnt * 2 + 0, 1.0));
        jac_entries.push_back(Entry(pbc_pairs[0][i][0] * 2 + 1, valid_cnt * 2 + 1, 1.0));

        // xi_x + Tij_x
        jac_entries.push_back(Entry(pbc_pairs[0][i][1]*2+0, valid_cnt*2+0, 1.0));
        jac_entries.push_back(Entry(pbc_pairs[0][i][1]*2+0, translation_dof_start, 1.0));

        // xi_y + Tij_y
        jac_entries.push_back(Entry(pbc_pairs[0][i][1]*2+1, valid_cnt*2+1, 1.0));
        jac_entries.push_back(Entry(pbc_pairs[0][i][1]*2+1, translation_dof_start+1, 1.0));

        valid_cnt++;
    }
    for (int i = 1; i < pbc_pairs[1].size()-1; i++)
    {
        reduced_dof.segment<2>(valid_cnt * 2) = 
            deformed.segment<2>(pbc_pairs[1][i][0] * 2);
        // std::cout << reduced_dof.segment<2>(valid_cnt * 2).transpose() << std::endl;

        jac_entries.push_back(Entry(pbc_pairs[1][i][0]*2+0, valid_cnt*2+0, 1.0));
        jac_entries.push_back(Entry(pbc_pairs[1][i][0]*2+1, valid_cnt*2+1, 1.0));

        jac_entries.push_back(Entry(pbc_pairs[1][i][1]*2+0, valid_cnt*2+0, 1.0));
        jac_entries.push_back(Entry(pbc_pairs[1][i][1] * 2 + 0, translation_dof_start+2, 1.0));

        jac_entries.push_back(Entry(pbc_pairs[1][i][1]*2+1, valid_cnt*2+1, 1.0));
        jac_entries.push_back(Entry(pbc_pairs[1][i][1] * 2 + 1, translation_dof_start+3, 1.0));


        valid_cnt++;
    }
    
    reduced_dof.segment<2>(valid_cnt * 2) = tij;
    valid_cnt ++;
    reduced_dof.segment<2>(valid_cnt * 2) = tkl;
    valid_cnt ++;
    for (int i = 0; i < 4; i++)
        jac_entries.push_back(Entry(full_dof - (4-i), n_reduced_dof - (4-i), 1.0));

    jac_full2reduced.setFromTriplets(jac_entries.begin(), jac_entries.end());

    VectorXT prediction = jac_full2reduced * reduced_dof;
    T error = (prediction - deformed).norm();
    if (std::abs(error) > 1e-8)
    {
        deformed = prediction;
        std::cout << " line 720 dof matching error " << error << std::endl;
        std::cout << __FILE__ << std::endl;
        // std::exit(0);
    }
   
    u.resize(reduced_dof.rows()); u.setZero();
}