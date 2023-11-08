#include "../include/FEMSolver.h"

void FEMSolver::checkTotalGradient(bool perturb)
{
    run_diff_test = true;
    // pbc_strain_w = 0;
    // pbc_w = 0;

    VectorXT du(num_nodes * 2);
    du.setRandom();
    du *= 1.0 / du.norm();
    du *= 0.001;
    if (perturb)
        u += du;

    std::cout << "======================== CHECK GRADIENT ========================" << std::endl;
    int n_dof = reduced_dof.rows();
    T epsilon = 1e-6;
    VectorXT gradient(n_dof);
    gradient.setZero();

    computeResidual(u, gradient);

    // std::cout << gradient.transpose() << std::endl;
    
    VectorXT gradient_FD(n_dof);
    gradient_FD.setZero();

    int cnt = 0;
    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        u(dof_i) += epsilon;
        // std::cout << W * dq << std::endl;
        T E0 = computeTotalEnergy(u);
        
        u(dof_i) -= 2.0 * epsilon;
        T E1 = computeTotalEnergy(u);
        u(dof_i) += epsilon;
        // std::cout << "E1 " << E1 << " E0 " << E0 << std::endl;
        gradient_FD(dof_i) = (E1 - E0) / (2.0 *epsilon);
        if( gradient_FD(dof_i) == 0 && gradient(dof_i) == 0)
            continue;
        if (std::abs( gradient_FD(dof_i) - gradient(dof_i)) < 1e-3 * std::abs(gradient(dof_i)))
            continue;
        std::cout << " dof " << dof_i << " " << gradient_FD(dof_i) << " " << gradient(dof_i) << std::endl;
        std::getchar();
        cnt++;   
    }
    run_diff_test = false;
}

void FEMSolver::checkTotalGradientScale(bool perturb)
{
    project_block_PD = false;
    run_diff_test = true;
    
    std::cout << "======================== CHECK GRADIENT 2nd Scale ========================" << std::endl;
    
    VectorXT du(reduced_dof.rows());
    du.setRandom();
    du *= 1.0 / du.norm();
    du *= 0.001;
    if (perturb)
        u += du;
    
    int n_dof = reduced_dof.rows();

    VectorXT gradient(n_dof);
    gradient.setZero();
    computeResidual(u, gradient);
    
    gradient *= -1;
    T E0 = computeTotalEnergy(u);
    VectorXT dx(n_dof);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    dx *= 0.01;
    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {
        T E1 = computeTotalEnergy(u + dx);
        T dE = E1 - E0;
        
        dE -= gradient.dot(dx);
        // std::cout << "dE " << dE << std::endl;
        if (i > 0)
        {
            std::cout << (previous/dE) << std::endl;
        }
        previous = dE;
        dx *= 0.5;
    }
    run_diff_test = false;
}

void FEMSolver::checkTotalHessian(bool perturb)
{
    project_block_PD = false;
    run_diff_test = true;
    T epsilon = 1e-5;
    unilateral_qubic = false;
    int n_dof = num_nodes * 3;
    // pbc_w = 0;
    // use_ipc=false;
    // E=0;
    // penalty_weight =0;
    // pbc_strain_w = 0;

    VectorXT du(num_nodes * 3);
    du.setRandom();
    du *= 1.0 / du.norm();
    du *= 0.001;
    if (perturb)
        u += du;
    VectorXT rest = undeformed;
    
    deformed = undeformed + u;
    StiffnessMatrix A(n_dof, n_dof);
    buildSystemMatrix(u, A);
    
    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        // std::cout << dof_i << std::endl;
        u(dof_i) += epsilon;
        VectorXT g0(n_dof), g1(n_dof);
        g0.setZero(); g1.setZero();
        
        computeResidual(u, g0); 
        
        u(dof_i) -= 2.0 * epsilon;
        
        computeResidual(u, g1); 
        
        u(dof_i) += epsilon;
        VectorXT row_FD = (g1 - g0) / (2.0 * epsilon);

        for(int i = 0; i < n_dof; i++)
        {
            
            if(A.coeff(i, dof_i) == 0 && row_FD(i) == 0)
                continue;
            if (std::abs( A.coeff(i, dof_i) - row_FD(i)) < 1e-3 * std::abs(row_FD(i)))
                continue;
            // std::cout << "node i: "  << std::floor(dof_i / T(dof)) << " dof " << dof_i%dof 
            //     << " node j: " << std::floor(i / T(dof)) << " dof " << i%dof 
            //     << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::cout << "H(" << i << ", " << dof_i << ") " << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::getchar();
        }
    }
    std::cout << "Hessian Diff Test Passed" << std::endl;
    run_diff_test = false;
}

void FEMSolver::checkdfdX(bool perturb)
{
    std::cout << "======================== CHECK HESSIAN ========================" << std::endl;
    run_diff_test = true;
    T epsilon = 1e-5;
    int n_dof = num_nodes * 3;
    
    VectorXT du(num_nodes * 3);
    du.setRandom();
    du *= 1.0 / du.norm();
    du *= 0.001;
    if (perturb)
        u += du;
    VectorXT rest = undeformed;
    
    deformed = undeformed + u;
    StiffnessMatrix A(n_dof, n_dof);
    builddfdX(u, A);
    
    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        // std::cout << dof_i << std::endl;
        undeformed(dof_i) += epsilon;
        VectorXT g0(n_dof), g1(n_dof);
        g0.setZero(); g1.setZero();
        
        addElasticForceEntries(g0);

        undeformed(dof_i) -= 2.0 * epsilon;
        
        addElasticForceEntries(g1);
        undeformed(dof_i) += epsilon;
        VectorXT row_FD = (g1 - g0) / (2.0 * epsilon);

        for(int i = 0; i < n_dof; i++)
        {
            
            if(A.coeff(i, dof_i) == 0 && row_FD(i) == 0)
                continue;
            if (std::abs( A.coeff(i, dof_i) - row_FD(i)) < 1e-3 * std::abs(row_FD(i)))
                continue;
            // std::cout << "node i: "  << std::floor(dof_i / T(dof)) << " dof " << dof_i%dof 
            //     << " node j: " << std::floor(i / T(dof)) << " dof " << i%dof 
            //     << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::cout << "H(" << i << ", " << dof_i << ") " << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::getchar();
        }
    }
    std::cout << "Hessian Diff Test Passed" << std::endl;
    run_diff_test = false;
}

void FEMSolver::checkdfdXScale(bool perturb)
{
    std::cout << "===================== check dfdX Scale =====================" << std::endl;
    run_diff_test = true;
    project_block_PD = false;
    int n_dof = num_nodes * 2;

    VectorXT du(num_nodes * 2);
    du.setRandom();
    du *= 1.0 / du.norm();
    du *= 0.001;
    if (perturb)
        u += du;
    
    deformed = undeformed + u;
    StiffnessMatrix A(n_dof, n_dof);
    
    builddfdX(u, A);

    VectorXT f0(n_dof);
    f0.setZero();
    addElasticForceEntries(f0);
    
    
    VectorXT dx(n_dof);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    for(int i = 0; i < n_dof; i++) dx[i] += 0.5;
    dx *= 0.001;
    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {
        
        VectorXT f1(n_dof);
        f1.setZero();
        undeformed += dx;
        addElasticForceEntries(f1);
        undeformed -= dx;
        T df_norm = (f0 + (A * dx) - f1).norm();
        // std::cout << "df_norm " << df_norm << std::endl;
        if (i > 0)
        {
            std::cout << (previous/df_norm) << std::endl;
        }
        previous = df_norm;
        dx *= 0.5;
    }
    run_diff_test = false;
}

void FEMSolver::checkTotalHessianScale(bool perturb)
{
    std::cout << "===================== check Hessian 2nd Scale =====================" << std::endl;
    run_diff_test = true;
    project_block_PD = false;
    int n_dof = reduced_dof.rows();

    VectorXT du(num_nodes * 2);
    du.setRandom();
    du *= 1.0 / du.norm();
    du *= 0.001;
    if (perturb)
        u += du;
    
    StiffnessMatrix A(n_dof, n_dof);
    
    buildSystemMatrix(u, A);

    VectorXT f0(n_dof);
    f0.setZero();
    computeResidual(u, f0);
    f0 *= -1;
    
    VectorXT dx(n_dof);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    for(int i = 0; i < n_dof; i++) dx[i] += 0.5;
    dx *= 0.001;
    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {
        
        VectorXT f1(n_dof);
        f1.setZero();
        computeResidual(u + dx, f1);
        f1 *= -1;
        T df_norm = (f0 + (A * dx) - f1).norm();
        // std::cout << "df_norm " << df_norm << std::endl;
        if (i > 0)
        {
            std::cout << (previous/df_norm) << std::endl;
        }
        previous = df_norm;
        dx *= 0.5;
    }
    run_diff_test = false;
}