#include "../include/FEMSolver.h"
#include "../include/autodiff/FEMEnergy.h"

void FEMSolver::computeFirstPiola(VectorXT& PKstress)
{
    auto computedNdb = [&](const TV &xi)
    {
        Matrix<T, 6, 2> dNdb;
        dNdb(0, 0) = -(1.0 - 2.0 * xi[0] - 2.0 * xi[1]) - 2.0 * (1.0 - xi[0] - xi[1]);
        dNdb(1, 0) = 4.0 * xi[0] - 1.0;
        dNdb(2, 0) = 0.0;
        dNdb(3, 0) = 4.0 * xi[1];
        dNdb(4, 0) = -4.0 * xi[1];
        dNdb(5, 0) = 4.0 * (1.0 - xi[0] - xi[1]) - 4.0 * xi[0];

        dNdb(0, 1) = -(1.0 - 2.0 * xi[0] - 2.0 * xi[1]) - 2.0 * (1.0 - xi[0] - xi[1]);
        dNdb(1, 1) = 0.0;
        dNdb(2, 1) = 4.0 * xi[1] - 1.0;
        dNdb(3, 1) = 4.0 * xi[0];
        dNdb(4, 1) = 4.0 * (1.0 - xi[0] - xi[1]) - 4.0 * xi[1];
        dNdb(5, 1) = -4.0 * xi[0];
        return dNdb;
    };

    auto computeDeformationGradient = [&](const Matrix<T, 2, 6> &X, 
        const Matrix<T, 2, 6> &x, const TV& xi)
    {
        Matrix<T, 6, 2> dNdb = computedNdb(xi);
        TM dXdb = X * dNdb;
        TM dxdb = x * dNdb;
        TM defGrad = dxdb * dXdb.inverse();
        return defGrad;
    };

    auto gauss2DP2Position = [&](int idx)
    {
        switch (idx)
        {
        case 0: return TV(T(1.0 / 6.0), T(1.0 / 6.0));
        case 1: return TV(T(2.0 / 3.0), T(1.0 / 6.0));
        case 2: return TV(T(1.0 / 6.0), T(2.0 / 3.0));
      
        }
    };

   
    T mu = E / 2.0 / (1.0 + nu);
    T lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);

    PKstress.resize(num_ele);
    PKstress.setZero();

    if (use_quadratic_triangle)
    {
        iterateQuadElementsParallel([&](const QuadEleNodes& x_deformed, 
            const QuadEleNodes& x_undeformed, const QuadEleIdx& indices, int ele_idx)
        {
            for (int idx = 0; idx < 3; idx++)
            {
                TV xi = gauss2DP2Position(idx);
                TM F = computeDeformationGradient(x_deformed.transpose(), x_undeformed.transpose(), xi);
                TM FinvT = F.inverse().transpose();
                TM piola = (mu * (F - FinvT) + lambda * std::log(F.determinant()) * FinvT);
                // PKstress[ele_idx] += std::sqrt(piola(0, 0) * piola(0, 0) 
                //     + piola(1, 1) * piola(1, 1)
                //     - piola(0, 0) * piola(1, 1)
                //     + 3.0 * piola(0, 1) * piola(0, 1));
                PKstress[ele_idx] += 1.0 / 3.0 * (mu * (F - FinvT) + lambda * std::log(F.determinant()) * FinvT).norm();
            }
            
            
        });
    }
    else
    {
        Matrix<T, 3, 2> dNdb;
        dNdb << -1.0, -1.0, 
            1.0, 0.0,
            0.0, 1.0;
        iterateElementsParallel([&](const EleNodes& x_deformed, 
            const EleNodes& x_undeformed, const EleIdx& indices, int ele_idx)
        {
            TM dXdb = x_undeformed.transpose() * dNdb;
            TM dxdb = x_deformed.transpose() * dNdb;

            TM F = dxdb * dXdb.inverse();
            TM FinvT = F.inverse().transpose();
            TM piola = (mu * (F - FinvT) + lambda * std::log(F.determinant()) * FinvT);
            // PKstress[ele_idx] = std::sqrt(piola(0, 0) * piola(0, 0) 
            //     + piola(1, 1) * piola(1, 1)
            //     - piola(0, 0) * piola(1, 1)
            //     + 3.0 * piola(0, 1) * piola(0, 1));
            PKstress[ele_idx] = (mu * (F - FinvT) + lambda * std::log(F.determinant()) * FinvT).norm();
        });
    }
}

void FEMSolver::computePrincipleStress(VectorXT& principle_stress)
{
    Matrix<T, 3, 2> dNdb;
        dNdb << -1.0, -1.0, 
            1.0, 0.0,
            0.0, 1.0;
    T mu = E / 2.0 / (1.0 + nu);
    T lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);

    principle_stress.resize(num_ele);
    principle_stress.setZero();
    iterateElementsParallel([&](const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, const EleIdx& indices, int ele_idx)
    {
        TM dXdb = x_undeformed.transpose() * dNdb;
        TM dxdb = x_deformed.transpose() * dNdb;

        TM F = dxdb * dXdb.inverse();
        TM FinvT = F.inverse().transpose();
        TM piola = mu * (F - FinvT) + lambda * std::log(F.determinant()) * FinvT;
        Eigen::JacobiSVD<TM, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(piola);
        TV singular_values = svd.singularValues();
        principle_stress[ele_idx] = singular_values.sum();
    });
}

T FEMSolver::computeTotalArea()
{
    VectorXT areas(num_ele);
    tbb::parallel_for(0, (int)(surface_indices.rows()/3), [&](int i)
    {
        TV e0 = undeformed.segment<2>(surface_indices[i*3+1]*2) - undeformed.segment<2>(surface_indices[i*3]*2);
        TV e1 = undeformed.segment<2>(surface_indices[i*3+2]*2) - undeformed.segment<2>(surface_indices[i*3]*2);
        areas[i] = 0.5 * std::abs(e0[0]*e1[1] - e0[1]*e1[0]);
    });
    return areas.sum();
}

void FEMSolver::addElastsicPotential(T& energy)
{
    VectorXT energies_neoHookean(num_ele);
    energies_neoHookean.setZero();
    if (use_quadratic_triangle)
        iterateQuadElementsParallel([&](const QuadEleNodes& x_deformed, 
            const QuadEleNodes& x_undeformed, const QuadEleIdx& indices, int tet_idx)
        {
            T ei;
            if (use_mooney_rivlin)
                compute2DQuadraticMooneyRivilinEnergy(x_deformed, x_undeformed, C10, C01, C11, lambda, ei);
            else
                computeQuadratic2DNeoHookeanEnergy(E, nu, x_deformed, x_undeformed, ei);
            energies_neoHookean[tet_idx] += ei;
        });
    else
        iterateElementsParallel([&](const EleNodes& x_deformed, 
            const EleNodes& x_undeformed, const EleIdx& indices, int tet_idx)
        {
            T ei;
            if (use_mooney_rivlin)
                compute2DLinearMooneyRivilinEnergy(x_deformed, x_undeformed, C10, C01, C11, lambda, ei);
            else
                computeLinear2DNeoHookeanEnergy(E, nu, x_deformed, x_undeformed, ei);
            
            energies_neoHookean[tet_idx] += ei;
        });
    energy += thickness * energies_neoHookean.sum();
}



void FEMSolver::addElasticForceEntries(VectorXT& residual)
{
    
    if (use_quadratic_triangle)
        iterateQuadElementsSerial([&](const QuadEleNodes& x_deformed, 
            const QuadEleNodes& x_undeformed, const QuadEleIdx& indices, int tet_idx)
        {

            Vector<T, 12> dedx;
            if (use_mooney_rivlin)
                compute2DQuadraticMooneyRivilinEnergyGradient(x_deformed, x_undeformed, C10, C01, C11, lambda, dedx);
            else
            {
                computeQuadratic2DNeoHookeanEnergyGradient(E, nu, x_deformed, x_undeformed, dedx);
            }
            
            addForceEntry<12>(residual, indices, -thickness * dedx);
        });
    else
        iterateElementsSerial([&](const EleNodes& x_deformed, 
            const EleNodes& x_undeformed, const EleIdx& indices, int tet_idx)
        {
            Vector<T, 6> dedx;
            if (use_mooney_rivlin)
                compute2DLinearMooneyRivilinEnergyGradient(x_deformed, x_undeformed, C10, C01, C11, lambda, dedx);
            else
                computeLinear2DNeoHookeanEnergyGradient(E, nu, x_deformed, x_undeformed, dedx);

            addForceEntry<6>(residual, indices, -thickness * dedx);
        });
}

void FEMSolver::addElasticHessianEntries(std::vector<Entry>& entries, bool project_PD)
{
    if (use_quadratic_triangle)
        iterateQuadElementsSerial([&](const QuadEleNodes& x_deformed, 
            const QuadEleNodes& x_undeformed, const QuadEleIdx& indices, int tet_idx)
        {
            Matrix<T, 12, 12> hessian;
            if (use_mooney_rivlin)
                compute2DQuadraticMooneyRivilinEnergyHessian(x_deformed, x_undeformed, C10, C01, C11, lambda, hessian);
            else
                computeQuadratic2DNeoHookeanEnergyHessian(E, nu, x_deformed, x_undeformed, hessian);
            if (project_PD)
                projectBlockPD<12>(hessian);
            
            addHessianEntry<12>(entries, indices, thickness * hessian);
        });
    else
        iterateElementsSerial([&](const EleNodes& x_deformed, 
            const EleNodes& x_undeformed, const EleIdx& indices, int tet_idx)
        {
            Matrix<T, 6, 6> hessian;
            if (use_mooney_rivlin)
                compute2DLinearMooneyRivilinEnergyHessian(x_deformed, x_undeformed, C10, C01, C11, lambda, hessian);
            else
                computeLinear2DNeoHookeanEnergyHessian(E, nu, x_deformed, x_undeformed, hessian);
            
            if (project_PD)
                projectBlockPD<6>(hessian);
            
            addHessianEntry<6>(entries, indices, thickness * hessian);
        });
}

void FEMSolver::addElasticdfdXEntries(std::vector<Entry>& entries)
{
    if (use_quadratic_triangle)
        iterateQuadElementsSerial([&](const QuadEleNodes& x_deformed, 
            const QuadEleNodes& x_undeformed, const QuadEleIdx& indices, int tet_idx)
        {
            Matrix<T, 12, 12> dfdX;
            computeQuadratic2DNeoHookeandfdX(E, nu, x_deformed, x_undeformed, dfdX);
            addHessianEntry<12>(entries, indices, dfdX);
        });
    else
        iterateElementsSerial([&](const EleNodes& x_deformed, 
            const EleNodes& x_undeformed, const EleIdx& indices, int tet_idx)
        {
            Matrix<T, 6, 6> dfdX;
            computeLinear2DNeoHookeandfdX(E, nu, x_deformed, x_undeformed, dfdX);
            addHessianEntry<6>(entries, indices, dfdX);
        });
}

T FEMSolver::computeInversionFreeStepsize(const VectorXT& _u, const VectorXT& du)
{
    if (use_quadratic_triangle)
    {
        auto computedNdb = [&](const TV &xi)
        {
            Matrix<T, 6, 2> dNdb;
            dNdb(0, 0) = -(1.0 - 2.0 * xi[0] - 2.0 * xi[1]) - 2.0 * (1.0 - xi[0] - xi[1]);
            dNdb(1, 0) = 4.0 * xi[0] - 1.0;
            dNdb(2, 0) = 0.0;
            dNdb(3, 0) = 4.0 * xi[1];
            dNdb(4, 0) = -4.0 * xi[1];
            dNdb(5, 0) = 4.0 * (1.0 - xi[0] - xi[1]) - 4.0 * xi[0];

            dNdb(0, 1) = -(1.0 - 2.0 * xi[0] - 2.0 * xi[1]) - 2.0 * (1.0 - xi[0] - xi[1]);
            dNdb(1, 1) = 0.0;
            dNdb(2, 1) = 4.0 * xi[1] - 1.0;
            dNdb(3, 1) = 4.0 * xi[0];
            dNdb(4, 1) = 4.0 * (1.0 - xi[0] - xi[1]) - 4.0 * xi[1];
            dNdb(5, 1) = -4.0 * xi[0];
            return dNdb;
        };

        auto gauss2DP2Position = [&](int idx)
        {
            switch (idx)
            {
            case 0: return TV(T(1.0 / 6.0), T(1.0 / 6.0));
            case 1: return TV(T(2.0 / 3.0), T(1.0 / 6.0));
            case 2: return TV(T(1.0 / 6.0), T(2.0 / 3.0));
        
            }
        };
        VectorXT step_sizes = VectorXT::Zero(num_ele * 3);
        iterateQuadElementsParallel([&](const QuadEleNodes& x_deformed, 
            const QuadEleNodes& x_undeformed, const QuadEleIdx& indices, int ele_idx)
        {
            for (int idx = 0; idx < 3; idx++)
            {
                TV xi = gauss2DP2Position(idx);
                Matrix<T, 6, 2> dNdb = computedNdb(xi);
                TM dXdb = x_undeformed.transpose() * dNdb;
                TM dxdb = x_deformed.transpose() * dNdb;
                TM A = dxdb * dXdb.inverse();
                T a, b, c, d;
                a = 0;
                b = A.determinant();
                c = A.diagonal().sum();
                d = 0.8;

                T t = getSmallestPositiveRealCubicRoot(a, b, c, d);
                if (t < 0 || t > 1) t = 1;
                    step_sizes(ele_idx * 3 + idx) = t;
            } 
        });
        return step_sizes.minCoeff();
    }
    else
    {
        Matrix<T, 3, 2> dNdb;
        dNdb << -1.0, -1.0, 
            1.0, 0.0,
            0.0, 1.0;
           
        VectorXT step_sizes = VectorXT::Zero(num_ele);

        iterateElementsParallel([&](const EleNodes& x_deformed, 
            const EleNodes& x_undeformed, const EleIdx& indices, int tet_idx)
        {
            TM dXdb = x_undeformed.transpose() * dNdb;
            TM dxdb = x_deformed.transpose() * dNdb;
            TM A = dxdb * dXdb.inverse();
            T a, b, c, d;
            a = 0;
            b = A.determinant();
            c = A.diagonal().sum();
            d = 0.8;

            T t = getSmallestPositiveRealCubicRoot(a, b, c, d);
            if (t < 0 || t > 1) t = 1;
                step_sizes(tet_idx) = t;
        });
        return step_sizes.minCoeff();
    }
    
}

