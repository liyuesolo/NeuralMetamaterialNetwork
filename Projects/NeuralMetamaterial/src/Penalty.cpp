#include "../include/FEMSolver.h"

void FEMSolver::savePenaltyForces(const std::string& filename)
{
    VectorXT penalty_forces(num_nodes * 2); penalty_forces.setZero();
    addBCPenaltyForceEntries(penalty_weight, penalty_forces);
    std::ofstream out(filename);
    out << std::setprecision(20) << penalty_forces;
    out.close();
}

void FEMSolver::addBCPenaltyEnergy(T w, T& energy)
{
    T penalty_energy = 0.0;
    iterateBCPenaltyPairs([&](int offset, T target)
    {
        penalty_energy += w * 0.5 * std::pow(deformed[offset] - target, 2);
    });
    energy += penalty_energy;
}

void FEMSolver::addBCPenaltyForceEntries(T w, VectorXT& residual)
{
    iterateBCPenaltyPairs([&](int offset, T target)
    {
        residual[offset] -= w * (deformed[offset] - target);
    });
}

void FEMSolver::addBCPenaltyHessianEntries(T w, std::vector<Entry>& entries)
{
    iterateBCPenaltyPairs([&](int offset, T target)
    {
        entries.push_back(Entry(offset, offset, w));
    });
}

void FEMSolver::addUnilateralQubicPenaltyEnergy(T w, T& energy)
{
    VectorXT energies(num_nodes); energies.setZero();

    tbb::parallel_for(0, num_nodes, [&](int i){
        TV xi = deformed.segment<2>(i * 2);
        if (xi[1] < y_bar)
            return;
        T d = xi[1] - y_bar;
        energies[i] += w * std::pow(d, 3);
    });
    energy += energies.sum();
}

void FEMSolver::addUnilateralQubicPenaltyForceEntries(T w, VectorXT& residuals)
{
    tbb::parallel_for(0, num_nodes, [&](int i){
        TV xi = deformed.segment<2>(i * 2);
        if (xi[1] < y_bar)
            return;
        T d = xi[1] - y_bar;
        residuals[i * 2 + 1] -= w * 3.0 * std::pow(d, 2);
    });
}

void FEMSolver::addUnilateralQubicPenaltyHessianEntries(T w, std::vector<Entry>& entries)
{
    for (int i = 0; i < num_nodes; i++)
    {
        TV xi = deformed.segment<2>(i * 2);
        if (xi[1] < y_bar)
            continue;
        T d = xi[1] - y_bar;
        entries.push_back(Entry(i * 2 + 1, i * 2 + 1, w * 6.0 * d));
    }
    
}