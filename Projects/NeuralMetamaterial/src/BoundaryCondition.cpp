#include "../include/FEMSolver.h"

void FEMSolver::addForceBox(const TV& min_corner, const TV& max_corner, const TV& force)
{
    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        TV xi = undeformed.segment<2>(i * dim);
        bool valid_x = xi[0] >= min_corner[0] && xi[0] <= max_corner[0];
        bool valid_y = xi[1] >= min_corner[1] && xi[1] <= max_corner[1];
        if (valid_x && valid_y)
            f.segment<2>(i * dim) += force;
    });
}

void FEMSolver::addDirichletBox(const TV& min_corner, const TV& max_corner, const TV& displacement)
{
    for (int i = 0; i < num_nodes; i++)
    {
        TV xi = undeformed.segment<2>(i * dim);
        bool valid_x = xi[0] >= min_corner[0] && xi[0] <= max_corner[0];
        bool valid_y = xi[1] >= min_corner[1] && xi[1] <= max_corner[1];
        if (valid_x && valid_y)
        {
            for (int d = 0; d < dim; d++)
            {
                dirichlet_data[i * dim + d] = displacement[d];
            }
        }
    }
}

void FEMSolver::addDirichletBoxX(const TV& min_corner, const TV& max_corner, T dx)
{
    for (int i = 0; i < num_nodes; i++)
    {
        TV xi = undeformed.segment<2>(i * dim);
        bool valid_x = xi[0] >= min_corner[0] && xi[0] <= max_corner[0];
        bool valid_y = xi[1] >= min_corner[1] && xi[1] <= max_corner[1];
        if (valid_x && valid_y)
        {
            dirichlet_data[i * dim] = dx;
        }
    }
}

void FEMSolver::addDirichletBoxY(const TV& min_corner, const TV& max_corner, const TV& displacement)
{
    for (int i = 0; i < num_nodes; i++)
    {
        TV xi = undeformed.segment<2>(i * dim);
        bool valid_x = xi[0] >= min_corner[0] && xi[0] <= max_corner[0];
        bool valid_y = xi[1] >= min_corner[1] && xi[1] <= max_corner[1];
        if (valid_x && valid_y)
        {
            dirichlet_data[i * dim + 1] = displacement[1];
        }
    }
}

void FEMSolver::addPenaltyPairsDisk(const TV& center, T radius, const TV& displacement)
{
    for (int i = 0; i < num_nodes; i++)
    {
        TV xi = undeformed.segment<2>(i * dim);
        if ((xi - center).norm() < radius + 1e-6)
        {
            for (int d = 0; d < dim; d++)
            {
                penalty_pairs[i * dim + d] = xi[d] + displacement[d];
            }
        }
    }
}

void FEMSolver::addPenaltyPairsBox(const TV& min_corner, const TV& max_corner, const TV& displacement)
{
    
    for (int i = 0; i < num_nodes; i++)
    {
        TV xi = undeformed.segment<2>(i * dim);
        bool valid_x = xi[0] >= min_corner[0] && xi[0] <= max_corner[0];
        bool valid_y = xi[1] >= min_corner[1] && xi[1] <= max_corner[1];
        if (valid_x && valid_y)
        {
            for (int d = 0; d < dim; d++)
            {
                penalty_pairs[i * dim + d] = xi[d] + displacement[d];
            }
        }
    }
}

void FEMSolver::addPenaltyPairsBoxXY(const TV& min_corner, const TV& max_corner, const TV& displacement)
{
    
    for (int i = 0; i < num_nodes; i++)
    {
        TV xi = undeformed.segment<2>(i * dim);
        bool valid_x = xi[0] >= min_corner[0] && xi[0] <= max_corner[0];
        bool valid_y = xi[1] >= min_corner[1] && xi[1] <= max_corner[1];
        if (valid_x && valid_y)
        {
            for (int d = 0; d < dim; d++)
            {
                penalty_pairs[i * dim + d] = xi[d] + displacement[d];
            }
        }
    }
}

void FEMSolver::addPenaltyPairsBoxX(const TV& min_corner, const TV& max_corner, T dx)
{
    
    for (int i = 0; i < num_nodes; i++)
    {
        TV xi = undeformed.segment<2>(i * dim);
        bool valid_x = xi[0] >= min_corner[0] && xi[0] <= max_corner[0];
        bool valid_y = xi[1] >= min_corner[1] && xi[1] <= max_corner[1];
        if (valid_x && valid_y)
        {
            int d = 0;
            penalty_pairs[i * dim] = xi[d] + dx;
        }
    }
}
