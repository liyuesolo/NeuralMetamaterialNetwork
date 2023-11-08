#include <igl/readOBJ.h>
#include "../include/FEMSolver.h"
#include <iomanip>
void FEMSolver::computeBoundingBox(TV& min_corner, TV& max_corner, bool rest_state)
{
    min_corner.setConstant(1e6);
    max_corner.setConstant(-1e6);

    for (int i = 0; i < num_nodes; i++)
    {
        for (int d = 0; d < dim; d++)
        {
            if (rest_state)
            {
                max_corner[d] = std::max(max_corner[d], undeformed[i * dim + d]);
                min_corner[d] = std::min(min_corner[d], undeformed[i * dim + d]);
            }
            else
            {
                max_corner[d] = std::max(max_corner[d], deformed[i * dim + d]);
                min_corner[d] = std::min(min_corner[d], deformed[i * dim + d]);
            }
        }
    }
}


void FEMSolver::loadOBJ(const std::string& filename, bool rest_shape)
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::readOBJ(filename, V, F);
    // undeformed.resize(V.rows()*2);
    // deformed.resize(V.rows()*2);
    // surface_indices.resize(F.rows() * 3);
    // num_nodes = V.rows();
    for (int i = 0; i < num_nodes; i++)
    {
        if (rest_shape)
            undeformed.segment<2>(i * 2) = V.row(i).segment<2>(0);
        else
            deformed.segment<2>(i * 2) = V.row(i).segment<2>(0);
    }
    if (rest_shape)
        deformed = undeformed;
    
    // for (int i = 0; i < F.rows(); i++)
    // {
    //     surface_indices.segment<3>(i * 3) = F.row(i);
    // }
    
    u = deformed - undeformed;
    if (use_ipc)
    {
        updateIPCVertices(u);
    }
}

void FEMSolver::saveToOBJ(const std::string& filename, bool rest_shape)
{
    std::ofstream out(filename);
    for (int i = 0; i < num_nodes; i++)
    {
        if (rest_shape)
            out << std::setprecision(20) << "v " << undeformed.segment<2>(i * dim).transpose() << " 0" << std::endl;
        else
            out << std::setprecision(20) << "v " << deformed.segment<2>(i * dim).transpose() << " 0" << std::endl;
    }
    for (int i = 0; i < num_ele; i++)
        out << "f " << (surface_indices.segment<3>(i * 3) + IV3::Ones()).transpose() << std::endl;
    out.close();
}

void FEMSolver::saveIPCMesh(const std::string& filename)
{
    if (!use_ipc)
        return;
    std::ofstream out(filename);
    for (int i = 0; i < ipc_vertices.rows(); i++)
        out << std::setprecision(20) << "v " << ipc_vertices.row(i) << " 0" << std::endl;
    for (int i = 0; i < ipc_edges.rows(); i++)
        out << "l " << ipc_edges(i, 0) + 1 << " " << ipc_edges(i, 1) + 1 << std::endl;
    out.close();
}