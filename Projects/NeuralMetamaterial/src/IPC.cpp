#include <igl/edges.h>
#include <igl/boundary_loop.h>
#include <ipc/ipc.hpp>
#include <ipc/barrier/adaptive_stiffness.hpp>
#include "../include/FEMSolver.h"
#include <unordered_set>
#include "../include/Timer.h"
// #include <igl/writeOBJ.h>

void FEMSolver::updateBarrierInfo(bool first_step)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    for (int i = 0; i < num_ipc_vtx; i++) 
        ipc_vertices_deformed.row(i) = deformed.segment<2>(coarse_to_fine[i] * 2);

    MatrixXT ipc_vertices_2x2_deformed;
    constructPeriodicContactPatch(ipc_vertices_deformed, ipc_vertices_2x2_deformed, deformed);

    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices_2x2, ipc_vertices_2x2_deformed, 
        ipc_edges_2x2, ipc_faces, barrier_distance, ipc_constraints, Eigen::MatrixXi(), 
        0.0, ipc::BroadPhaseMethod::HASH_GRID, 
        [&](size_t vtx0, size_t vtx1)
            {
                if (is_interior_vtx[vtx0%num_ipc_vtx]
                    || is_interior_vtx[vtx1%num_ipc_vtx])
                    return false;
                return true; 
            }
        );
        
    T current_min_dis = ipc::compute_minimum_distance(ipc_vertices_2x2, ipc_edges_2x2, ipc_faces, ipc_constraints);
    if (first_step)
        ipc_min_dis = current_min_dis;
    else
    {
        TV min_corner, max_corner;
        computeBoundingBox(min_corner, max_corner);
        T bb_diag = (max_corner - min_corner).norm();
        ipc::update_barrier_stiffness(ipc_min_dis, current_min_dis, max_barrier_weight, barrier_weight, bb_diag);
        ipc_min_dis = current_min_dis;
    }
}

void FEMSolver::constructPeriodicContactPatch(const MatrixXT& ipc_vertices_unit, 
    MatrixXT& _ipc_vertices_2x2, const VectorXT& position)
{
    if (pbc_type == PBC_XY)
    {
        TV left0 = position.segment<2>(pbc_pairs[0][0][0] * 2);
        TV right0 = position.segment<2>(pbc_pairs[0][0][1] * 2);
        TV top0 = position.segment<2>(pbc_pairs[1][0][1] * 2);
        TV bottom0 = position.segment<2>(pbc_pairs[1][0][0] * 2);
        TV dx = (right0 - left0);
        TV dy = (top0 - bottom0);

        int n_unit_vtx = ipc_vertices_unit.rows();
        _ipc_vertices_2x2.resize(n_unit_vtx * 4, 2);
        for (int i = 0; i < 4; i++)
            _ipc_vertices_2x2.block(n_unit_vtx*i, 0, n_unit_vtx, 2) = ipc_vertices_unit;    

        for (int i = 0; i < n_unit_vtx; i++)
        {
            _ipc_vertices_2x2.row(n_unit_vtx + i) += dx;
            _ipc_vertices_2x2.row(n_unit_vtx * 2 + i) += dy;
            _ipc_vertices_2x2.row(n_unit_vtx * 3 + i) += (dx + dy);
        } 
    }
    else
    {
        _ipc_vertices_2x2 = ipc_vertices_unit;
    }
}

void FEMSolver::computeIPCRestData()
{
    if (use_quadratic_triangle)
    {
        // linear to quadratic node map, use linear triangle for contact.
        // could use higher order IPC in the future
        std::unordered_set<int> surface_index_map;
        for (int i = 0; i < surface_indices.rows(); i++)
        {
            surface_index_map.insert(surface_indices[i]);
        }
        coarse_to_fine.resize(surface_index_map.size());
        fine_to_coarse.clear();
        int cnt = 0;
        for (const auto& value: surface_index_map)
        {
            coarse_to_fine[cnt] = value;
            fine_to_coarse[value] = cnt;
            cnt++;
        }

        num_ipc_vtx = surface_index_map.size();
        ipc_vertices.resize(num_ipc_vtx, 2);
        for (int i = 0; i < num_ipc_vtx; i++)
        {
            ipc_vertices.row(i) = undeformed.segment<2>(coarse_to_fine[i] * 2);
        }

    }
    else
    {
        ipc_vertices.resize(num_nodes, 2);
        coarse_to_fine.resize(num_nodes);
        fine_to_coarse.clear();
        for (int i = 0; i < num_nodes; i++)
            ipc_vertices.row(i) = undeformed.segment<2>(i * 2);
        num_ipc_vtx = ipc_vertices.rows();
        for (int i = 0; i < num_ipc_vtx; i++)
        {
            coarse_to_fine[i] = i;
            fine_to_coarse[i] = i;
        }
        
    }
    
    ipc_faces.resize(num_ele, 3);

    for (int i = 0; i < num_ele; i++)
    {
        if (use_quadratic_triangle)
        {
            ipc_faces.row(i) = IV3(fine_to_coarse[surface_indices[i * 3 + 0]],
                                    fine_to_coarse[surface_indices[i * 3 + 1]],
                                    fine_to_coarse[surface_indices[i * 3 + 2]]);
        }
        else
            ipc_faces.row(i) = indices.segment<3>(i * 3);
    }

    std::vector<std::vector<int>> boundary_vertices;
    igl::boundary_loop(ipc_faces, boundary_vertices);

    int n_bd_edge = 0;
    for (auto loop : boundary_vertices)
    {
        n_bd_edge += loop.size();
    }
    
    ipc_edges.resize(n_bd_edge, 2);
    int edge_cnt = 0;
    for (auto loop : boundary_vertices)
    {
        for (int i = 0; i < loop.size(); i++)
        {
            ipc_edges.row(edge_cnt++) = Edge(loop[i], loop[(i+1)%loop.size()]);
        }
    }

    ipc_faces.resize(0, 0);
    is_pbc_vtx.resize(num_ipc_vtx, false);

    is_interior_vtx.resize(num_ipc_vtx, true);
    if (pbc_type == PBC_XY)
    {
        std::vector<int> pbc_vtx;
        for (int dir = 0; dir < 2; dir++)
            for (IV& pbc_pair : pbc_pairs[dir])
            {
                int idx0 = pbc_pair[0], idx1 = pbc_pair[1];
                is_pbc_vtx[fine_to_coarse[idx0]] = true; is_pbc_vtx[fine_to_coarse[idx1]] = true;
                pbc_vtx.push_back(fine_to_coarse[idx0]), pbc_vtx.push_back(fine_to_coarse[idx1]);
                
            }

        int n_unit_edge = ipc_edges.rows();
        std::vector<int> valid_list;
        for (int i = 0; i < n_unit_edge; i++)
        {
            bool first_point_found = is_pbc_vtx[ipc_edges(i, 0)];
            bool second_point_found = is_pbc_vtx[ipc_edges(i, 1)];
            if (first_point_found && second_point_found)
            {
                continue;
            }
            valid_list.push_back(i);
        }
        // std::cout << valid_list.size() << std::endl;
        Eigen::MatrixXi ipc_edge_valid(valid_list.size(), 2);
        T min_edge_length = 1e3;
        for (int i = 0; i < valid_list.size(); i++)
        {
            ipc_edge_valid.row(i) = ipc_edges.row(valid_list[i]);
            is_interior_vtx[ipc_edge_valid(i, 0)] = false || is_pbc_vtx[ipc_edge_valid(i, 0)];
            is_interior_vtx[ipc_edge_valid(i, 1)] = false || is_pbc_vtx[ipc_edge_valid(i, 1)];
            T edge_length = (ipc_vertices.row(ipc_edge_valid(i, 0)) - ipc_vertices.row(ipc_edge_valid(i, 1))).norm();
            if (edge_length < min_edge_length)
                min_edge_length = edge_length;
        }
        // std::cout << "min edge length " << min_edge_length << " " << barrier_distance << std::endl;
        ipc_edges = ipc_edge_valid;

        int n_unit_vtx = ipc_vertices.rows();

        Eigen::MatrixXi offset(ipc_edges.rows(), 2);
        offset.setConstant(n_unit_vtx);

        ipc_edges_2x2.resize(ipc_edges.rows() * 4, 2);
        for (int i = 0; i < 4; i++)
            ipc_edges_2x2.block(ipc_edges.rows() * i, 0, ipc_edges.rows(), 2) = ipc_edges + i * offset;
        
        // construct contact jacobian, we use the DoF of one simulation unit
        jacobian.resize(n_unit_vtx * 4 * 2, n_unit_vtx * 2); 
        std::vector<Entry> jacobian_entries;
        for (int i = 0; i < n_unit_vtx*2; i++)
        {
            jacobian_entries.push_back(Entry(i, i, 1.0));
            jacobian_entries.push_back(Entry(n_unit_vtx * 2 + i, i, 1.0));
            jacobian_entries.push_back(Entry(n_unit_vtx * 2 * 2 + i, i, 1.0));
            jacobian_entries.push_back(Entry(n_unit_vtx * 3 * 2 + i, i, 1.0));
        }

        int dof_left = fine_to_coarse[pbc_pairs[0][0][0]];
        int dof_right = fine_to_coarse[pbc_pairs[0][0][1]];
        int dof_top = fine_to_coarse[pbc_pairs[1][0][1]];
        int dof_bottom = fine_to_coarse[pbc_pairs[1][0][0]];
        
        for (int i = 0; i < n_unit_vtx; i++)
        {
            for (int d = 0; d < 2; d++)
            {
                jacobian_entries.push_back(Entry(n_unit_vtx * 2 + i * 2 + d, dof_left * 2 + d, -1.0));
                jacobian_entries.push_back(Entry(n_unit_vtx * 2 + i * 2 + d, dof_right * 2 + d, 1.0));

                jacobian_entries.push_back(Entry(n_unit_vtx * 2 * 2 + i * 2 + d, dof_bottom * 2 + d, -1.0));
                jacobian_entries.push_back(Entry(n_unit_vtx * 2 * 2 + i * 2 + d, dof_top * 2 + d, 1.0));

                jacobian_entries.push_back(Entry(n_unit_vtx * 2 * 3 + i * 2 + d, dof_left * 2 + d, -1.0));
                jacobian_entries.push_back(Entry(n_unit_vtx * 2 * 3 + i * 2 + d, dof_right * 2 + d, 1.0));

                jacobian_entries.push_back(Entry(n_unit_vtx * 2 * 3 + i * 2 + d, dof_bottom * 2 + d, -1.0));
                jacobian_entries.push_back(Entry(n_unit_vtx * 2 * 3 + i * 2 + d, dof_top * 2 + d, 1.0));
            }
        }
        jacobian.setFromTriplets(jacobian_entries.begin(), jacobian_entries.end());
    }
    else
    {
        ipc_edges_2x2 = ipc_edges;
        for (int i = 0; i < ipc_edges.rows(); i++)
        {
            is_interior_vtx[ipc_edges(i, 0)] = false;
            is_interior_vtx[ipc_edges(i, 1)] = false;
        }
        jacobian.resize(num_ipc_vtx * 2, num_ipc_vtx * 2); 
        jacobian.setIdentity();
    }

    // map to 2x2 tile
    constructPeriodicContactPatch(ipc_vertices, ipc_vertices_2x2, deformed);


    VectorXT ipc_vertices_flat(ipc_vertices.rows() * 2),
         ipc_vertices_2x2_flat(ipc_vertices_2x2.rows() * 2);
    for (int i = 0; i < ipc_vertices.rows(); i++)
    {
        ipc_vertices_flat.segment<2>(i * 2) = ipc_vertices.row(i);
    }
    for (int i = 0; i < ipc_vertices_2x2.rows(); i++)
    {
        ipc_vertices_2x2_flat.segment<2>(i * 2) = ipc_vertices_2x2.row(i);
    }
    
    VectorXT prediction = jacobian * ipc_vertices_flat;
    T error = (prediction - ipc_vertices_2x2_flat).norm();
    if (error > 1e-8)
    {
        std::cout << "prediction error: " << (prediction - ipc_vertices_2x2_flat).norm() << std::endl;
        std::cout << __FILE__ << std::endl;
        // std::exit(0);
    }

    
    if (verbose)
        std::cout << "ipc has ixn in rest state: " 
            << ipc::has_intersections(ipc_vertices_2x2, ipc_edges_2x2, ipc_faces, 
            [&](size_t vtx0, size_t vtx1)
            {
                if (is_interior_vtx[vtx0%num_ipc_vtx]
                    || is_interior_vtx[vtx1%num_ipc_vtx])
                    return false;
                return true; 
            }) 
            << std::endl;
    
    TV min_corner, max_corner;
    computeBoundingBox(min_corner, max_corner);
    T bb_diag = (max_corner - min_corner).norm();
    
    VectorXT dedx(reduced_dof.rows()), dbdx(deformed.rows());
    dedx.setZero(); dbdx.setZero();
    barrier_weight = 1.0;
    addIPCForceEntries(dbdx); dbdx *= -1.0;
    
    computeResidual(u, dedx); dedx *= -1.0; dedx -= jac_full2reduced.transpose() * dbdx;
    
    barrier_weight = ipc::initial_barrier_stiffness(bb_diag, barrier_distance, 1.0, 
        dedx, dbdx, max_barrier_weight);
    if (verbose)
        std::cout << "barrier weight " <<  barrier_weight << " max_barrier_weight " << max_barrier_weight << std::endl;
    
}

T FEMSolver::computeCollisionFreeStepsize(const VectorXT& _u, const VectorXT& du)
{

    VectorXT u_full = jac_full2reduced * _u;
    VectorXT du_full = jac_full2reduced * du;

    Eigen::MatrixXd current_position = ipc_vertices, 
        next_step_position = ipc_vertices;
        
    for (int i = 0; i < num_ipc_vtx; i++)
    {
        current_position.row(i) = undeformed.segment<2>(coarse_to_fine[i] * 2) + u_full.segment<2>(coarse_to_fine[i] * 2);
        next_step_position.row(i) = undeformed.segment<2>(coarse_to_fine[i] * 2) + u_full.segment<2>(coarse_to_fine[i] * 2) + du_full.segment<2>(coarse_to_fine[i] * 2);
    }

    MatrixXT ipc_vertices_2x2_current, ipc_vertices_2x2_next_step;
    constructPeriodicContactPatch(current_position, ipc_vertices_2x2_current, undeformed + u_full);
    constructPeriodicContactPatch(next_step_position, ipc_vertices_2x2_next_step, undeformed + u_full + du_full);

    return ipc::compute_collision_free_stepsize(ipc_vertices_2x2_current, 
            ipc_vertices_2x2_next_step, ipc_edges_2x2, ipc_faces, 
            ipc::BroadPhaseMethod::HASH_GRID, 1e-6, 1e7,
            [&](size_t vtx0, size_t vtx1)
            {
                if (is_interior_vtx[vtx0%num_ipc_vtx]
                    || is_interior_vtx[vtx1%num_ipc_vtx])
                    return false;
                return true; 
            });
}



void FEMSolver::updateIPCVertices(const VectorXT& _u)
{
    VectorXT projected = jac_full2reduced * _u;
    iterateDirichletDoF([&](int offset, T target)
    {
        projected[offset] = target;
    });
    deformed = undeformed + projected;
    for (int i = 0; i < num_ipc_vtx; i++)
        ipc_vertices.row(i) = deformed.segment<2>(coarse_to_fine[i] * 2);
    
    constructPeriodicContactPatch(ipc_vertices, ipc_vertices_2x2, deformed);
}

void FEMSolver::addIPCEnergy(T& energy)
{
    T contact_energy = 0.0;
    
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    for (int i = 0; i < num_ipc_vtx; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<2>(coarse_to_fine[i] * 2);
    }

    MatrixXT ipc_vertices_2x2_deformed;
    constructPeriodicContactPatch(ipc_vertices_deformed, ipc_vertices_2x2_deformed, deformed);

    ipc::Constraints ipc_constraints;

    ipc::construct_constraint_set(ipc_vertices_2x2, ipc_vertices_2x2_deformed, 
        ipc_edges_2x2, ipc_faces, barrier_distance, ipc_constraints, Eigen::MatrixXi(), 
        0.0, ipc::BroadPhaseMethod::HASH_GRID, 
        [&](size_t vtx0, size_t vtx1)
            {
                if (is_interior_vtx[vtx0%num_ipc_vtx]
                    || is_interior_vtx[vtx1%num_ipc_vtx])
                    return false;
                return true; 
            }
        );

    contact_energy = barrier_weight * ipc::compute_barrier_potential(ipc_vertices_2x2_deformed, 
    ipc_edges_2x2, ipc_faces, ipc_constraints, barrier_distance);

    if (add_friction)
    {
        ipc::FrictionConstraints ipc_friction_constraints;
        ipc::construct_friction_constraint_set(
            ipc_vertices_2x2_deformed, ipc_edges_2x2, ipc_faces, ipc_constraints,
            barrier_distance, barrier_weight, friction_mu, ipc_friction_constraints
        );
        T friction_energy = ipc::compute_friction_potential<T>(
            ipc_vertices_2x2, ipc_vertices_2x2_deformed, ipc_edges_2x2,
            ipc_faces, ipc_friction_constraints, epsv_times_h
        );
        energy += friction_energy;
    }

    energy += contact_energy;

    
}
void FEMSolver::addIPCForceEntries(VectorXT& residual)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    for (int i = 0; i < num_ipc_vtx; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<2>(coarse_to_fine[i] * 2);
    }


    MatrixXT ipc_vertices_2x2_deformed;
    constructPeriodicContactPatch(ipc_vertices_deformed, ipc_vertices_2x2_deformed, deformed);

    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices_2x2, ipc_vertices_2x2_deformed, 
        ipc_edges_2x2, ipc_faces, barrier_distance, ipc_constraints, Eigen::MatrixXi(), 
        0.0, ipc::BroadPhaseMethod::HASH_GRID, 
        [&](size_t vtx0, size_t vtx1)
            {

                if (is_interior_vtx[vtx0%num_ipc_vtx]
                    || is_interior_vtx[vtx1%num_ipc_vtx])
                    return false;    
                return true; 
            }
        );
    VectorXT contact_gradient;
    if (use_reduced)
        contact_gradient = barrier_weight * 
            ipc::compute_barrier_potential_gradient(ipc_vertices_2x2_deformed, 
            ipc_edges_2x2, ipc_faces, ipc_constraints, barrier_distance).transpose() * jacobian;
    else
        contact_gradient = barrier_weight * 
            ipc::compute_barrier_potential_gradient(ipc_vertices_2x2_deformed, 
            ipc_edges_2x2, ipc_faces, ipc_constraints, barrier_distance).transpose();

    if (add_friction)
    {
        ipc::FrictionConstraints ipc_friction_constraints;
        ipc::construct_friction_constraint_set(
            ipc_vertices_2x2_deformed, ipc_edges_2x2, ipc_faces, ipc_constraints,
            barrier_distance, barrier_weight, friction_mu, ipc_friction_constraints
        );
        VectorXT friction_energy_gradient = ipc::compute_friction_potential_gradient(
            ipc_vertices_2x2, ipc_vertices_2x2_deformed, ipc_edges_2x2,
            ipc_faces, ipc_friction_constraints, epsv_times_h
        );
        if (use_reduced) 
            contact_gradient += friction_energy_gradient * jacobian;
        else
            contact_gradient += friction_energy_gradient;
    }    
    

    for (int i = 0; i < num_ipc_vtx; i++)
    {
        residual.segment<2>(coarse_to_fine[i] * 2) -= contact_gradient.segment<2>(i * 2);
    }
    

}
void FEMSolver::addIPCHessianEntries(std::vector<Entry>& entries,bool project_PD)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    for (int i = 0; i < num_ipc_vtx; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<2>(coarse_to_fine[i] * 2);
    }

    
    MatrixXT ipc_vertices_2x2_deformed;
    constructPeriodicContactPatch(ipc_vertices_deformed, ipc_vertices_2x2_deformed, deformed);
    
    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices_2x2, ipc_vertices_2x2_deformed, 
        ipc_edges_2x2, ipc_faces, barrier_distance, ipc_constraints, Eigen::MatrixXi(), 
        0.0, ipc::BroadPhaseMethod::HASH_GRID, 
        [&](size_t vtx0, size_t vtx1)
            {
                if (is_interior_vtx[vtx0%num_ipc_vtx]
                    || is_interior_vtx[vtx1%num_ipc_vtx])
                    return false;
                return true; 
            }
        );
    StiffnessMatrix contact_hessian;
    if (use_reduced)
        contact_hessian = barrier_weight *  
            jacobian.transpose() * ipc::compute_barrier_potential_hessian(ipc_vertices_2x2_deformed, 
        ipc_edges_2x2, ipc_faces, ipc_constraints, barrier_distance, project_PD) * jacobian;
    else
        contact_hessian = barrier_weight *  
            ipc::compute_barrier_potential_hessian(ipc_vertices_2x2_deformed, 
        ipc_edges_2x2, ipc_faces, ipc_constraints, barrier_distance, project_PD);
    
    if (add_friction)
    {
        ipc::FrictionConstraints ipc_friction_constraints;
        ipc::construct_friction_constraint_set(
            ipc_vertices_2x2_deformed, ipc_edges_2x2, ipc_faces, ipc_constraints,
            barrier_distance, barrier_weight, friction_mu, ipc_friction_constraints
        );
        StiffnessMatrix friction_energy_hessian = ipc::compute_friction_potential_hessian(
            ipc_vertices_2x2, ipc_vertices_2x2_deformed, ipc_edges_2x2,
            ipc_faces, ipc_friction_constraints, epsv_times_h
        );
        if (use_reduced)
            contact_hessian += jacobian.transpose() * friction_energy_hessian * jacobian;
        else
            contact_hessian += friction_energy_hessian;
    }

    std::vector<Entry> contact_entries = entriesFromSparseMatrix(contact_hessian);
    
    for (Entry& entry : contact_entries)
    {
        int node_i = std::floor(entry.row() / 2);
        int node_j = std::floor(entry.col() / 2);
        entries.push_back(Entry(coarse_to_fine[node_i] * 2 + entry.row() % 2, 
                            coarse_to_fine[node_j] * 2 + entry.col() % 2, 
                            entry.value()));
    }
    
}

