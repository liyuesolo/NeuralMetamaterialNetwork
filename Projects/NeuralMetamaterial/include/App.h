#ifndef APP_H
#define APP_H

#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/edges.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include <igl/png/writePNG.h>
#include "Tiling2D.h"
class Tiling2D;

class App
{
    
public:
    using TV = Vector<double, 2>;
    using TV3 = Vector<double, 3>;
    using VectorXT = Matrix<double, Eigen::Dynamic, 1>;
    using VectorXi = Matrix<int, Eigen::Dynamic, 1>;
    using IV = Vector<int, 2>;
    using IV3 = Vector<int, 3>;
    using MatrixXT = Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
    using CMat = Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>;

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd C;

    bool enable_selection = false;
    bool show_PKstress = false;
    int modes = 0;

    bool static_solve = false;

    bool check_modes = false;
    double t = 0.0;

    Eigen::MatrixXd evectors;
    Eigen::VectorXd evalues;

    int static_solve_step = 0;

public:
    virtual void updateScreen(igl::opengl::glfw::Viewer& viewer) {}
    virtual void setViewer(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu) {}

    virtual void loadDisplacementVectors(const std::string& filename);

    virtual void appendCylindersToEdges(const std::vector<std::pair<TV3, TV3>>& edge_pairs, 
        const std::vector<TV3>& color, T radius,
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C);

    App() {}
    ~App() {}
};

class SimulationApp : public App
{
public:
    Tiling2D& tiling;

    bool connect_pbc_pairs = false;
    bool tile_in_x_only = false;
    bool tile_XY = false;
    bool thicken_edges = false;
    bool show_ipc_mesh = false;
    
    void updateScreen(igl::opengl::glfw::Viewer& viewer);
    void setViewer(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu);
    

    void renderMeshSquence(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu, const std::string& folder,
        const std::string& prefix, const std::string& suffix, int n_meshes);

    SimulationApp(Tiling2D& _tiling) : tiling(_tiling) {}
    ~SimulationApp() {}
};


#endif