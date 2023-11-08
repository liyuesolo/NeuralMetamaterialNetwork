#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/png/writePNG.h>
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include <igl/jet.h>

#include "../include/App.h"
#include <boost/filesystem.hpp>


inline bool fileExist (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

using CMat = Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>;
using TV = Vector<T, 2>;
using TV3 = Vector<T, 3>;
using VectorXT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

int main(int argc, char** argv)
{
    
    FEMSolver fem_solver;
    fem_solver.use_quadratic_triangle = true;
    Tiling2D tiling(fem_solver);
    
    // for running simulation on the server
    if (argc > 1)
    {
        int IH = std::stoi(argv[1]);
        std::string result_folder = argv[2];
        if (IH == -1)
        {
            T pi = std::stod(argv[3]);
            std::vector<T> params = {pi};
            tiling.generateGreenStrainSecondPKPairsServerToyExample(params, result_folder);
        }
        else
        {
            int n_params = std::stoi(argv[3]);
            std::vector<T> params(n_params);
            for (int i = 0; i < n_params; i++)
            {
                params[i] = std::stod(argv[4+i]);
            }
            int resume_start = std::stoi(argv[4 + n_params]);
            tiling.generateGreenStrainSecondPKPairsServer(params, IH, "", result_folder, resume_start);
        }
        
    }
    // local viewer
    else
    {
        igl::opengl::glfw::Viewer viewer;
        igl::opengl::glfw::imgui::ImGuiMenu menu;

        viewer.plugins.push_back(&menu);
        SimulationApp app(tiling);

        app.setViewer(viewer, menu);

        auto runSimApp = [&]()
        {
            viewer.launch(true, false, "NMN viewer", 2000, 1600);
        };


        runSimApp();
        
    }
    return 0;
}