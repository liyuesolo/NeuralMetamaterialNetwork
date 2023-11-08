#ifndef TILING2D_H
#define TILING2D_H

#include "../tactile/tiling.hpp"
#include "../../Libs/clipper/clipper.hpp"

#include <utility>
#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <unordered_map>
#include <unordered_set>

#include <random>
#include <cmath>
#include <fstream>
#include <gmsh.h>

#include "Util.h"
#include "VecMatDef.h"
#include "FEMSolver.h"
#include "PoissonDisk.h"

template <int dim>
struct VectorHash
{
    typedef Vector<int, dim> IV;
    size_t operator()(const IV& a) const{
        std::size_t h = 0;
        for (int d = 0; d < dim; ++d) {
            h ^= std::hash<int>{}(a(d)) + 0x9e3779b9 + (h << 6) + (h >> 2); 
        }
        return h;
    }
};


class Tiling2D
{
public: 
    using dvec2 = glm::dvec2;
    using dmat3 = glm::dmat3;
    using TV = Vector<T, 2>;
    using TV2 = Vector<T, 2>;
    using TV3 = Vector<T, 3>;
    using TM2 = Matrix<T, 2, 2>;
    using TM = Matrix<T, 2, 2>;
    using IV3 = Vector<int, 3>;
    using TM3 = Matrix<T, 3, 3>;
    using IV = Vector<int, 2>;

    using PointLoops = std::vector<TV2>;
    using IdList = std::vector<int>;
    using Face = Vector<int, 3>;
    using Edge = Vector<int, 2>;
    using EdgeList = std::vector<Edge>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using Entry = Eigen::Triplet<T>;
    
    FEMSolver& solver;

    std::string data_folder = "./";
    
    T handle_width = 0.05; //cm
    bool for_printing = false;
public:
    Tiling2D(FEMSolver& _solver) : solver(_solver) {}
    ~Tiling2D() {}

    void inverseDesignFD();

    void getTilingConfig(int IH, int& n_tiling_params, 
        int& actual_IH, std::vector<TV>& bounds, VectorXT& ti_default, int& unit);

    // ########################## Tiling2D.cpp ########################## 
    void generateSurfaceMeshFromVTKFile(const std::string& vtk_file, const std::string surface_mesh_file);
    void initializeSimulationDataFromVTKFile(const std::string& filename);
    bool initializeSimulationDataFromFiles(const std::string& filename, 
        PBCType pbc_type = PBC_None, bool use_current_scale = false);
    void generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::MatrixXd& C, bool show_train = false);

    void tilingMeshInX(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);
    void tileUnitCell(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C, int n_unit = 2);

    void generateForceDisplacementPolarCurve(const std::string& result_folder);
    void generateForceDisplacementCurve(const std::string& result_folder);
    void generateForceDisplacementCurveSingleStructure(const std::string& vtk_file, const std::string& result_folder);
    
    // validation
    void generateStiffnessDataFromParams(const std::string& result_folder, int IH);
    void generatePoissonRatioDataFromParams(const std::string& result_folder, int IH);
    void generateStrainStressDataFromParams(const std::string& result_folder, int IH, bool save_result = true);
    void generateStrainStressDataFromParams(const std::string& result_folder, int IH, T theta, 
        const std::vector<T>& params, const TV& strain_range, int n_samples, const std::string& suffix, bool save_result = true);
    void generateUniaxialStrainStressDataFromParams(const std::string& result_folder, int IH, T theta, 
        const std::vector<T>& params, const TV& strain_range, int n_samples, const std::string& suffix, bool save_result = true);
    void generateStrainStressSimulationData(const std::string& result_folder, int IH, int n_samples);
    void generateStrainStressSimulationDataFromFile(const std::string& result_folder, 
        const std::string& filename, const std::string& suffix, int IH, int n_samples);
    
    void generateDeformationSequenceFromParams(const std::string& result_folder, int IH, T theta, 
        const std::vector<T>& params, const TV& strain_range, int n_samples, 
        const std::string& suffix, bool save_result = true, int n_tile = 4, bool crop = true);
    void generateDeformationSequenceAllTiling();

    void generateDeformationSequenceFromParamsStiffness(const std::string& result_folder, 
        int IH, T strain, 
        const std::vector<T>& params, const TV& theta_range, int n_samples, 
        const std::string& suffix, bool save_result = true);
    void generateDeformationSequenceAllTilingStiffness();



    
    void generatePoisonDiskSample();

    // ################ Generate Result ###############
    void generateTenPointUniaxialStrainData(const std::string& result_folder,
        int IH, T theta, const TV& strain_range, T strain_delta, const std::vector<T>& params);
    void runSimUniAxialStrainAlongDirection(const std::string& result_folder,
        int IH, int n_sample, const TV& strain_range, T theta, const std::vector<T>& params);

    // ################ Generate Training Data ###############
    void generateNHHomogenousData(const std::string& result_folder);
    void sampleUniaxialStrain(const std::string& result_folder, T strain);
    void sampleSingleStructurePoissonDisk(const std::string& result_folder, 
        const TV& uniaxial_strain_range, const TV& biaxial_strain_range, 
        const TV& theta_range, int n_sample_total, int IH);
    void sampleSingleFamily(const std::string& result_folder, 
        const TV& uniaxial_strain_range, const TV& biaxial_strain_range, 
        const TV& theta_range, int n_sp_params, int n_sp_uni, 
        int n_sp_bi, int n_sp_theta, int IH = 0);
    void sampleUniaxialStrainSingleStructure(const std::string& result_folder);
    void sampleUniaxialStrainSingleFamily(const std::string& result_folder, int IH = 0);
    void sampleBiaxialStrainSingleFamily(const std::string& result_folder, int IH = 0);
    void computeMarcoStressFromNetworkInputs(const TV3& macro_strain, int IH, 
        const VectorXT& tiling_params);
    void sampleUniAxialStrainAlongDirection(const std::string& result_folder,
        int n_sample, const TV& strain_range, T theta);
    void sampleDirectionWithUniaxialStrain(const std::string& result_folder,
        int n_sample, const TV& theta_range, T strain);
    void computeEnergyForSimData(const std::string& result_folder);
    void generateGreenStrainSecondPKPairs(const std::string& result_folder);
    void generateGreenStrainSecondPKPairsServer(const std::vector<T>& params, 
        int IH, const std::string& prefix,
        const std::string& result_folder, int resume_start = 0);
    void generateGreenStrainSecondPKPairsServerToyExample(const std::vector<T>& params,
        const std::string& result_folder);
    void sampleStrain(const std::string& result_folder);
    void sampleTilingParamsAlongStrain(const std::string& result_folder);

    void sampleFixedTilingParamsAlongStrain(const std::string& result_folder);

    // ########################## UnitPatch.cpp ########################## 
    // generate periodic mesh
    void generatePeriodicMesh(std::vector<std::vector<TV2>>& polygons, 
        std::vector<TV2>& pbc_corners, bool save_to_file = false, 
        std::string prefix = "");
    
    void generatePeriodicMeshHardCodeResolution(std::vector<std::vector<TV2>>& polygons, 
        std::vector<TV2>& pbc_corners, bool save_to_file = false, 
        std::string prefix = "");
    
    void generateNonPeriodicMesh(std::vector<std::vector<TV2>>& polygons, 
        std::vector<TV2>& pbc_corners, bool save_to_file, std::string prefix);

    void generateHomogenousMesh(std::vector<std::vector<TV2>>& polygons, 
        std::vector<TV2>& pbc_corners, bool save_to_file, std::string prefix);
    
    void generateSandwichMeshPerodicInX(std::vector<std::vector<TV2>>& polygons, 
        std::vector<TV2>& pbc_corners, 
        bool save_to_file = false, std::string filename = "",
        int resolution = 0, int element_order = 1);
    void generateSandwichMeshNonPeridoic(std::vector<std::vector<TV2>>& polygons, 
        std::vector<TV2>& pbc_corners, bool save_to_file = false, std::string filename = "");

    void generateToyExample(T param);
    void generateToyExampleStructure(const std::vector<T>& params,
        const std::string& result_folder);

    void generateOneStructureWithRotation();
    void generateOneNonperiodicStructure();
    void generateOnePerodicUnit();


    void generateOneStructureSquarePatch(int IH, const std::vector<T>& params);

    void extrudeToMesh(const std::string& tiling_param,
        const std::string& mesh3d, int n_unit);

    void loadTilingStructureFromTxt(const std::string& filename,
        std::vector<std::vector<TV2>>& eigen_polygons,
        std::vector<TV2>& eigen_base, int n_unit);

    bool fetchUnitCellFromOneFamily(int IH, int n_unit,
        std::vector<std::vector<TV2>>& eigen_polygons,
        std::vector<TV2>& eigen_base, 
        const std::vector<T>& params,
        const Vector<T, 4>& eij, const std::string& filename, T unit = 5.0, T angle = 0.0);

private:
    void minMaxCornerFromBounds(const std::vector<TV>& bounds, VectorXT& min_corner, VectorXT& max_corner);

    void generate3DSandwichMesh(std::vector<std::vector<TV2>>& polygons, 
        std::vector<TV2>& pbc_corners, bool save_to_file = false, std::string filename = "");
    
    void sampleOneFamilyWithOrientation(int IH, T angle, int n_unit, T height,
        std::vector<std::vector<TV2>>& eigen_polygons,
        std::vector<TV2>& eigen_base, const std::vector<T>& params,
        const Vector<T, 4>& eij, const std::string& filename);

    
    void sampleRegion(int IH, 
        std::vector<std::vector<TV2>>& eigen_polygons,
        std::vector<TV2>& eigen_base, 
        const std::vector<T>& params,
        const Vector<T, 4>& eij, const std::string& filename, T unit = 5.0);


    void saveClip(const ClipperLib::Paths& final_shape, 
        const Vector<T, 8>& periodic, T mult,
        const std::string& filename, bool add_box = true);

    T evalDistance(const TV& p1, const TV& p2, const TV& q, T t)
    {
        return std::sqrt(std::pow((0.1e1 - t) * p1[0] + t * p2[0] - q[0], 0.2e1) + std::pow((0.1e1 - t) * p1[1] + t * p2[1] - q[1], 0.2e1));
    };

    T closestTToLine(const TV& p1, const TV& p2, const TV& q){
        return (p1[0] * p1[0] + (-p2[0] - q[0]) * p1[0] + p1[1] * p1[1] + (-p2[1] - q[1]) * p1[1] + p2[0] * q[0] + p2[1] * q[1]) / (p1[0] * p1[0] - 2 * p1[0] * p2[0] + p1[1] * p1[1] - 2 * p1[1] * p2[1] + p2[0] * p2[0] + p2[1] * p2[1]);
    };

    // TilingHelpers.cpp 
    // Operation on the tiling unit
    glm::dmat3 centrePSRect(T xmin, T ymin, T xmax, T ymax);
    std::vector<glm::dvec2> outShapeVec(const std::vector<glm::dvec2>& vec, const glm::dmat3& M);
    void getTilingShape(std::vector<dvec2>& shape, const csk::IsohedralTiling& tiling,
        const std::vector<std::vector<dvec2>>& edges);
    void getTilingEdges(const csk::IsohedralTiling& tiling,
        const Vector<T, 4>& eij,
        std::vector<std::vector<dvec2>>& edges);
    void getTranslationUnitPolygon(std::vector<std::vector<dvec2>>& polygons_v,
        const std::vector<dvec2>& shape,
        const csk::IsohedralTiling& tiling, Vector<T, 4>& transf,
        int width, int depth, TV2& xy_shift);
    void shapeToPolygon(ClipperLib::Paths& final_shape, 
        std::vector<std::vector<TV2>>& polygons, T mult);
    void periodicToBase(const Vector<T, 8>& periodic, std::vector<TV2>& eigen_base);
        
};


#endif