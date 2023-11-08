#include <ipc/ipc.hpp>
#include "../include/Tiling2D.h"
#include <time.h>
std::random_device rd;
std::mt19937 gen( rd() );
std::uniform_real_distribution<> dis( 0.0, 1.0 );

static T zeta()
{
    return dis(gen);
}

void Tiling2D::generate3DSandwichMesh(std::vector<std::vector<TV2>>& polygons, 
    std::vector<TV2>& pbc_corners, bool save_to_file, std::string filename)
{
    T eps = 1e-5;
    gmsh::initialize();

    gmsh::model::add("tiling");
    gmsh::logger::start();

    gmsh::option::setNumber("Geometry.ToleranceBoolean", eps);
    gmsh::option::setNumber("Geometry.Tolerance", eps);
    gmsh::option::setNumber("Mesh.ElementOrder", 1);

    // disable set resolution from point option
    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

 
    //Points
    int acc = 1;
    // clamping box 1 2 3 4
    for (int i = 0; i < pbc_corners.size(); ++i)
		gmsh::model::occ::addPoint(pbc_corners[i][0], pbc_corners[i][1], 0, 2, acc++);
    
    // sandwich boxes bottom 5 6 the other two points already exist
    T dx = 0.02 * (pbc_corners[1][0] - pbc_corners[0][0]);
    gmsh::model::occ::addPoint(pbc_corners[0][0],  pbc_corners[0][1] - dx, 0, 2, acc++);
    gmsh::model::occ::addPoint(pbc_corners[1][0], pbc_corners[1][1] - dx, 0, 2, acc++);
    
    // sandwich boxes top 7 8 
    gmsh::model::occ::addPoint(pbc_corners[2][0], pbc_corners[2][1] + dx, 0, 2, acc++);
    gmsh::model::occ::addPoint(pbc_corners[3][0], pbc_corners[3][1] + dx, 0, 2, acc++);

    // inner lattice
    for (int i=0; i<polygons.size(); ++i)
    {
        for(int j=0; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addPoint(polygons[i][j][0], polygons[i][j][1], 0, 2, acc++);
        }
    }
    
    //Lines
    acc = 1;

    int acc_line = 1;
    
    // add clipping box
    gmsh::model::occ::addLine(1, 2, acc++); 
    gmsh::model::occ::addLine(2, 3, acc++); 
    gmsh::model::occ::addLine(3, 4, acc++); 
    gmsh::model::occ::addLine(4, 1, acc++);
    
    // bottom box
    gmsh::model::occ::addLine(5, 6, acc++); 
    gmsh::model::occ::addLine(6, 2, acc++); 
    gmsh::model::occ::addLine(2, 1, acc++); 
    gmsh::model::occ::addLine(1, 5, acc++);

    // top box
    gmsh::model::occ::addLine(4, 3, acc++); 
    gmsh::model::occ::addLine(3, 7, acc++); 
    gmsh::model::occ::addLine(7, 8, acc++); 
    gmsh::model::occ::addLine(8, 4, acc++);

    acc_line = 9;

    for (int i = 0; i < polygons.size(); i++)
    {
        for(int j=1; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addLine(acc_line++, acc_line, acc++);
        }
        gmsh::model::occ::addLine(acc_line, acc_line-polygons[i].size()+1, acc++);
        ++acc_line;
    }
    
    acc = 1;
    int acc_loop = 1;
    // clipping box
    gmsh::model::occ::addCurveLoop({1, 2, 3, 4}, acc++);
    gmsh::model::occ::addCurveLoop({5, 6, 7, 8}, acc++);
    gmsh::model::occ::addCurveLoop({9, 10, 11, 12}, acc++);
    acc_loop = 13;
    // std::cout << "#polygons " << polygons.size() << std::endl;
    for (int i = 0; i < polygons.size(); i++)
    {
        std::vector<int> polygon_loop;
        for(int j=1; j < polygons[i].size()+1; j++)
            polygon_loop.push_back(acc_loop++);
        gmsh::model::occ::addCurveLoop(polygon_loop, acc++);
    }
    
    for (int i = 0; i < polygons.size()+3; i++)
    {
        gmsh::model::occ::addPlaneSurface({i+1});
    }
    

    std::vector<std::pair<int ,int>> poly_idx;
    for(int i=0; i<polygons.size(); ++i)
        poly_idx.push_back(std::make_pair(2, i+4));
    
    std::vector<std::pair<int, int>> ov;
    std::vector<std::vector<std::pair<int, int> > > ovv;
    gmsh::model::occ::cut({{2, 1}}, poly_idx, ov, ovv);

    std::vector<std::pair<int, int>> fuse_bottom_block;
    std::vector<std::vector<std::pair<int, int> > > _dummy;
    gmsh::model::occ::fragment({{2, 2}}, ov, fuse_bottom_block, _dummy);

    std::vector<std::pair<int, int>> fuse_top_block;
    std::vector<std::vector<std::pair<int, int> > > _dummy2;
    gmsh::model::occ::fragment({{2, 3}}, fuse_bottom_block, fuse_top_block, _dummy2);

    std::vector<std::pair<int, int> > ext;
    T depth = (pbc_corners[1] - pbc_corners[0]).norm() * 1.0;
	gmsh::model::occ::extrude(fuse_top_block, 0, 0, depth, ext);

    gmsh::model::mesh::field::add("Distance", 1);

    gmsh::model::mesh::field::add("Threshold", 2);
    gmsh::model::mesh::field::setNumber(2, "InField", 1);
    // gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.2);
    // gmsh::model::mesh::field::setNumber(2, "SizeMax", 1.5);
    gmsh::model::mesh::field::setNumber(2, "SizeMin", 1.0);
    gmsh::model::mesh::field::setNumber(2, "SizeMax", 5.0);
    gmsh::model::mesh::field::setAsBackgroundMesh(2);

    gmsh::model::occ::synchronize();

    gmsh::model::occ::synchronize();
    gmsh::model::mesh::generate(3);
    

    
    if (save_to_file)
    {
        gmsh::write(filename);
    }
    gmsh::finalize();
}

void Tiling2D::loadTilingStructureFromTxt(const std::string& filename,
    std::vector<std::vector<TV2>>& eigen_polygons,
    std::vector<TV2>& eigen_base, int n_unit)
{
    using namespace csk;
    using namespace std;
    using namespace glm;

    std::ifstream in(filename);
    int IH;
    std::string token;
    in >> token;
    in >> IH;
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    size_t num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    T pi;
    for (int i = 0; i < num_params; i++)
        in >> new_params[i];
    a_tiling.setParameters( new_params );
    in >> token;
    int num_edge_shape;
    in >> num_edge_shape;
    vector<vector<dvec2>> edges(a_tiling.numEdgeShapes());
    for (int i = 0; i < num_edge_shape; i++)
    {
        std::vector<dvec2> ej;
        for (int j = 0; j < 4; j++)
        {
            T x, y;
            in >> x >> y;
            ej.push_back(dvec2(x, y));
        }
        edges[i] = ej;
    }
    T angle;
    in >> angle;
    in.close();
    
    std::vector<dvec2> shape;
    getTilingShape(shape, a_tiling, edges);

    std::vector<std::vector<dvec2>> polygons_v;
    Vector<T, 4> transf; TV2 xy;
    getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, 4.0, 4.0, xy);
    
    // Vector<T, 8> periodic;
    
    // periodic.head<2>() = TV2(0,0);
    // periodic.segment<2>(2) = periodic.head<2>() + T(n_unit) * TV2(transf[0],transf[2]);
    // periodic.segment<2>(4) = periodic.head<2>() + T(n_unit) * TV2(transf[0],transf[2]) + T(n_unit) * TV2(transf[1],transf[3]);
    // periodic.segment<2>(6) = periodic.head<2>() + T(n_unit) * TV2(transf[1],transf[3]);

    
    // std::cout << periodic.transpose() << std::endl;
    // TV t1 = periodic.segment<2>(2);
    // TV t2 = periodic.segment<2>(6);
    // TV t1_unit = t1.normalized();
    // TV x_axis(1, 0);

    // T theta_to_x = -std::acos(t1_unit.dot(x_axis));
    // if (TV3(t1_unit[0], t1_unit[1], 0).cross(TV3(1, 0, 0)).dot(TV3(0, 0, 1)) > 0.0)
    //     theta_to_x *= -1.0;
    // TM2 R;
    // R << std::cos(theta_to_x), -std::sin(theta_to_x), std::sin(theta_to_x), std::cos(theta_to_x);
    TM R = rotMat(angle);
    // std::cout << R << std::endl;

    ClipperLib::Paths polygons(polygons_v.size());
    T mult = 1e12;
    for(int i=0; i<polygons_v.size(); ++i)
    {
        for(int j=0; j<polygons_v[i].size(); ++j)
        {
            // TV curr = R * TV(polygons_v[i][j][0]-xy[0], polygons_v[i][j][1]-xy[1]);
            // polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
            TV curr = TV(polygons_v[i][j][0]-xy[0], polygons_v[i][j][1]-xy[1]);
            polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
        }
    }
    // periodic.segment<2>(2) = R * periodic.segment<2>(2);
    // periodic.segment<2>(6) = TV(0, 1) * periodic.segment<2>(2).norm();
    // periodic[4] = periodic[2]; periodic[5] = periodic[7];

    // T dx = std::abs(periodic[2]);
    // TV shift = TV(0.1 * dx, 0.0);
    // for (int i = 0; i < 4; i++)
    //     periodic.segment<2>(i * 2) += shift;

    T distance = -1.5;

    ClipperLib::ClipperOffset c;
    ClipperLib::Paths final_shape;
    c.AddPaths(polygons, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);
    c.Execute(final_shape, distance*mult);
    shapeToPolygon(final_shape, eigen_polygons, mult);

    T min_x = 1e10, min_y = 1e10, max_x = -1e10, max_y = -1e10;
    for (auto polygon : eigen_polygons)
        for (auto pt : polygon)
        {
            min_x = std::min(min_x, pt[0]); min_y = std::min(min_y, pt[1]);
            max_x = std::max(max_x, pt[0]); max_y = std::max(max_y, pt[1]);
        }
    T dx = max_x - min_x;
    T dy = max_y - min_y;
    T scale = 0.35;
    Vector<T, 8> periodic;
    periodic << min_x + scale * dx, min_y + scale * dy, max_x - scale * dx, 
                min_y + scale * dy, max_x - scale * dx, max_y - scale * dy, 
                min_x + scale * dx, max_y - scale * dy;
    periodic.segment<2>(0) = R * periodic.segment<2>(0);
    periodic.segment<2>(2) = R * periodic.segment<2>(2);
    periodic.segment<2>(4) = R * periodic.segment<2>(4);
    periodic.segment<2>(6) = R * periodic.segment<2>(6);
    // saveClip(final_shape, periodic, mult, "tiling_unit_clip_in_x.obj", false);
    // std::cout << "CLIPPER done" << std::endl;
    // saveClip(final_shape, periodic, mult, "tiling_unit_clip_in_x.obj", true);
    // std::exit(0);
    periodicToBase(periodic, eigen_base);
}

void Tiling2D::sampleRegion(int IH, 
        std::vector<std::vector<TV2>>& eigen_polygons,
        std::vector<TV2>& eigen_base, 
        const std::vector<T>& params,
        const Vector<T, 4>& eij, const std::string& filename, T unit)
{
    using namespace csk;
    using namespace std;
    using namespace glm;

    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    size_t num_params = a_tiling.numParameters();
    std::ofstream out(filename);
    out << "IH " << IH << std::endl;
    out << "num_params " << int(num_params) << " ";
    T new_params[ num_params ];
    
    a_tiling.getParameters( new_params );
    // Change a parameter
    for( size_t idx = 0; idx < a_tiling.numParameters(); ++idx ) 
    {
        new_params[idx] = params[idx];
        out << params[idx] << " ";
    }
    out << std::endl;
    a_tiling.setParameters( new_params );

    vector<vector<dvec2>> edges(a_tiling.numEdgeShapes());
    out << "numEdgeShapes " << int(a_tiling.numEdgeShapes()) << " ";
    // Generate some random edge shapes.
    for( U8 idx = 0; idx < a_tiling.numEdgeShapes(); ++idx ) {
        vector<dvec2> ej;

        ej.push_back( dvec2( 0, 0.0 ) );
        ej.push_back( dvec2( eij[0], eij[1] ) );
        ej.push_back( dvec2( eij[2], eij[3] ) );
        ej.push_back( dvec2( 1.0, 0.0 ) );
        
        // Now, depending on the edge shape class, enforce symmetry 
        // constraints on edges.
        switch( a_tiling.getEdgeShape( idx ) ) {
        case J: 
            break;
        case U:
            ej[2].x = 1.0 - ej[1].x;
            ej[2].y = ej[1].y;
            break;
        case S:
            ej[2].x = 1.0 - ej[1].x;
            ej[2].y = -ej[1].y;
            break;
        case I:
            ej[1].y = 0.0;
            ej[2].y = 0.0;
            break;
        }
        edges[idx] = ej;
        for (dvec2 e : ej)
        {
            out << e[0] << " " << e[1] << " ";
        }
        out << std::endl;
    }
    out.close();
    
    std::vector<dvec2> shape;
    getTilingShape(shape, a_tiling, edges);


    std::vector<std::vector<dvec2>> polygons_v;


    Vector<T, 4> transf; TV2 xy;
    getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, unit, unit, xy);

    T min_y=10000, max_y=-10000, min_x=10000, max_x=-10000;
    for(int i=0; i<polygons_v.size(); ++i)
    {
        for(int j=0; j<polygons_v[i].size(); ++j)
        {
            max_x = std::max(max_x, polygons_v[i][j][0]);
            min_x = std::min(min_x, polygons_v[i][j][0]);
            min_y = std::min(min_y, polygons_v[i][j][1]);
            max_y = std::max(max_y, polygons_v[i][j][1]);
        }
    }
    T dx = max_x - min_x, dy = max_y - min_y;
    
    TM R = TM::Zero();
    R.row(0) = TV(0, -1);
    R.row(1) = TV(1, 0);
    // R.setIdentity();

    for(int i=0; i<polygons_v.size(); ++i)
    {
        for(int j=0; j<polygons_v[i].size(); ++j)
        {
            polygons_v[i][j][0] = polygons_v[i][j][0] - min_x - 0.5 * dx;
            polygons_v[i][j][1] = polygons_v[i][j][1] - min_y - 0.5 * dy;
        }
    }

    ClipperLib::Paths polygons(polygons_v.size());
    T mult = 1e12;
    for(int i=0; i<polygons_v.size(); ++i)
    {
        for(int j=0; j<polygons_v[i].size(); ++j)
        {
            TV curr = TV(polygons_v[i][j][0], polygons_v[i][j][1]);
            polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
        }
    }

    T distance = -1.5; //4 for silicone

    ClipperLib::ClipperOffset c;
    ClipperLib::Paths final_shape;
    c.AddPaths(polygons, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);
    c.Execute(final_shape, distance*mult);

    shapeToPolygon(final_shape, eigen_polygons, mult);

    
    // min_y=10000; max_y=-10000; min_x=10000; max_x=-10000;
    for (auto& polygon : eigen_polygons)
    {
        for(TV2& vtx : polygon)
        {
            if (IH == 0)
                vtx *= 0.014; // IH01
            else if (IH == 46)
            {
                vtx *= 0.012; // IH50
                vtx = R * vtx;
            }
        }
    }
    // dx = max_x - min_x, dy = max_y - min_y;
    // std::cout << dx << " " << dy << std::endl;
    // std::getchar();

    eigen_base.resize(0);
    
    T box_length = 2.5;
    eigen_base.push_back(TV2(-box_length, -box_length));
    eigen_base.push_back(TV2(box_length, -box_length));
    eigen_base.push_back(TV2(box_length, box_length));
    eigen_base.push_back(TV2(-box_length, box_length));

}



void Tiling2D::sampleOneFamilyWithOrientation(int IH, T angle, int n_unit, 
        T height, std::vector<std::vector<TV2>>& eigen_polygons,
        std::vector<TV2>& eigen_base, const std::vector<T>& params,
        const Vector<T, 4>& eij, const std::string& filename)
{
    using namespace csk;
    using namespace std;
    using namespace glm;

    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    size_t num_params = a_tiling.numParameters();
    std::ofstream out(filename);
    out << "IH " << IH << std::endl;
    out << "num_params " << int(num_params) << " ";
    T new_params[ num_params ];
    
    a_tiling.getParameters( new_params );
    // Change a parameter
    for( size_t idx = 0; idx < a_tiling.numParameters(); ++idx ) 
    {
        new_params[idx] = params[idx];
        out << params[idx] << " ";
    }
    out << std::endl;
    a_tiling.setParameters( new_params );

    out << "numEdgeShapes " << int(a_tiling.numEdgeShapes()) << " ";

    std::vector<std::vector<dvec2>> edges(a_tiling.numEdgeShapes());
    getTilingEdges(a_tiling, eij, edges);
    for (auto ej : edges)
    {
        for (dvec2 e : ej)
        {
            out << e[0] << " " << e[1] << " ";
        }
        out << std::endl;
    }
    out.close();
    
    std::vector<dvec2> shape;
    getTilingShape(shape, a_tiling, edges);

    std::vector<std::vector<dvec2>> polygons_v;
    Vector<T, 4> transf; TV2 xy;
    getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, 5.0, 5.0, xy);

    Vector<T, 8> periodic; periodic.setZero();
    periodic.segment<2>(2) = TV2(transf[0],transf[2]);
    periodic.segment<2>(4) = TV2(transf[0],transf[2]) + TV2(transf[1],transf[3]);
    periodic.segment<2>(6) = TV2(transf[1],transf[3]);

    
    TV t1 = periodic.segment<2>(2);
    TV t2 = periodic.segment<2>(6);

    TV t1_unit = t1.normalized();
    TV x_axis(1, 0);

    T theta_to_x = -std::acos(t1_unit.dot(x_axis));
    if (TV3(t1_unit[0], t1_unit[1], 0).cross(TV3(1, 0, 0)).dot(TV3(0, 0, 1)) > 0.0)
        theta_to_x *= -1.0;
    TM R = rotMat(theta_to_x);

    
    TM R2 = rotMat(angle);

    periodic.segment<2>(2) = R2 * R * periodic.segment<2>(2);
    periodic.segment<2>(4) = R2 * R * periodic.segment<2>(4);
    periodic.segment<2>(6) = R2 * R * periodic.segment<2>(6);
    
    t1 = periodic.segment<2>(2);
    t2 = periodic.segment<2>(6);

    int n = 1;
    for (; n < 100; n++)
    {
        TV a = t1 * T(n);
        TV b = a.dot(TV(1, 0)) * TV(1, 0);
        TV c = a - b;
        T beta = std::acos(t1.normalized().dot(t2.normalized()));
        T alpha = angle;
        T target = c.norm() / std::cos(beta - (M_PI/2.0 - alpha));
        T div_check = target / t2.norm();
        T delta = std::abs(std::round(div_check) - div_check);
        std::cout << "n " << n << " delta " << delta << " round " << std::round(div_check) << " true " <<  div_check << std::endl;
        if (delta < 1e-6)
        {
            break;
        }
    }
    std::cout << n << std::endl;
    // std::getchar();
    periodic.segment<2>(2) = t1.dot(TV(1, 0)) * TV(1, 0) * n * n_unit;
    periodic[6] = 0; periodic[7] = height;
    periodic[4] = periodic[2]; periodic[5] = periodic[7]; 

    ClipperLib::Paths polygons(polygons_v.size());
    T mult = 10000000.0;
    for(int i=0; i < polygons_v.size(); ++i)
    {
        for(int j=0; j < polygons_v[i].size(); ++j)
        {
            TV curr = R2 * R * TV(polygons_v[i][j][0]-xy[0], polygons_v[i][j][1]-xy[1]);
            // TV curr = TV(polygons_v[i][j][0]-xy[0], polygons_v[i][j][1]-xy[1]);
            polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
        }
    }
      

    T distance = -2.0;

    ClipperLib::ClipperOffset c;
    ClipperLib::Paths final_shape;
    c.AddPaths(polygons, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);
    c.Execute(final_shape, distance*mult);

    saveClip(final_shape, periodic, mult, "tiling_unit_clip_in_x.obj", true);
    
    shapeToPolygon(final_shape, eigen_polygons, mult);
    periodicToBase(periodic, eigen_base);
}



void Tiling2D::generateOneStructureSquarePatch(int IH, const std::vector<T>& params)
{
    data_folder = "./";
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    size_t num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    for( size_t idx = 0; idx < a_tiling.numParameters(); ++idx ) 
    {
        new_params[idx] = params[idx];
    }
    a_tiling.setParameters( new_params );

    std::vector<std::vector<dvec2>> edges(a_tiling.numEdgeShapes());
    Vector<T, 4> eij;
    eij << 0.25, 0., 0.75, 0.;
    getTilingEdges(a_tiling, eij, edges);
    std::vector<dvec2> shape;
    getTilingShape(shape, a_tiling, edges);

    
    std::vector<std::vector<dvec2>> polygons_v;
    Vector<T, 4> transf; TV2 xy;
    // getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, 16.0, 16.0, xy);
    // getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, 6.0, 6.0, xy);
    getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, 5.0, 5.0, xy);
    // getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, 8.0, 8.0, xy);

    ClipperLib::Paths polygons(polygons_v.size());
    T mult = 1e12;
    for(int i=0; i<polygons_v.size(); ++i)
    {
        for(int j=0; j<polygons_v[i].size(); ++j)
        {
            polygons[i] << ClipperLib::IntPoint((polygons_v[i][j][0]-xy[0])*mult, 
                (polygons_v[i][j][1]-xy[1])*mult);
        }
    }
    
    T distance = -1.5;
    ClipperLib::Paths final_shape;

    ClipperLib::ClipperOffset c;
    c.AddPaths(polygons, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);
    
    c.Execute(final_shape, distance*mult);
    // saveClip(final_shape, periodic, mult, "tiling_unit_clip_in_x.obj", true);
    std::vector<std::vector<TV2>> eigen_polygons;
    std::vector<TV2> eigen_base;
    shapeToPolygon(final_shape, eigen_polygons, mult);
    T min_x = 1e10, min_y = 1e10, max_x = -1e10, max_y = -1e10;
    for (auto polygon : eigen_polygons)
        for (auto pt : polygon)
        {
            min_x = std::min(min_x, pt[0]); min_y = std::min(min_y, pt[1]);
            max_x = std::max(max_x, pt[0]); max_y = std::max(max_y, pt[1]);
        }
    T dx = max_x - min_x;
    T dy = max_y - min_y;
    // T scale_x = 0.08, scale_y = 0.4;
    T scale_x = 0.25, scale_y = 0.25;
    Vector<T, 8> periodic;
    periodic << min_x + scale_x * dx, min_y + scale_y * dy, max_x - scale_x * dx, 
                min_y + scale_y * dy, max_x - scale_x * dx, max_y - scale_y * dy, 
                min_x + scale_x * dx, max_y - scale_y * dy;
    
    // periodic *= 2.0;
    // for (auto& polygon : eigen_polygons)
    //     for (TV2& vtx : polygon)
    //         vtx *= 2.0;

    // periodic << min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y;
    periodicToBase(periodic, eigen_base);
    generateNonPeriodicMesh(eigen_polygons, eigen_base, true, data_folder + "a_structure");
    initializeSimulationDataFromFiles(data_folder + "a_structure.vtk", PBC_None);
    
}

void Tiling2D::generateToyExampleStructure(const std::vector<T>& params,
    const std::string& result_folder)
{
    std::vector<std::vector<TV2>> eigen_polygons;
    std::vector<TV2> eigen_base;
    std::string filename = result_folder + "structure.txt";
    // std::ofstream out(filename);
    TV center = TV(1, 1);
    
    T scale = 1.0;
    std::vector<TV> corners = {center + TV(-1, -1) * scale, 
        center +TV(1, -1) * scale, 
        center + TV(1, 1) * scale, 
        center + TV(-1, 1) * scale};

    T d_domain = (corners[2] - corners[0]).norm();

    std::vector<TV> inner_square(4);
    for (int i = 0; i < 4; i++)
        inner_square[i] = center + std::abs(params[0]) * (corners[i] - center);
    
    
    std::vector<std::vector<TV>> polygons_v(5);
    polygons_v[0] = inner_square;
    

    polygons_v[1] = {inner_square[1] - TV(2, 0) * scale,
                    corners[0],
                    inner_square[0], inner_square[3], 
                    corners[3],
                    inner_square[2] - TV(2, 0) * scale};

    polygons_v[2] = {corners[3],
                    inner_square[3], inner_square[2], 
                    corners[2],
                    inner_square[1] + TV(0, 2) * scale,
                    inner_square[0] + TV(0, 2) * scale
                    };

    polygons_v[3] = {
        corners[2], inner_square[2], inner_square[1], corners[1], 
        inner_square[0] + TV(2, 0) * scale,
        inner_square[3] + TV(2, 0) * scale
    };

    polygons_v[4] = {
        corners[0], 
        inner_square[3] - TV(0, 2) * scale,
        inner_square[2] - TV(0, 2) * scale,
        corners[1], inner_square[1], inner_square[0]
    };

    Vector<T, 8> periodic;
    for (int i = 0; i < 4; i++)
        periodic.segment<2>(i * 2) = corners[i];


    ClipperLib::Paths polygons(polygons_v.size());
    T mult = 1e12;
    
    for(int i=0; i<polygons_v.size(); ++i)
    {
        for(int j=0; j<polygons_v[i].size(); ++j)
        {
            TV curr = TV(polygons_v[i][j][0], polygons_v[i][j][1]);
    
            polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
            // std::cout << polygons[i] << std::endl;
        }
        TV curr = TV(polygons_v[i][0][0], polygons_v[i][0][1]);
    
        polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
    }
    
    
    T distance = -0.015 * d_domain;
    ClipperLib::Paths final_shape;
    ClipperLib::ClipperOffset c;
    c.AddPaths(polygons, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);
    
    c.Execute(final_shape, distance*mult);
    shapeToPolygon(final_shape, eigen_polygons, mult);
    periodicToBase(periodic, eigen_base);
    
    generatePeriodicMeshHardCodeResolution(eigen_polygons, eigen_base, true, result_folder + "structure");
    
}

void Tiling2D::generateToyExample(T param)
{
    std::vector<std::vector<TV2>> eigen_polygons;
    std::vector<TV2> eigen_base;
    data_folder = "./";
    std::string filename = data_folder + "a_structure.txt";
    // std::ofstream out(filename);
    TV center = TV(1, 1);
    
    T scale = 1.0;
    std::vector<TV> corners = {center + TV(-1, -1) * scale, 
        center +TV(1, -1) * scale, 
        center + TV(1, 1) * scale, 
        center + TV(-1, 1) * scale};

    T d_domain = (corners[2] - corners[0]).norm();

    std::vector<TV> inner_square(4);
    for (int i = 0; i < 4; i++)
        inner_square[i] = center + std::abs(param) * (corners[i] - center);
    
    
    std::vector<std::vector<TV>> polygons_v(5);
    polygons_v[0] = inner_square;
    

    polygons_v[1] = {inner_square[1] - TV(2, 0) * scale,
                    corners[0],
                    inner_square[0], inner_square[3], 
                    corners[3],
                    inner_square[2] - TV(2, 0) * scale};

    polygons_v[2] = {corners[3],
                    inner_square[3], inner_square[2], 
                    corners[2],
                    inner_square[1] + TV(0, 2) * scale,
                    inner_square[0] + TV(0, 2) * scale
                    };

    polygons_v[3] = {
        corners[2], inner_square[2], inner_square[1], corners[1], 
        inner_square[0] + TV(2, 0) * scale,
        inner_square[3] + TV(2, 0) * scale
    };

    polygons_v[4] = {
        corners[0], 
        inner_square[3] - TV(0, 2) * scale,
        inner_square[2] - TV(0, 2) * scale,
        corners[1], inner_square[1], inner_square[0]
    };

    Vector<T, 8> periodic;
    for (int i = 0; i < 4; i++)
        periodic.segment<2>(i * 2) = corners[i];


    ClipperLib::Paths polygons(polygons_v.size());
    T mult = 1e12;
    
    for(int i=0; i<polygons_v.size(); ++i)
    {
        for(int j=0; j<polygons_v[i].size(); ++j)
        {
            TV curr = TV(polygons_v[i][j][0], polygons_v[i][j][1]);
    
            polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
            // std::cout << polygons[i] << std::endl;
        }
        TV curr = TV(polygons_v[i][0][0], polygons_v[i][0][1]);
    
        polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
    }
    
    
    T distance = -0.015 * d_domain;
    ClipperLib::Paths final_shape;
    ClipperLib::ClipperOffset c;
    c.AddPaths(polygons, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);
    
    c.Execute(final_shape, distance*mult);
    shapeToPolygon(final_shape, eigen_polygons, mult);
    periodicToBase(periodic, eigen_base);

    generatePeriodicMeshHardCodeResolution(eigen_polygons, eigen_base, true, data_folder + "a_structure");
    solver.pbc_translation_file = data_folder + "a_structure_translation.txt";
    initializeSimulationDataFromFiles(data_folder + "a_structure.vtk", PBC_XY);
}

void Tiling2D::generateOnePerodicUnit()
{
    
    data_folder = "./";
    
    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    // int tiling_idx = 19;
    int tiling_idx = 46; // IH50
    // int tiling_idx = 0;
    // int tiling_idx = 60;
    // int tiling_idx = 26;
    // int tiling_idx = 20;
    // int tiling_idx = 27;

    // scale the unit by area
    T unit = 5.0;
    if (tiling_idx == 0 || tiling_idx == 19)
        unit = 5.0;
    else if (tiling_idx == 26)
        unit = 6.0;
    else if (tiling_idx == 27 )
        unit = 10.0;
    else if (tiling_idx == 60)
        unit = 6.0;
    else if (tiling_idx == 46)
        unit = 3.0;
    else if (tiling_idx == 20)
        unit = 7.0;
    csk::IsohedralTiling a_tiling( csk::tiling_types[ tiling_idx ] );
    int num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    std::vector<T> params(num_params);
    for (int j = 0; j < num_params;j ++)
        params[j] = new_params[j];
    
    
    for (int k = 0; k < num_params - 1; k++)
        std::cout << params[k] << ", ";
    std::cout << params[num_params - 1] << std::endl;
    // std::cout << params[0] << " " << params[1] << std::endl;
    // params = {0.115, 0.765};
    // params[0] += 1e-4;

    // params = {0.093};
    
    Vector<T, 4> cubic_weights;
    cubic_weights << 0.25, 0., 0.75, 0.;
    bool non_ixn_unit = fetchUnitCellFromOneFamily(tiling_idx, 2, polygons, pbc_corners, params, 
        cubic_weights, data_folder + "a_structure.txt", unit);
    if (!non_ixn_unit)
    {
        std::cout << "self intersecting unit" << std::endl;
        return;
    }
    
    generatePeriodicMesh(polygons, pbc_corners, true, data_folder + "a_structure");
    // generateHomogenousMesh(polygons, pbc_corners, true, data_folder + "a_structure");
    // generateSandwichMeshPerodicInX(polygons, pbc_corners, true, 
    //     data_folder + "a_structure.vtk");   
    solver.pbc_translation_file = data_folder + "a_structure_translation.txt";
    initializeSimulationDataFromFiles(data_folder + "a_structure.vtk", PBC_XY);
}

void Tiling2D::generateOneStructureWithRotation()
{
    int IH = 0;
    
    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    Vector<T, 4> cubic_weights;
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    int num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    std::vector<T> params(num_params);
    for (int j = 0; j < num_params;j ++)
        params[j] = new_params[j];
    T angle = M_PI / 8.0;
    cubic_weights << 0.25, 0, 0.75, 0;
    sampleOneFamilyWithOrientation(IH, angle, 2, 50, polygons, pbc_corners, params,
        cubic_weights, data_folder + "a_structure.txt");
    generateSandwichMeshPerodicInX(polygons, pbc_corners, true, 
        data_folder + "a_structure.vtk");   
    initializeSimulationDataFromFiles(data_folder + "a_structure.vtk", PBC_X);
}


void Tiling2D::generateOneNonperiodicStructure()
{
    data_folder = "./";
    
    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    Vector<T, 4> cubic_weights;
    int tiling_idx = 0;
    csk::IsohedralTiling a_tiling( csk::tiling_types[ tiling_idx ] );
    int num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    std::vector<T> params(num_params);
    for (int j = 0; j < num_params;j ++)
        params[j] = new_params[j];
    std::vector<T> diff_params = params;
    for (int k = 0; k < num_params; k++)
    {
        T rand_params = 0.1 * (zeta() * 2.0 - 1.0);
        diff_params[k] = std::max(std::min(params[k] + rand_params, 0.92), 0.08);
    }
    // cubic_weights << 0.25, 0.25, 0.75, 0.75;
    T x0_rand = zeta(), y0_rand = zeta();
    // cubic_weights << x0_rand, y0_rand, x0_rand + (1.0 - x0_rand) * zeta(), y0_rand + (1.0 - y0_rand) * zeta();
    // cubic_weights << 0.46529, 0.134614, 0.787798, 0.638062;
    cubic_weights << 0.25, 0, 0.75, 0;
    // sampleSandwichFromOneFamilyFromDiffParamsDilation(tiling_idx, polygons, pbc_corners, params, cubic_weights, 
    //     data_folder + "a_structure.txt");
    sampleRegion(tiling_idx, polygons, pbc_corners, diff_params, cubic_weights, 
        data_folder + "a_structure.txt");
    
    generateSandwichMeshNonPeridoic(polygons, pbc_corners, true, 
        data_folder + "a_structure.vtk");   
    initializeSimulationDataFromFiles(data_folder + "a_structure.vtk", PBC_None);
}


bool Tiling2D::fetchUnitCellFromOneFamily(int IH, int n_unit,
    std::vector<std::vector<TV2>>& eigen_polygons,
    std::vector<TV2>& eigen_base, 
    const std::vector<T>& params,
    const Vector<T, 4>& eij, const std::string& filename,
    T unit, T angle)
{
    using namespace csk;
    using namespace std;
    using namespace glm;

    std::ofstream out(filename);
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    out << "IH " << IH << std::endl;
    size_t num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    // Change a parameter
    for( size_t idx = 0; idx < a_tiling.numParameters(); ++idx ) 
    {
        new_params[idx] = params[idx];
        out << params[idx] << " ";
    }
    out << std::endl;
    a_tiling.setParameters( new_params );

    out << "numEdgeShapes " << int(a_tiling.numEdgeShapes()) << " ";
    std::vector<std::vector<dvec2>> edges(a_tiling.numEdgeShapes());
    getTilingEdges(a_tiling, eij, edges);
    for (auto ej : edges)
    {
        for (dvec2 e : ej)
            out << e[0] << " " << e[1] << " ";
        out << std::endl;
    }
    out << angle << std::endl;
    out.close();

    std::vector<dvec2> shape;
    getTilingShape(shape, a_tiling, edges);

    std::vector<std::vector<dvec2>> polygons_v;

    Vector<T, 4> transf; TV2 xy;
    getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, unit, unit, xy);
    // getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, 8.0, 8.0, xy);
    // getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, 5.0, 5.0, xy);
    // getTranslationUnitPolygon(polygons_v, shape, a_tiling, transf, 4.0, 4.0, xy);


    csk::IsohedralTiling default_tiling( csk::tiling_types[ IH ] );

    int n_tiling_vtx = a_tiling.numVertices();
    Eigen::MatrixXd ipc_vtx(n_tiling_vtx, 2);
    Eigen::MatrixXd ipc_vtx_rest(n_tiling_vtx, 2);
    Eigen::MatrixXi ipc_edges(n_tiling_vtx, 2), ipc_faces;
    for (int i = 0; i < a_tiling.numVertices(); i++)
    {
        glm::vec2 vtx = a_tiling.getVertex(i);
        glm::vec2 vtx_rest = default_tiling.getVertex(i);
        ipc_vtx(i, 0) = vtx[0]; ipc_vtx(i, 1) = vtx[1];
        ipc_vtx_rest(i, 0) = vtx_rest[0]; ipc_vtx_rest(i, 1) = vtx_rest[1];
        ipc_edges(i, 0) = i; ipc_edges(i, 1) = (i+1)%n_tiling_vtx; 
    }

    auto signedArea = [&](const Eigen::MatrixXd& vtx) -> T
    {
        T area = 0.0;
        for (int i = 0; i < n_tiling_vtx; i++)
        {
            int vi = ipc_edges(i, 0), vj = ipc_edges(i, 1);
            TV xi = vtx.row(vi), xj = vtx.row(vj);
            area += 0.5 * (xi[0] * xj[1] - xi[1] * xj[0]);
        }
        
        return area;
    };

    T area_rest = std::abs(signedArea(ipc_vtx_rest));
    T area = std::abs(signedArea(ipc_vtx));
    // std::cout << std::abs(area/area_rest) * 100.0 << "%" << std::endl;
    if (std::abs(area/area_rest) < 0.15)
        return false;

    if (ipc::has_intersections(ipc_vtx, ipc_edges, ipc_faces))
        return false;
    
    // std::ofstream obj_out("tiling_unit.obj");
    // for (int i = 0; i < a_tiling.numVertices(); i++)
    // {
    //     glm::vec2 vtx = a_tiling.getVertex(i);
    //     obj_out << "v " << vtx[0] << " " << vtx[1] << " 0.0" << std::endl;
    // }
    // for (int i = 0; i < a_tiling.numVertices(); i++)
    // {
    //     obj_out << "l " << i+1 << " " << (i+1)%a_tiling.numVertices() + 1 << std::endl;
    // }
    // obj_out.close();
    // std::exit(0);
    

    Vector<T, 8> periodic;
    periodic.head<2>() = TV2(0,0);
    periodic.segment<2>(2) = periodic.head<2>() + T(n_unit) * TV2(transf[0],transf[2]);
    periodic.segment<2>(4) = periodic.head<2>() + T(n_unit) * TV2(transf[0],transf[2]) + T(n_unit) * TV2(transf[1],transf[3]);
    periodic.segment<2>(6) = periodic.head<2>() + T(n_unit) * TV2(transf[1],transf[3]);

    // TM R = rotMat(angle);
    TM R = TM::Identity();

    ClipperLib::Paths polygons(polygons_v.size());
    T mult = 1e12;
    for(int i=0; i<polygons_v.size(); ++i)
    {
        for(int j=0; j<polygons_v[i].size(); ++j)
        {
            TV curr = R * TV(polygons_v[i][j][0]-xy[0], polygons_v[i][j][1]-xy[1]);
            polygons[i] << ClipperLib::IntPoint(curr[0]*mult, curr[1]*mult);
            // polygons[i] << ClipperLib::IntPoint((polygons_v[i][j][0]-xy[0])*mult, 
            //     (polygons_v[i][j][1]-xy[1])*mult);
            // std::cout << " " << polygons[i][j];
        }
        // break;
        // std::cout << std::endl;
    }
    // std::ofstream polygon_obj("polygon_obj.obj");
    // // for (auto polygon : polygons)
    // for (int i = 0; i < polygons.size(); i++)
    // {
    //     auto polygon = polygons[i];
        
    //     for (auto vtx : polygon)
    //         polygon_obj << "v " << vtx.X << " " << vtx.Y << " 0" << std::endl;
        
    // }
    // polygon_obj.close();

    periodic.segment<2>(2) = R * periodic.segment<2>(2);
    periodic.segment<2>(4) = R * periodic.segment<2>(4);
    periodic.segment<2>(6) = R * periodic.segment<2>(6);
    
    T distance = -1.5;
    ClipperLib::Paths final_shape;

    ClipperLib::ClipperOffset c;
    c.AddPaths(polygons, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);
    
    c.Execute(final_shape, distance*mult);
    // saveClip(final_shape, periodic, mult, "tiling_unit_clip_in_x.obj", true);
    shapeToPolygon(final_shape, eigen_polygons, mult);
    periodicToBase(periodic, eigen_base);
    return true;
}

void Tiling2D::extrudeToMesh(const std::string& tiling_param, const std::string& mesh3d, int n_unit)
{
    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    loadTilingStructureFromTxt(tiling_param, polygons, pbc_corners, n_unit);
    generate3DSandwichMesh(polygons, pbc_corners, true, mesh3d);
}

void Tiling2D::generateSandwichMeshNonPeridoic(std::vector<std::vector<TV2>>& polygons, 
        std::vector<TV2>& pbc_corners, bool save_to_file, std::string filename)
{
    T eps = 1e-6;
    gmsh::initialize();

    gmsh::model::add("tiling");
    gmsh::logger::start();

    gmsh::option::setNumber("Geometry.ToleranceBoolean", eps);
    gmsh::option::setNumber("Geometry.Tolerance", eps);
    if (solver.use_quadratic_triangle)
        gmsh::option::setNumber("Mesh.ElementOrder", 2);
    else
        gmsh::option::setNumber("Mesh.ElementOrder", 1);

    // disable set resolution from point option
    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

 
    //Points
    int acc = 1;
    // clamping box 1 2 3 4
    for (int i = 0; i < pbc_corners.size(); ++i)
		gmsh::model::occ::addPoint(pbc_corners[i][0], pbc_corners[i][1], 0, 2, acc++);
    
    // sandwich boxes bottom 5 6 7 8
    // T dx = 0.05 * (pbc_corners[1][0] - pbc_corners[0][0]);
    T dy = for_printing ? 0.5 : handle_width; // 5 mm
    T dx = 0.5;
    gmsh::model::occ::addPoint(pbc_corners[0][0]-dx,  pbc_corners[0][1], 0, 2, acc++);
    gmsh::model::occ::addPoint(pbc_corners[0][0]-dx,  pbc_corners[0][1] - dy, 0, 2, acc++);
    gmsh::model::occ::addPoint(pbc_corners[1][0]+dx, pbc_corners[1][1] - dy, 0, 2, acc++);
    gmsh::model::occ::addPoint(pbc_corners[1][0]+dx, pbc_corners[1][1], 0, 2, acc++);
    
    // sandwich boxes top 9 10 11 12
    gmsh::model::occ::addPoint(pbc_corners[2][0]+dx, pbc_corners[2][1], 0, 2, acc++);
    gmsh::model::occ::addPoint(pbc_corners[2][0]+dx, pbc_corners[2][1] + dy, 0, 2, acc++);
    gmsh::model::occ::addPoint(pbc_corners[3][0]-dx, pbc_corners[3][1] + dy, 0, 2, acc++);
    gmsh::model::occ::addPoint(pbc_corners[3][0]-dx, pbc_corners[3][1], 0, 2, acc++);

    // inner lattice
    for (int i=0; i<polygons.size(); ++i)
    {
        for(int j=0; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addPoint(polygons[i][j][0], polygons[i][j][1], 0, 2, acc++);
        }
    }
    
    //Lines
    acc = 1;

    std::vector<double> poly_lines;
    
    // add clipping box
    gmsh::model::occ::addLine(1, 2, acc++); 
    poly_lines.push_back(acc);
    gmsh::model::occ::addLine(2, 3, acc++); 
    poly_lines.push_back(acc);
    gmsh::model::occ::addLine(3, 4, acc++); 
    poly_lines.push_back(acc);
    gmsh::model::occ::addLine(4, 1, acc++);
    poly_lines.push_back(acc);

    std::cout << "add clipping box" << std::endl;
    
    // bottom box
    gmsh::model::occ::addLine(5, 6, acc++); 
    poly_lines.push_back(acc);
    gmsh::model::occ::addLine(6, 7, acc++); 
    poly_lines.push_back(acc);
    gmsh::model::occ::addLine(7, 8, acc++); 
    poly_lines.push_back(acc);
    gmsh::model::occ::addLine(8, 5, acc++);
    poly_lines.push_back(acc);

    std::cout << "add bottom box" << std::endl;

    // top box
    gmsh::model::occ::addLine(9, 10, acc++); 
    poly_lines.push_back(acc);
    gmsh::model::occ::addLine(10, 11, acc++); 
    poly_lines.push_back(acc);
    gmsh::model::occ::addLine(11, 12, acc++); 
    poly_lines.push_back(acc);
    gmsh::model::occ::addLine(12, 9, acc++);
    poly_lines.push_back(acc);

    std::cout << "add top box" << std::endl;

    int acc_line = 13; // start with the thirdteenth line, because we added 12

    // for (int i = 0; i < polygons.size(); i++)
    // {
    //     for(int j=1; j<polygons[i].size(); ++j)
    //     {
    //         gmsh::model::occ::addLine(acc_line++, acc_line, acc++);
    //     }
    //     gmsh::model::occ::addLine(acc_line, acc_line-polygons[i].size()+1, acc++);
    //     ++acc_line;
    // }




    auto eval_distance = [](const TV& p1, const TV& p2, const TV& q, T t)
    {
        return sqrt(pow((0.1e1 - t) * p1[0] + t * p2[0] - q[0], 0.2e1) + pow((0.1e1 - t) * p1[1] + t * p2[1] - q[1], 0.2e1));
    };

    auto closest_t_to_line = [](const TV& p1, const TV& p2, const TV& q){
        return (p1[0] * p1[0] + (-p2[0] - q[0]) * p1[0] + p1[1] * p1[1] + (-p2[1] - q[1]) * p1[1] + p2[0] * q[0] + p2[1] * q[1]) / (p1[0] * p1[0] - 2 * p1[0] * p2[0] + p1[1] * p1[1] - 2 * p1[1] * p2[1] + p2[0] * p2[0] + p2[1] * p2[1]);
    };


    TV p1 = pbc_corners[0];
    TV p2 = pbc_corners[1];
    TV p3 = pbc_corners[2];
    TV p4 = pbc_corners[3];

    
    for(int i=0; i<polygons.size(); ++i)
    {
        for(int j=1; j<polygons[i].size(); ++j)
        {
            TV q1 = polygons[i][j-1];
            TV q2 = polygons[i][j];

            T t_left1 = closest_t_to_line(p1, p4, q1);
            T t_right1 = closest_t_to_line(p2, p3, q1);
            T t_bottom1 = closest_t_to_line(p1, p2, q1);
            T t_top1 = closest_t_to_line(p4, p3, q1);

            T t_left2 = closest_t_to_line(p1, p4, q2);
            T t_right2 = closest_t_to_line(p2, p3, q2);
            T t_bottom2 = closest_t_to_line(p1, p2, q2);
            T t_top2 = closest_t_to_line(p4, p3, q2);

            bool pass1 = false;
            bool pass2 = false;

            if(eval_distance(p1, p4, q1, t_left1)<1e-5)
                pass1 = true;
            if(eval_distance(p2, p3, q1, t_right1)<1e-5)
                pass1 = true;
            if(eval_distance(p1, p2, q1, t_bottom1)<1e-5)
                pass1 = true;
            if(eval_distance(p4, p3, q1, t_top1)<1e-5)
                pass1 = true;

            if(eval_distance(p1, p4, q2, t_left2)<1e-5)
                pass2 = true;
            if(eval_distance(p2, p3, q2, t_right2)<1e-5)
                pass2 = true;
            if(eval_distance(p1, p2, q2, t_bottom2)<1e-5)
                pass2 = true;
            if(eval_distance(p4, p3, q2, t_top2)<1e-5)
                pass2 = true;

            if(!pass1 || !pass2)
                poly_lines.push_back(acc);
            gmsh::model::occ::addLine(acc_line++, acc_line, acc++);
        }

        TV q1 = polygons[i].back();
        TV q2 = polygons[i][0];

        T t_left1 = closest_t_to_line(p1, p4, q1);
        T t_right1 = closest_t_to_line(p2, p3, q1);
        T t_bottom1 = closest_t_to_line(p1, p2, q1);
        T t_top1 = closest_t_to_line(p4, p3, q1);

        T t_left2 = closest_t_to_line(p1, p4, q2);
        T t_right2 = closest_t_to_line(p2, p3, q2);
        T t_bottom2 = closest_t_to_line(p1, p2, q2);
        T t_top2 = closest_t_to_line(p4, p3, q2);

        bool pass1 = false;
        bool pass2 = false;

        if(eval_distance(p1, p4, q1, t_left1)<1e-5)
            pass1 = true;
        if(eval_distance(p2, p3, q1, t_right1)<1e-5)
            pass1 = true;
        if(eval_distance(p1, p2, q1, t_bottom1)<1e-5)
            pass1 = true;
        if(eval_distance(p4, p3, q1, t_top1)<1e-5)
            pass1 = true;

        if(eval_distance(p1, p4, q2, t_left2)<1e-5)
            pass2 = true;
        if(eval_distance(p2, p3, q2, t_right2)<1e-5)
            pass2 = true;
        if(eval_distance(p1, p2, q2, t_bottom2)<1e-5)
            pass2 = true;
        if(eval_distance(p4, p3, q2, t_top2)<1e-5)
            pass2 = true;

        if(!pass1 || !pass2)
            poly_lines.push_back(acc);
        gmsh::model::occ::addLine(acc_line, acc_line-polygons[i].size()+1, acc++);
        ++acc_line;
    }






    std::cout << "add line polygon" << std::endl;
    acc = 1;
    
    // clipping box
    gmsh::model::occ::addCurveLoop({1, 2, 3, 4}, acc++);
    std::cout << "addCurveLoop clipping box" << std::endl;
    gmsh::model::occ::addCurveLoop({5, 6, 7, 8}, acc++);
    std::cout << "addCurveLoop bottom box" << std::endl;
    gmsh::model::occ::addCurveLoop({9, 10, 11, 12}, acc++);
    std::cout << "addCurveLoop top box" << std::endl;
    int acc_loop = 13;
    
    for (int i = 0; i < polygons.size(); i++)
    {
        std::vector<int> polygon_loop;
        for(int j=1; j < polygons[i].size()+1; j++)
            polygon_loop.push_back(acc_loop++);
        gmsh::model::occ::addCurveLoop(polygon_loop, acc++);
    }
    
    for (int i = 0; i < polygons.size()+3; i++)
    {
        gmsh::model::occ::addPlaneSurface({i+1});
    }
    

    std::vector<std::pair<int ,int>> poly_idx;
    for(int i=0; i<polygons.size(); ++i)
        poly_idx.push_back(std::make_pair(2, i+4));
    
    std::vector<std::pair<int, int>> ov;
    std::vector<std::vector<std::pair<int, int> > > ovv;
    std::cout << "add geometry done" << std::endl;
    gmsh::model::occ::cut({{2, 1}}, poly_idx, ov, ovv);
    std::cout << "add cut box done" << std::endl;

    std::vector<std::pair<int, int>> fuse_bottom_block;
    std::vector<std::vector<std::pair<int, int> > > _dummy;
    gmsh::model::occ::fragment({{2, 2}}, ov, fuse_bottom_block, _dummy);
    std::cout << "add bottom box done" << std::endl;

    std::vector<std::pair<int, int>> fuse_top_block;
    std::vector<std::vector<std::pair<int, int> > > _dummy2;
    gmsh::model::occ::fragment({{2, 3}}, fuse_bottom_block, fuse_top_block, _dummy2);
    std::cout << "add top box done" << std::endl;
    

    // add holes
    std::vector<int> circle_tags;
    std::vector<std::vector<std::pair<int, int>>> holder(11);
    // unit is cm

    if (for_printing)
    {
        holder[0] = fuse_top_block;
        for (int i = 0; i < 5; i++)
        {
            int circle_i = gmsh::model::occ::addDisk(
                pbc_corners[0][0] + i * 1.25, pbc_corners[0][1] - 0.25, 0.0, 0.15, 0.15);
            gmsh::model::occ::cut(holder[i], {{2, circle_i}}, holder[i+1], ovv);
        }
        for (int i = 5; i < 10; i++)
        {
            int circle_i = gmsh::model::occ::addDisk(
                pbc_corners[2][0] - (i-5) * 1.25, pbc_corners[2][1] + 0.25, 0.0, 0.15, 0.15);
            gmsh::model::occ::cut(holder[i], {{2, circle_i}}, holder[i+1], ovv);
        }
    }
    

    gmsh::model::mesh::field::add("Distance", 1);

    gmsh::model::mesh::field::setNumbers(1, "CurvesList", poly_lines);

    gmsh::model::mesh::field::add("Threshold", 2);
    gmsh::model::mesh::field::setNumber(2, "InField", 1);
    
    // gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.02);
    // gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.04);
    gmsh::model::mesh::field::setNumber(2, "DistMin", 0.5);
	gmsh::model::mesh::field::setNumber(2, "DistMax", 0.8);
    // gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.1);
    // gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.1);
    gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.01);
    gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.1);
    gmsh::model::mesh::field::setAsBackgroundMesh(2);

    gmsh::model::occ::synchronize();


    gmsh::model::mesh::generate(2);

    if (save_to_file)
    {
        gmsh::write(filename);
    }
    gmsh::finalize();
}

// void Tiling2D::generateSandwichMeshNonPeridoic(std::vector<std::vector<TV2>>& polygons, 
//         std::vector<TV2>& pbc_corners, bool save_to_file, std::string filename)
// {
//     T eps = 1e-6;

//     TV p1 = pbc_corners[0];
//     TV p2 = pbc_corners[1];
//     TV p3 = pbc_corners[2];
//     TV p4 = pbc_corners[3];


//     auto eval_distance = [](const TV& p1, const TV& p2, const TV& q, T t)
//     {
//         return sqrt(pow((0.1e1 - t) * p1[0] + t * p2[0] - q[0], 0.2e1) + pow((0.1e1 - t) * p1[1] + t * p2[1] - q[1], 0.2e1));
//     };

//     auto closest_t_to_line = [](const TV& p1, const TV& p2, const TV& q){
//         return (p1[0] * p1[0] + (-p2[0] - q[0]) * p1[0] + p1[1] * p1[1] + (-p2[1] - q[1]) * p1[1] + p2[0] * q[0] + p2[1] * q[1]) / (p1[0] * p1[0] - 2 * p1[0] * p2[0] + p1[1] * p1[1] - 2 * p1[1] * p2[1] + p2[0] * p2[0] + p2[1] * p2[1]);
//     };

//     gmsh::initialize();

//     gmsh::model::add("tiling");
//     gmsh::logger::start();

//     gmsh::option::setNumber("Geometry.ToleranceBoolean", eps);
//     gmsh::option::setNumber("Geometry.Tolerance", eps);
//     if (solver.use_quadratic_triangle)
//         gmsh::option::setNumber("Mesh.ElementOrder", 2);
//     else
//         gmsh::option::setNumber("Mesh.ElementOrder", 1);

//     // disable set resolution from point option
//     gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
//     gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 1);
//     gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

//     T point_def = 0.01;
 
//     //Points
//     int acc = 1;
//     // clamping box 1 2 3 4
//     for (int i = 0; i < pbc_corners.size(); ++i)
// 		gmsh::model::occ::addPoint(pbc_corners[i][0], pbc_corners[i][1], 0, 2 * point_def, acc++);
    
//     // sandwich boxes bottom 5 6 7 8
//     // T dx = 0.05 * (pbc_corners[1][0] - pbc_corners[0][0]);
//     T dy = for_printing ? 0.5 : handle_width; // 5 mm
//     T dx = 0.5;
//     gmsh::model::occ::addPoint(pbc_corners[0][0]-dx,  pbc_corners[0][1], 0, 2 * point_def, acc++);
//     gmsh::model::occ::addPoint(pbc_corners[0][0]-dx,  pbc_corners[0][1] - dy, 0, 2 * point_def, acc++);
//     gmsh::model::occ::addPoint(pbc_corners[1][0]+dx, pbc_corners[1][1] - dy, 0, 2 * point_def, acc++);
//     gmsh::model::occ::addPoint(pbc_corners[1][0]+dx, pbc_corners[1][1], 0, 2 * point_def, acc++);
    
//     // sandwich boxes top 9 10 11 12
//     gmsh::model::occ::addPoint(pbc_corners[2][0]+dx, pbc_corners[2][1], 0, 2 * point_def, acc++);
//     gmsh::model::occ::addPoint(pbc_corners[2][0]+dx, pbc_corners[2][1] + dy, 0, 2 * point_def, acc++);
//     gmsh::model::occ::addPoint(pbc_corners[3][0]-dx, pbc_corners[3][1] + dy, 0, 2 * point_def, acc++);
//     gmsh::model::occ::addPoint(pbc_corners[3][0]-dx, pbc_corners[3][1], 0, 2 * point_def, acc++);

//     // inner lattice
//     for (int i=0; i<polygons.size(); ++i)
//     {
//         for(int j=0; j<polygons[i].size(); ++j)
//         {
//             gmsh::model::occ::addPoint(polygons[i][j][0], polygons[i][j][1], 0, point_def, acc++);
//         }
//     }
    
//     //Lines
//     acc = 1;

//     std::vector<double> poly_lines;
    
//     // add clipping box
//     gmsh::model::occ::addLine(1, 2, acc++); 
//     // poly_lines.push_back(acc);
//     gmsh::model::occ::addLine(2, 3, acc++); 
//     // poly_lines.push_back(acc);
//     gmsh::model::occ::addLine(3, 4, acc++); 
//     // poly_lines.push_back(acc);
//     gmsh::model::occ::addLine(4, 1, acc++);
//     // poly_lines.push_back(acc);

//     std::cout << "add clipping box" << std::endl;
    
//     // bottom box
//     gmsh::model::occ::addLine(5, 6, acc++); 
//     // poly_lines.push_back(acc);
//     gmsh::model::occ::addLine(6, 7, acc++); 
//     // poly_lines.push_back(acc);
//     gmsh::model::occ::addLine(7, 8, acc++); 
//     // poly_lines.push_back(acc);
//     gmsh::model::occ::addLine(8, 5, acc++);
//     // poly_lines.push_back(acc);

//     std::cout << "add bottom box" << std::endl;

//     // top box
//     gmsh::model::occ::addLine(9, 10, acc++); 
//     // poly_lines.push_back(acc);
//     gmsh::model::occ::addLine(10, 11, acc++); 
//     // poly_lines.push_back(acc);
//     gmsh::model::occ::addLine(11, 12, acc++); 
//     // poly_lines.push_back(acc);
//     gmsh::model::occ::addLine(12, 9, acc++);
//     // poly_lines.push_back(acc);

//     std::cout << "add top box" << std::endl;

//     int acc_line = 13; // start with the thirdteenth line, because we added 12

//     // for (int i = 0; i < polygons.size(); i++)
//     // {
//     //     for(int j=1; j<polygons[i].size(); ++j)
//     //     {
//     //         gmsh::model::occ::addLine(acc_line++, acc_line, acc++);
//     //     }
//     //     gmsh::model::occ::addLine(acc_line, acc_line-polygons[i].size()+1, acc++);
//     //     ++acc_line;
//     // }


    
//     for(int i=0; i<polygons.size(); ++i)
//     {
//         for(int j=1; j<polygons[i].size(); ++j)
//         {
//             TV q1 = polygons[i][j-1];
//             TV q2 = polygons[i][j];

//             T t_left1 = closest_t_to_line(p1, p4, q1);
//             T t_right1 = closest_t_to_line(p2, p3, q1);
//             T t_bottom1 = closest_t_to_line(p1, p2, q1);
//             T t_top1 = closest_t_to_line(p4, p3, q1);

//             T t_left2 = closest_t_to_line(p1, p4, q2);
//             T t_right2 = closest_t_to_line(p2, p3, q2);
//             T t_bottom2 = closest_t_to_line(p1, p2, q2);
//             T t_top2 = closest_t_to_line(p4, p3, q2);

//             bool pass1 = false;
//             bool pass2 = false;

//             if(eval_distance(p1, p4, q1, t_left1)<1e-5)
//                 pass1 = true;
//             if(eval_distance(p2, p3, q1, t_right1)<1e-5)
//                 pass1 = true;
//             if(eval_distance(p1, p2, q1, t_bottom1)<1e-5)
//                 pass1 = true;
//             if(eval_distance(p4, p3, q1, t_top1)<1e-5)
//                 pass1 = true;

//             if(eval_distance(p1, p4, q2, t_left2)<1e-5)
//                 pass2 = true;
//             if(eval_distance(p2, p3, q2, t_right2)<1e-5)
//                 pass2 = true;
//             if(eval_distance(p1, p2, q2, t_bottom2)<1e-5)
//                 pass2 = true;
//             if(eval_distance(p4, p3, q2, t_top2)<1e-5)
//                 pass2 = true;

//             if(!pass1 || !pass2)
//                 poly_lines.push_back(acc);
//             gmsh::model::occ::addLine(acc_line++, acc_line, acc++);
//         }

//         TV q1 = polygons[i].back();
//         TV q2 = polygons[i][0];

//         T t_left1 = closest_t_to_line(p1, p4, q1);
//         T t_right1 = closest_t_to_line(p2, p3, q1);
//         T t_bottom1 = closest_t_to_line(p1, p2, q1);
//         T t_top1 = closest_t_to_line(p4, p3, q1);

//         T t_left2 = closest_t_to_line(p1, p4, q2);
//         T t_right2 = closest_t_to_line(p2, p3, q2);
//         T t_bottom2 = closest_t_to_line(p1, p2, q2);
//         T t_top2 = closest_t_to_line(p4, p3, q2);

//         bool pass1 = false;
//         bool pass2 = false;

//         if(eval_distance(p1, p4, q1, t_left1)<1e-5)
//             pass1 = true;
//         if(eval_distance(p2, p3, q1, t_right1)<1e-5)
//             pass1 = true;
//         if(eval_distance(p1, p2, q1, t_bottom1)<1e-5)
//             pass1 = true;
//         if(eval_distance(p4, p3, q1, t_top1)<1e-5)
//             pass1 = true;

//         if(eval_distance(p1, p4, q2, t_left2)<1e-5)
//             pass2 = true;
//         if(eval_distance(p2, p3, q2, t_right2)<1e-5)
//             pass2 = true;
//         if(eval_distance(p1, p2, q2, t_bottom2)<1e-5)
//             pass2 = true;
//         if(eval_distance(p4, p3, q2, t_top2)<1e-5)
//             pass2 = true;

//         if(!pass1 || !pass2)
//             poly_lines.push_back(acc);
//         gmsh::model::occ::addLine(acc_line, acc_line-polygons[i].size()+1, acc++);
//         ++acc_line;
//     }

    
//     gmsh::model::mesh::field::add("Distance", 1);
//     gmsh::model::mesh::field::setNumbers(1, "CurvesList", poly_lines);

//     gmsh::model::mesh::field::add("Threshold", 2);
//     gmsh::model::mesh::field::setNumber(2, "InField", 1);
//     gmsh::model::mesh::field::setNumber(2, "SizeMin", point_def);
//     gmsh::model::mesh::field::setNumber(2, "SizeMax", point_def*5.0);
//     gmsh::model::mesh::field::setNumber(2, "DistMin", 0.5);
//     gmsh::model::mesh::field::setNumber(2, "DistMax", 0.8);

//     gmsh::model::mesh::field::setAsBackgroundMesh(2);



//     std::cout << "add line polygon" << std::endl;
//     acc = 1;
    
//     // clipping box
//     gmsh::model::occ::addCurveLoop({1, 2, 3, 4}, acc++);
//     std::cout << "addCurveLoop clipping box" << std::endl;
//     gmsh::model::occ::addCurveLoop({5, 6, 7, 8}, acc++);
//     std::cout << "addCurveLoop bottom box" << std::endl;
//     gmsh::model::occ::addCurveLoop({9, 10, 11, 12}, acc++);
//     std::cout << "addCurveLoop top box" << std::endl;
//     int acc_loop = 13;
    
//     for (int i = 0; i < polygons.size(); i++)
//     {
//         std::vector<int> polygon_loop;
//         for(int j=1; j < polygons[i].size()+1; j++)
//             polygon_loop.push_back(acc_loop++);
//         gmsh::model::occ::addCurveLoop(polygon_loop, acc++);
//     }
    
//     for (int i = 0; i < polygons.size()+3; i++)
//     {
//         gmsh::model::occ::addPlaneSurface({i+1});
//     }
    

//     std::vector<std::pair<int ,int>> poly_idx;
//     for(int i=0; i<polygons.size(); ++i)
//         poly_idx.push_back(std::make_pair(2, i+4));
    
//     std::vector<std::pair<int, int>> ov;
//     std::vector<std::vector<std::pair<int, int> > > ovv;
//     std::cout << "add geometry done" << std::endl;
//     gmsh::model::occ::cut({{2, 1}}, poly_idx, ov, ovv);
//     std::cout << "add cut box done" << std::endl;

//     std::vector<std::pair<int, int>> fuse_bottom_block;
//     std::vector<std::vector<std::pair<int, int> > > _dummy;
//     gmsh::model::occ::fragment({{2, 2}}, ov, fuse_bottom_block, _dummy);
//     std::cout << "add bottom box done" << std::endl;

//     std::vector<std::pair<int, int>> fuse_top_block;
//     std::vector<std::vector<std::pair<int, int> > > _dummy2;
//     gmsh::model::occ::fragment({{2, 3}}, fuse_bottom_block, fuse_top_block, _dummy2);
//     std::cout << "add top box done" << std::endl;
    

//     // add holes
//     std::vector<int> circle_tags;
//     std::vector<std::vector<std::pair<int, int>>> holder(11);
//     // unit is cm

//     if (for_printing)
//     {
//         holder[0] = fuse_top_block;
//         for (int i = 0; i < 5; i++)
//         {
//             int circle_i = gmsh::model::occ::addDisk(
//                 pbc_corners[0][0] + i * 1.25, pbc_corners[0][1] - 0.25, 0.0, 0.15, 0.15);
//             gmsh::model::occ::cut(holder[i], {{2, circle_i}}, holder[i+1], ovv);
//         }
//         for (int i = 5; i < 10; i++)
//         {
//             int circle_i = gmsh::model::occ::addDisk(
//                 pbc_corners[2][0] - (i-5) * 1.25, pbc_corners[2][1] + 0.25, 0.0, 0.15, 0.15);
//             gmsh::model::occ::cut(holder[i], {{2, circle_i}}, holder[i+1], ovv);
//         }
//     }
    

    

//     gmsh::model::occ::synchronize();


//     gmsh::model::mesh::generate(2);

//     if (save_to_file)
//     {
//         gmsh::write(filename);
//     }
//     gmsh::finalize();
// }

void Tiling2D::generateSandwichMeshPerodicInX(std::vector<std::vector<TV2>>& polygons, 
    std::vector<TV2>& pbc_corners, bool save_to_file, std::string filename,
    int resolution, int element_order)
{
    T eps = 1e-6;
    gmsh::initialize();

    gmsh::model::add("tiling");
    gmsh::logger::start();

    gmsh::option::setNumber("Geometry.ToleranceBoolean", eps);
    gmsh::option::setNumber("Geometry.Tolerance", eps);
    gmsh::option::setNumber("Mesh.ElementOrder", element_order);

    // disable set resolution from point option
    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);
 
    //Points
    int acc = 1;
    // clamping box 1 2 3 4
    for (int i = 0; i < pbc_corners.size(); ++i)
		gmsh::model::occ::addPoint(pbc_corners[i][0], pbc_corners[i][1], 0, 2, acc++);
    
    // sandwich boxes bottom 5 6 the other two points already exist
    T dx = 0.05 * (pbc_corners[1][0] - pbc_corners[0][0]);
    gmsh::model::occ::addPoint(pbc_corners[0][0],  pbc_corners[0][1] - dx, 0, 2, acc++);
    gmsh::model::occ::addPoint(pbc_corners[1][0], pbc_corners[1][1] - dx, 0, 2, acc++);
    
    // sandwich boxes top 7 8 
    gmsh::model::occ::addPoint(pbc_corners[2][0], pbc_corners[2][1] + dx, 0, 2, acc++);
    gmsh::model::occ::addPoint(pbc_corners[3][0], pbc_corners[3][1] + dx, 0, 2, acc++);

    // inner lattice
    for (int i=0; i<polygons.size(); ++i)
    {
        for(int j=0; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addPoint(polygons[i][j][0], polygons[i][j][1], 0, 2, acc++);
        }
    }
    
    //Lines
    acc = 1;

    int acc_line = 1;
    
    // add clipping box
    gmsh::model::occ::addLine(1, 2, acc++); 
    gmsh::model::occ::addLine(2, 3, acc++); 
    gmsh::model::occ::addLine(3, 4, acc++); 
    gmsh::model::occ::addLine(4, 1, acc++);
    
    // bottom box
    gmsh::model::occ::addLine(5, 6, acc++); 
    gmsh::model::occ::addLine(6, 2, acc++); 
    gmsh::model::occ::addLine(2, 1, acc++); 
    gmsh::model::occ::addLine(1, 5, acc++);

    // top box
    gmsh::model::occ::addLine(4, 3, acc++); 
    gmsh::model::occ::addLine(3, 7, acc++); 
    gmsh::model::occ::addLine(7, 8, acc++); 
    gmsh::model::occ::addLine(8, 4, acc++);

    acc_line = 9;

    for (int i = 0; i < polygons.size(); i++)
    {
        for(int j=1; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addLine(acc_line++, acc_line, acc++);
        }
        gmsh::model::occ::addLine(acc_line, acc_line-polygons[i].size()+1, acc++);
        ++acc_line;
    }
    
    acc = 1;
    int acc_loop = 1;
    // clipping box
    gmsh::model::occ::addCurveLoop({1, 2, 3, 4}, acc++);
    gmsh::model::occ::addCurveLoop({5, 6, 7, 8}, acc++);
    gmsh::model::occ::addCurveLoop({9, 10, 11, 12}, acc++);
    acc_loop = 13;
    // std::cout << "#polygons " << polygons.size() << std::endl;
    for (int i = 0; i < polygons.size(); i++)
    {
        std::vector<int> polygon_loop;
        for(int j=1; j < polygons[i].size()+1; j++)
            polygon_loop.push_back(acc_loop++);
        gmsh::model::occ::addCurveLoop(polygon_loop, acc++);
    }
    
    for (int i = 0; i < polygons.size()+3; i++)
    {
        gmsh::model::occ::addPlaneSurface({i+1});
    }
    

    std::vector<std::pair<int ,int>> poly_idx;
    for(int i=0; i<polygons.size(); ++i)
        poly_idx.push_back(std::make_pair(2, i+4));
    
    std::vector<std::pair<int, int>> ov;
    std::vector<std::vector<std::pair<int, int> > > ovv;
    std::cout << "add geometry done" << std::endl;
    gmsh::model::occ::cut({{2, 1}}, poly_idx, ov, ovv);
    std::cout << "add cut box done" << std::endl;

    std::vector<std::pair<int, int>> fuse_bottom_block;
    std::vector<std::vector<std::pair<int, int> > > _dummy;
    gmsh::model::occ::fragment({{2, 2}}, ov, fuse_bottom_block, _dummy);
    std::cout << "add bottom box done" << std::endl;

    std::vector<std::pair<int, int>> fuse_top_block;
    std::vector<std::vector<std::pair<int, int> > > _dummy2;
    gmsh::model::occ::fragment({{2, 3}}, fuse_bottom_block, fuse_top_block, _dummy2);
    std::cout << "add top box done" << std::endl;
    

    gmsh::model::mesh::field::add("Distance", 1);

    gmsh::model::mesh::field::add("Threshold", 2);
    gmsh::model::mesh::field::setNumber(2, "InField", 1);
    if (resolution == 0)
    {
        gmsh::model::mesh::field::setNumber(2, "SizeMin", 1.0);
        gmsh::model::mesh::field::setNumber(2, "SizeMax", 2.0);    
    }
    else if (resolution == 1)
    {
        gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.2);
        gmsh::model::mesh::field::setNumber(2, "SizeMax", 1.5);
    }
    else if (resolution == 2)
    {
        gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.1);
        gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.5);
    }
    else if (resolution == 3)
    {
        gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.01);
        gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.2);
    }
    else if (resolution == 4)
    {
        gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.01);
        gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.1);
    }

    gmsh::model::mesh::field::setAsBackgroundMesh(2);

    gmsh::model::occ::synchronize();

    int zero_idx;
    for(int i=0; i < pbc_corners.size(); i++)
    {
        if(pbc_corners[i].norm()<1e-6)
        {
            zero_idx = i;
            break;
        }
    }

    TV2 t1 = pbc_corners[(zero_idx+1)%pbc_corners.size()];
    TV2 t2 = pbc_corners[(zero_idx+3)%pbc_corners.size()];

    std::vector<T> translation_hor({1, 0, 0, t1[0], 0, 1, 0, t1[1], 0, 0, 1, 0, 0, 0, 0, 1});
	std::vector<T> translation_ver({1, 0, 0, t2[0], 0, 1, 0, t2[1], 0, 0, 1, 0, 0, 0, 0, 1});
    int x_pair_cnt = 0;
    std::vector<std::pair<int, int>> sleft;
    gmsh::model::getEntitiesInBoundingBox(std::min(0.0,t2[0])-eps, std::min(0.0,t2[1])-eps, -eps, std::max(0.0,t2[0])+eps, std::max(0.0,t2[1])+eps, eps, sleft, 1);
    // std::ofstream pbc_output(data_folder + "pbc_data.txt");
    for(auto i : sleft) {
        T xmin, ymin, zmin, xmax, ymax, zmax;
        gmsh::model::getBoundingBox(i.first, i.second, xmin, ymin, zmin, xmax, ymax, zmax);
        std::vector<std::pair<int, int> > sright;
        gmsh::model::getEntitiesInBoundingBox(xmin-eps+t1[0], ymin-eps+t1[1], zmin - eps, xmax+eps+t1[0], ymax+eps+t1[1], zmax + eps, sright, 1);

        for(auto j : sright) {
            T xmin2, ymin2, zmin2, xmax2, ymax2, zmax2;
            gmsh::model::getBoundingBox(j.first, j.second, xmin2, ymin2, zmin2, xmax2, ymax2, zmax2);
            xmin2 -= t1[0];
            ymin2 -= t1[1];
            xmax2 -= t1[0];
            ymax2 -= t1[1];
            if(std::abs(xmin2 - xmin) < eps && std::abs(xmax2 - xmax) < eps &&
                std::abs(ymin2 - ymin) < eps && std::abs(ymax2 - ymax) < eps &&
                std::abs(zmin2 - zmin) < eps && std::abs(zmax2 - zmax) < eps) 
            {
                gmsh::model::mesh::setPeriodic(1, {j.second}, {i.second}, translation_hor);
                x_pair_cnt++;
            }
        }
    }
    gmsh::model::occ::synchronize();
    gmsh::model::mesh::generate(2);

    if (save_to_file)
    {
        gmsh::write(filename);
    }
    gmsh::finalize();
    
}

void Tiling2D::generateHomogenousMesh(std::vector<std::vector<TV2>>& polygons, 
        std::vector<TV2>& pbc_corners, bool save_to_file, std::string prefix)
{
    T eps = 1e-5;
    gmsh::initialize();

    T mult = 10000000.0;

    gmsh::model::add("tiling");
    gmsh::logger::start();

    gmsh::option::setNumber("Geometry.ToleranceBoolean", eps);
    gmsh::option::setNumber("Geometry.Tolerance", eps);
    if (solver.use_quadratic_triangle)
        gmsh::option::setNumber("Mesh.ElementOrder", 2);
    else
        gmsh::option::setNumber("Mesh.ElementOrder", 1);

    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);
    
    //Points
    int acc = 1;
    pbc_corners = {TV(0, 0), TV(10,0), TV(10,10), TV(0,10)};

    // clamping box 1 2 3 4
    for (int i = 0; i < pbc_corners.size(); ++i)
		gmsh::model::occ::addPoint(pbc_corners[i][0], pbc_corners[i][1], 0, 1, acc++);

    
    //Lines
    acc = 1;
    int starting_vtx = 1;

    // add clipping box
    gmsh::model::occ::addLine(1, 2, acc++); 
    gmsh::model::occ::addLine(2, 3, acc++); 
    gmsh::model::occ::addLine(3, 4, acc++); 
    gmsh::model::occ::addLine(4, 1, acc++);

    starting_vtx = 5;

    
    
    acc = 1;
    int acc_loop = 1;

    gmsh::model::occ::addCurveLoop({1, 2, 3, 4}, acc++);
   

    gmsh::model::occ::addPlaneSurface({1});

    gmsh::model::occ::synchronize();

    int zero_idx;
    for(int i=0; i < pbc_corners.size(); i++)
    {
        if(pbc_corners[i].norm()<1e-6)
        {
            zero_idx = i;
            break;
        }
    }

    TV2 t1 = pbc_corners[(zero_idx+1)%pbc_corners.size()];
    TV2 t2 = pbc_corners[(zero_idx+3)%pbc_corners.size()];

    std::vector<T> translation_hor({1, 0, 0, t1[0], 0, 1, 0, t1[1], 0, 0, 1, 0, 0, 0, 0, 1});
	std::vector<T> translation_ver({1, 0, 0, t2[0], 0, 1, 0, t2[1], 0, 0, 1, 0, 0, 0, 0, 1});

    std::vector<std::pair<int, int>> sleft;
    gmsh::model::getEntitiesInBoundingBox(std::min(0.0,t2[0])-eps, std::min(0.0,t2[1])-eps, -eps, std::max(0.0,t2[0])+eps, std::max(0.0,t2[1])+eps, eps, sleft, 1);
    // std::ofstream pbc_output(data_folder + "pbc_data.txt");
    for(auto i : sleft) {
        T xmin, ymin, zmin, xmax, ymax, zmax;
        gmsh::model::getBoundingBox(i.first, i.second, xmin, ymin, zmin, xmax, ymax, zmax);
        std::vector<std::pair<int, int> > sright;
        gmsh::model::getEntitiesInBoundingBox(xmin-eps+t1[0], ymin-eps+t1[1], zmin - eps, xmax+eps+t1[0], ymax+eps+t1[1], zmax + eps, sright, 1);

        for(auto j : sright) {
            T xmin2, ymin2, zmin2, xmax2, ymax2, zmax2;
            gmsh::model::getBoundingBox(j.first, j.second, xmin2, ymin2, zmin2, xmax2, ymax2, zmax2);
            xmin2 -= t1[0];
            ymin2 -= t1[1];
            xmax2 -= t1[0];
            ymax2 -= t1[1];
            if(std::abs(xmin2 - xmin) < eps && std::abs(xmax2 - xmax) < eps &&
                std::abs(ymin2 - ymin) < eps && std::abs(ymax2 - ymax) < eps &&
                std::abs(zmin2 - zmin) < eps && std::abs(zmax2 - zmax) < eps) 
            {
                gmsh::model::mesh::setPeriodic(1, {j.second}, {i.second}, translation_hor);
                // pbc_output << "X " << j.second << " " << i.second << std::endl;
            }
        }
    }

    std::vector<std::pair<int, int>> sbottom;
    gmsh::model::getEntitiesInBoundingBox(std::min(0.0,t1[0])-eps, std::min(0.0,t1[1])-eps, -eps, std::max(0.0,t1[0])+eps, std::max(0.0,t1[1])+eps, eps, sbottom, 1);

    for(auto i : sbottom) {
        T xmin, ymin, zmin, xmax, ymax, zmax;
        gmsh::model::getBoundingBox(i.first, i.second, xmin, ymin, zmin, xmax, ymax, zmax);
        std::vector<std::pair<int, int> > stop;
        gmsh::model::getEntitiesInBoundingBox(xmin-eps+t2[0], ymin-eps+t2[1], zmin - eps, xmax+eps+t2[0], ymax+eps+t2[1], zmax + eps, stop, 1);

        for(auto j : stop) {
            T xmin2, ymin2, zmin2, xmax2, ymax2, zmax2;
            gmsh::model::getBoundingBox(j.first, j.second, xmin2, ymin2, zmin2, xmax2, ymax2, zmax2);
            xmin2 -= t2[0];
            ymin2 -= t2[1];
            xmax2 -= t2[0];
            ymax2 -= t2[1];
            if(std::abs(xmin2 - xmin) < eps && std::abs(xmax2 - xmax) < eps &&
                std::abs(ymin2 - ymin) < eps && std::abs(ymax2 - ymax) < eps &&
                std::abs(zmin2 - zmin) < eps && std::abs(zmax2 - zmax) < eps) 
            {
                gmsh::model::mesh::setPeriodic(1, {j.second}, {i.second}, translation_ver);
                // pbc_output << "Y " << j.second << " " << i.second << std::endl;
            }
        }
    }
    gmsh::model::mesh::field::add("Distance", 1);

    gmsh::model::mesh::field::add("Threshold", 2);
    gmsh::model::mesh::field::setNumber(2, "InField", 1);
    // gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.05);
    // gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.1);
    gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.1);
    gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.2);
    // gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.3);
    // gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.8);
    // gmsh::model::mesh::field::setNumber(2, "SizeMin", 1.0);
    // gmsh::model::mesh::field::setNumber(2, "SizeMax", 2.0);
    gmsh::model::mesh::field::setAsBackgroundMesh(2);
    gmsh::model::occ::synchronize();

    gmsh::model::mesh::generate(2);

    
    gmsh::write(prefix + ".vtk");
    std::ofstream translation(prefix + "_translation.txt");
    translation << t1.transpose() << std::endl;
    translation << t2.transpose() << std::endl;
    translation.close();
    gmsh::finalize();
}
void Tiling2D::generateNonPeriodicMesh(std::vector<std::vector<TV2>>& polygons, 
        std::vector<TV2>& pbc_corners, bool save_to_file, std::string prefix)
{
    TV p1 = pbc_corners[0];
    TV p2 = pbc_corners[1];
    TV p3 = pbc_corners[2];
    TV p4 = pbc_corners[3];

    T eps = 1e-6;
    gmsh::initialize();

    gmsh::model::add("tiling");
    // gmsh::logger::start();
    // gmsh::logger::stop();

    gmsh::option::setNumber("Geometry.ToleranceBoolean", eps);
    gmsh::option::setNumber("Geometry.Tolerance", eps);
    gmsh::option::setNumber("Mesh.ElementOrder", 2);

    // gmsh::option::setNumber("General.Verbosity", 0);

    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

    T th = eps;
    //Points
    int acc = 1;

    // clamping box 1 2 3 4
    for (int i = 0; i < pbc_corners.size(); ++i)
		gmsh::model::occ::addPoint(pbc_corners[i][0], pbc_corners[i][1], 0, 1, acc++);

    for(int i=0; i<polygons.size(); ++i)
    {
        for(int j=0; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addPoint(polygons[i][j][0], polygons[i][j][1], 0, 1, acc++);
        }
    }
    
    //Lines
    acc = 1;
    int starting_vtx = 1;

    // add clipping box
    gmsh::model::occ::addLine(1, 2, acc++); 
    gmsh::model::occ::addLine(2, 3, acc++); 
    gmsh::model::occ::addLine(3, 4, acc++); 
    gmsh::model::occ::addLine(4, 1, acc++);

    starting_vtx = 5;
    std::vector<T> poly_lines;
    for (int i = 0; i < polygons.size(); i++)
    {
        for(int j=1; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addLine(starting_vtx++, starting_vtx, acc++);
            poly_lines.push_back(acc);
        }
        gmsh::model::occ::addLine(starting_vtx, starting_vtx-polygons[i].size()+1, acc++);
        poly_lines.push_back(acc);
        ++starting_vtx;
    }

    gmsh::model::mesh::field::add("Distance", 1);
    // gmsh::model::mesh::field::setNumbers(1, "CurvesList", poly_lines);

    gmsh::model::mesh::field::add("Threshold", 2);
    gmsh::model::mesh::field::setNumber(2, "InField", 1);
    gmsh::model::mesh::field::setNumber(2, "SizeMin", 2.0);
    gmsh::model::mesh::field::setNumber(2, "SizeMax", 2.0);
    // gmsh::model::mesh::field::setNumber(2, "DistMin", 0.005);

    gmsh::model::mesh::field::setAsBackgroundMesh(2);
    
    acc = 1;
    int acc_loop = 1;

    gmsh::model::occ::addCurveLoop({1, 2, 3, 4}, acc++);
    acc_loop = 5;
    for (int i = 0; i < polygons.size(); i++)
    {
        std::vector<int> polygon_loop;
        for(int j=1; j < polygons[i].size()+1; j++)
            polygon_loop.push_back(acc_loop++);
        gmsh::model::occ::addCurveLoop(polygon_loop, acc++);
    }

    for (int i = 0; i < polygons.size()+1; i++)
    {
        gmsh::model::occ::addPlaneSurface({i+1});
    }

    std::vector<std::pair<int ,int>> poly_idx;
    for(int i=0; i<polygons.size(); ++i)
        poly_idx.push_back(std::make_pair(2, i+2));
    
    std::vector<std::pair<int, int>> ov;
    std::vector<std::vector<std::pair<int, int> > > ovv;
    gmsh::model::occ::cut({{2, 1}}, poly_idx, ov, ovv);
    gmsh::model::occ::synchronize();

    gmsh::model::mesh::generate(2);

    
    gmsh::write(prefix + ".vtk");
    gmsh::finalize();
}

void Tiling2D::generatePeriodicMesh(std::vector<std::vector<TV2>>& polygons, 
    std::vector<TV2>& pbc_corners, bool save_to_file, std::string prefix)
{
      // Before using any functions in the C++ API, Gmsh must be initialized:

    TV p1 = pbc_corners[0];
    TV p2 = pbc_corners[1];
    TV p3 = pbc_corners[2];
    TV p4 = pbc_corners[3];

    T eps = 1e-6;
    gmsh::initialize();

    gmsh::model::add("tiling");
    // gmsh::logger::start();
    // gmsh::logger::stop();

    gmsh::option::setNumber("Geometry.ToleranceBoolean", eps);
    gmsh::option::setNumber("Geometry.Tolerance", eps);
    if (solver.use_quadratic_triangle)
        gmsh::option::setNumber("Mesh.ElementOrder", 2);
    else
        gmsh::option::setNumber("Mesh.ElementOrder", 1);

    gmsh::option::setNumber("General.Verbosity", 0);

    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

    T th = eps;
    //Points
    int acc = 1;

    // clamping box 1 2 3 4
    for (int i = 0; i < pbc_corners.size(); ++i)
		gmsh::model::occ::addPoint(pbc_corners[i][0], pbc_corners[i][1], 0, 1, acc++);

    for(int i=0; i<polygons.size(); ++i)
    {
        for(int j=0; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addPoint(polygons[i][j][0], polygons[i][j][1], 0, 1, acc++);
        }
    }
    
    //Lines
    acc = 1;
    int starting_vtx = 1;

    // add clipping box
    gmsh::model::occ::addLine(1, 2, acc++); 
    gmsh::model::occ::addLine(2, 3, acc++); 
    gmsh::model::occ::addLine(3, 4, acc++); 
    gmsh::model::occ::addLine(4, 1, acc++);

    starting_vtx = 5;
    std::vector<T> poly_lines;
    for (int i = 0; i < polygons.size(); i++)
    {
        for(int j=1; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addLine(starting_vtx++, starting_vtx, acc++);
            poly_lines.push_back(acc);
        }
        gmsh::model::occ::addLine(starting_vtx, starting_vtx-polygons[i].size()+1, acc++);
        poly_lines.push_back(acc);
        ++starting_vtx;
    }

    gmsh::model::mesh::field::add("Distance", 1);
    // gmsh::model::mesh::field::setNumbers(1, "CurvesList", poly_lines);

    gmsh::model::mesh::field::add("Threshold", 2);
    gmsh::model::mesh::field::setNumber(2, "InField", 1);
    gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.2);
    gmsh::model::mesh::field::setNumber(2, "SizeMax", 1.0);
    gmsh::model::mesh::field::setNumber(2, "DistMin", 0.005);

    // gmsh::model::mesh::field::setNumber(2, "SizeMin", 1.0);
    // gmsh::model::mesh::field::setNumber(2, "SizeMax", 2.0);
    // gmsh::model::mesh::field::setNumber(2, "DistMin", 0.005);

    gmsh::model::mesh::field::setAsBackgroundMesh(2);
    
    acc = 1;
    int acc_loop = 1;

    gmsh::model::occ::addCurveLoop({1, 2, 3, 4}, acc++);
    acc_loop = 5;
    for (int i = 0; i < polygons.size(); i++)
    {
        std::vector<int> polygon_loop;
        for(int j=1; j < polygons[i].size()+1; j++)
            polygon_loop.push_back(acc_loop++);
        gmsh::model::occ::addCurveLoop(polygon_loop, acc++);
    }

    for (int i = 0; i < polygons.size()+1; i++)
    {
        gmsh::model::occ::addPlaneSurface({i+1});
    }

    std::vector<std::pair<int ,int>> poly_idx;
    for(int i=0; i<polygons.size(); ++i)
        poly_idx.push_back(std::make_pair(2, i+2));
    
    std::vector<std::pair<int, int>> ov;
    std::vector<std::vector<std::pair<int, int> > > ovv;
    gmsh::model::occ::cut({{2, 1}}, poly_idx, ov, ovv);
    gmsh::model::occ::synchronize();

    int zero_idx;
    for(int i=0; i < pbc_corners.size(); i++)
    {
        if(pbc_corners[i].norm()<1e-6)
        {
            zero_idx = i;
            break;
        }
    }

    TV2 t1 = pbc_corners[(zero_idx+1)%pbc_corners.size()];
    TV2 t2 = pbc_corners[(zero_idx+3)%pbc_corners.size()];

    std::vector<T> translation_hor({1, 0, 0, t1[0], 0, 1, 0, t1[1], 0, 0, 1, 0, 0, 0, 0, 1});
	std::vector<T> translation_ver({1, 0, 0, t2[0], 0, 1, 0, t2[1], 0, 0, 1, 0, 0, 0, 0, 1});

    std::vector<std::pair<int, int>> sleft;
    gmsh::model::getEntitiesInBoundingBox(std::min(0.0,t2[0])-eps, std::min(0.0,t2[1])-eps, -eps, std::max(0.0,t2[0])+eps, std::max(0.0,t2[1])+eps, eps, sleft, 1);
    // std::ofstream pbc_output(data_folder + "pbc_data.txt");
    for(auto i : sleft) {
        T xmin, ymin, zmin, xmax, ymax, zmax;
        gmsh::model::getBoundingBox(i.first, i.second, xmin, ymin, zmin, xmax, ymax, zmax);
        std::vector<std::pair<int, int> > sright;
        gmsh::model::getEntitiesInBoundingBox(xmin-eps+t1[0], ymin-eps+t1[1], zmin - eps, xmax+eps+t1[0], ymax+eps+t1[1], zmax + eps, sright, 1);

        for(auto j : sright) {
            T xmin2, ymin2, zmin2, xmax2, ymax2, zmax2;
            gmsh::model::getBoundingBox(j.first, j.second, xmin2, ymin2, zmin2, xmax2, ymax2, zmax2);
            xmin2 -= t1[0];
            ymin2 -= t1[1];
            xmax2 -= t1[0];
            ymax2 -= t1[1];
            if(std::abs(xmin2 - xmin) < eps && std::abs(xmax2 - xmax) < eps &&
                std::abs(ymin2 - ymin) < eps && std::abs(ymax2 - ymax) < eps &&
                std::abs(zmin2 - zmin) < eps && std::abs(zmax2 - zmax) < eps) 
            {
                gmsh::model::mesh::setPeriodic(1, {j.second}, {i.second}, translation_hor);
                // pbc_output << "X " << j.second << " " << i.second << std::endl;
            }
        }
    }

    std::vector<std::pair<int, int>> sbottom;
    gmsh::model::getEntitiesInBoundingBox(std::min(0.0,t1[0])-eps, std::min(0.0,t1[1])-eps, -eps, std::max(0.0,t1[0])+eps, std::max(0.0,t1[1])+eps, eps, sbottom, 1);

    for(auto i : sbottom) {
        T xmin, ymin, zmin, xmax, ymax, zmax;
        gmsh::model::getBoundingBox(i.first, i.second, xmin, ymin, zmin, xmax, ymax, zmax);
        std::vector<std::pair<int, int> > stop;
        gmsh::model::getEntitiesInBoundingBox(xmin-eps+t2[0], ymin-eps+t2[1], zmin - eps, xmax+eps+t2[0], ymax+eps+t2[1], zmax + eps, stop, 1);

        for(auto j : stop) {
            T xmin2, ymin2, zmin2, xmax2, ymax2, zmax2;
            gmsh::model::getBoundingBox(j.first, j.second, xmin2, ymin2, zmin2, xmax2, ymax2, zmax2);
            xmin2 -= t2[0];
            ymin2 -= t2[1];
            xmax2 -= t2[0];
            ymax2 -= t2[1];
            if(std::abs(xmin2 - xmin) < eps && std::abs(xmax2 - xmax) < eps &&
                std::abs(ymin2 - ymin) < eps && std::abs(ymax2 - ymax) < eps &&
                std::abs(zmin2 - zmin) < eps && std::abs(zmax2 - zmax) < eps) 
            {
                gmsh::model::mesh::setPeriodic(1, {j.second}, {i.second}, translation_ver);
                // pbc_output << "Y " << j.second << " " << i.second << std::endl;
            }
        }
    }
    
    gmsh::model::occ::synchronize();

    gmsh::model::mesh::generate(2);

    
    gmsh::write(prefix + ".vtk");
    std::ofstream translation(prefix + "_translation.txt");
    translation << std::setprecision(20) << t1.transpose() << std::endl;
    translation << std::setprecision(20) << t2.transpose() << std::endl;
    translation.close();
    gmsh::finalize();
    
}

void Tiling2D::generatePeriodicMeshHardCodeResolution(std::vector<std::vector<TV2>>& polygons, 
    std::vector<TV2>& pbc_corners, bool save_to_file, std::string prefix)
{
      // Before using any functions in the C++ API, Gmsh must be initialized:

    TV p1 = pbc_corners[0];
    TV p2 = pbc_corners[1];
    TV p3 = pbc_corners[2];
    TV p4 = pbc_corners[3];

    T eps = 1e-6;
    gmsh::initialize();

    gmsh::model::add("tiling");
    // gmsh::logger::start();
    // gmsh::logger::stop();

    gmsh::option::setNumber("Geometry.ToleranceBoolean", eps);
    gmsh::option::setNumber("Geometry.Tolerance", eps);
    gmsh::option::setNumber("Mesh.ElementOrder", 2);

    gmsh::option::setNumber("General.Verbosity", 0);

    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

    T th = eps;
    //Points
    int acc = 1;

    // clamping box 1 2 3 4
    for (int i = 0; i < pbc_corners.size(); ++i)
		gmsh::model::occ::addPoint(pbc_corners[i][0], pbc_corners[i][1], 0, 1, acc++);

    for(int i=0; i<polygons.size(); ++i)
    {
        for(int j=0; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addPoint(polygons[i][j][0], polygons[i][j][1], 0, 1, acc++);
        }
    }
    
    //Lines
    acc = 1;
    int starting_vtx = 1;

    // add clipping box
    gmsh::model::occ::addLine(1, 2, acc++); 
    gmsh::model::occ::addLine(2, 3, acc++); 
    gmsh::model::occ::addLine(3, 4, acc++); 
    gmsh::model::occ::addLine(4, 1, acc++);

    starting_vtx = 5;
    std::vector<T> poly_lines;
    for (int i = 0; i < polygons.size(); i++)
    {
        for(int j=1; j<polygons[i].size(); ++j)
        {
            gmsh::model::occ::addLine(starting_vtx++, starting_vtx, acc++);
            poly_lines.push_back(acc);
        }
        gmsh::model::occ::addLine(starting_vtx, starting_vtx-polygons[i].size()+1, acc++);
        poly_lines.push_back(acc);
        ++starting_vtx;
    }

    gmsh::model::mesh::field::add("Distance", 1);
    // gmsh::model::mesh::field::setNumbers(1, "CurvesList", poly_lines);

    gmsh::model::mesh::field::add("Threshold", 2);
    gmsh::model::mesh::field::setNumber(2, "InField", 1);
    gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.02);
    gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.02);
    // gmsh::model::mesh::field::setNumber(2, "SizeMin", 0.008);
    // gmsh::model::mesh::field::setNumber(2, "SizeMax", 0.008);
    // gmsh::model::mesh::field::setNumber(2, "DistMin", 0.005);

    gmsh::model::mesh::field::setAsBackgroundMesh(2);
    
    acc = 1;
    int acc_loop = 1;

    gmsh::model::occ::addCurveLoop({1, 2, 3, 4}, acc++);
    acc_loop = 5;
    for (int i = 0; i < polygons.size(); i++)
    {
        std::vector<int> polygon_loop;
        for(int j=1; j < polygons[i].size()+1; j++)
            polygon_loop.push_back(acc_loop++);
        gmsh::model::occ::addCurveLoop(polygon_loop, acc++);
    }

    for (int i = 0; i < polygons.size()+1; i++)
    {
        gmsh::model::occ::addPlaneSurface({i+1});
    }

    std::vector<std::pair<int ,int>> poly_idx;
    for(int i=0; i<polygons.size(); ++i)
        poly_idx.push_back(std::make_pair(2, i+2));
    
    std::vector<std::pair<int, int>> ov;
    std::vector<std::vector<std::pair<int, int> > > ovv;
    gmsh::model::occ::cut({{2, 1}}, poly_idx, ov, ovv);
    gmsh::model::occ::synchronize();

    int zero_idx;
    for(int i=0; i < pbc_corners.size(); i++)
    {
        if(pbc_corners[i].norm()<1e-6)
        {
            zero_idx = i;
            break;
        }
    }

    TV2 t1 = pbc_corners[(zero_idx+1)%pbc_corners.size()];
    TV2 t2 = pbc_corners[(zero_idx+3)%pbc_corners.size()];

    std::vector<T> translation_hor({1, 0, 0, t1[0], 0, 1, 0, t1[1], 0, 0, 1, 0, 0, 0, 0, 1});
	std::vector<T> translation_ver({1, 0, 0, t2[0], 0, 1, 0, t2[1], 0, 0, 1, 0, 0, 0, 0, 1});

    std::vector<std::pair<int, int>> sleft;
    gmsh::model::getEntitiesInBoundingBox(std::min(0.0,t2[0])-eps, std::min(0.0,t2[1])-eps, -eps, std::max(0.0,t2[0])+eps, std::max(0.0,t2[1])+eps, eps, sleft, 1);
    // std::ofstream pbc_output(data_folder + "pbc_data.txt");
    for(auto i : sleft) {
        T xmin, ymin, zmin, xmax, ymax, zmax;
        gmsh::model::getBoundingBox(i.first, i.second, xmin, ymin, zmin, xmax, ymax, zmax);
        std::vector<std::pair<int, int> > sright;
        gmsh::model::getEntitiesInBoundingBox(xmin-eps+t1[0], ymin-eps+t1[1], zmin - eps, xmax+eps+t1[0], ymax+eps+t1[1], zmax + eps, sright, 1);

        for(auto j : sright) {
            T xmin2, ymin2, zmin2, xmax2, ymax2, zmax2;
            gmsh::model::getBoundingBox(j.first, j.second, xmin2, ymin2, zmin2, xmax2, ymax2, zmax2);
            xmin2 -= t1[0];
            ymin2 -= t1[1];
            xmax2 -= t1[0];
            ymax2 -= t1[1];
            if(std::abs(xmin2 - xmin) < eps && std::abs(xmax2 - xmax) < eps &&
                std::abs(ymin2 - ymin) < eps && std::abs(ymax2 - ymax) < eps &&
                std::abs(zmin2 - zmin) < eps && std::abs(zmax2 - zmax) < eps) 
            {
                gmsh::model::mesh::setPeriodic(1, {j.second}, {i.second}, translation_hor);
                // pbc_output << "X " << j.second << " " << i.second << std::endl;
            }
        }
    }

    std::vector<std::pair<int, int>> sbottom;
    gmsh::model::getEntitiesInBoundingBox(std::min(0.0,t1[0])-eps, std::min(0.0,t1[1])-eps, -eps, std::max(0.0,t1[0])+eps, std::max(0.0,t1[1])+eps, eps, sbottom, 1);

    for(auto i : sbottom) {
        T xmin, ymin, zmin, xmax, ymax, zmax;
        gmsh::model::getBoundingBox(i.first, i.second, xmin, ymin, zmin, xmax, ymax, zmax);
        std::vector<std::pair<int, int> > stop;
        gmsh::model::getEntitiesInBoundingBox(xmin-eps+t2[0], ymin-eps+t2[1], zmin - eps, xmax+eps+t2[0], ymax+eps+t2[1], zmax + eps, stop, 1);

        for(auto j : stop) {
            T xmin2, ymin2, zmin2, xmax2, ymax2, zmax2;
            gmsh::model::getBoundingBox(j.first, j.second, xmin2, ymin2, zmin2, xmax2, ymax2, zmax2);
            xmin2 -= t2[0];
            ymin2 -= t2[1];
            xmax2 -= t2[0];
            ymax2 -= t2[1];
            if(std::abs(xmin2 - xmin) < eps && std::abs(xmax2 - xmax) < eps &&
                std::abs(ymin2 - ymin) < eps && std::abs(ymax2 - ymax) < eps &&
                std::abs(zmin2 - zmin) < eps && std::abs(zmax2 - zmax) < eps) 
            {
                gmsh::model::mesh::setPeriodic(1, {j.second}, {i.second}, translation_ver);
                // pbc_output << "Y " << j.second << " " << i.second << std::endl;
            }
        }
    }
    
    gmsh::model::occ::synchronize();

    gmsh::model::mesh::generate(2);

    
    gmsh::write(prefix + ".vtk");
    std::ofstream translation(prefix + "_translation.txt");
    translation << std::setprecision(20) << t1.transpose() << std::endl;
    translation << std::setprecision(20) << t2.transpose() << std::endl;
    translation.close();
    gmsh::finalize();
    
}