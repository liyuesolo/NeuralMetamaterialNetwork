#include "../include/Tiling2D.h"

glm::dmat3 Tiling2D::centrePSRect(T xmin, T ymin, T xmax, T ymax)
{
    T sc = std::min( 6.5*72.0 / (xmax-xmin), 9.0*72.0 / (ymax-ymin) );
    return glm::dmat3( 1, 0, 0, 0, 1, 0, 4.25*72.0, 5.5*72.0, 1.0 )
        * glm::dmat3( sc, 0, 0, 0, sc, 0, 0, 0, 1 )
        * glm::dmat3( 1, 0, 0, 0, 1, 0, -0.5*(xmin+xmax), -0.5*(ymin+ymax), 1 );
}

std::vector<glm::dvec2> Tiling2D::outShapeVec(const std::vector<glm::dvec2>& vec, const glm::dmat3& M)
{
    std::vector<glm::dvec2> data_points;

    glm::dvec2 p = M * glm::dvec3( vec.back(), 1.0 );
    data_points.push_back(glm::dvec2(p[0], p[1]));

    for( size_t idx = 0; idx < vec.size(); idx += 3 ) {
        glm::dvec2 p1 = M * glm::dvec3( vec[idx], 1.0 );
        glm::dvec2 p2 = M * glm::dvec3( vec[idx+1], 1.0 );
        glm::dvec2 p3 = M * glm::dvec3( vec[idx+2], 1.0 );

        data_points.push_back(glm::dvec2(p1[0], p1[1]));
        data_points.push_back(glm::dvec2(p2[0], p2[1]));
        data_points.push_back(glm::dvec2(p3[0], p3[1]));
    }

    return data_points;
}

void Tiling2D::getTilingShape(std::vector<dvec2>& shape, const csk::IsohedralTiling& tiling, 
    const std::vector<std::vector<dvec2>>& edges)
{
    for( auto i : tiling.shape() ) {
        // Get the relevant edge shape created above using i->getId().
        const std::vector<dvec2>& ed = edges[ i->getId() ];
        // Also get the transform that maps to the line joining consecutive
        // tiling vertices.
        const glm::dmat3& TT = i->getTransform();

        // If i->isReversed() is true, we need to run the parameterization
        // of the path backwards.
        if( i->isReversed() ) {
            for( size_t idx = 1; idx < ed.size(); ++idx ) {
                shape.push_back( TT * glm::dvec3( ed[ed.size()-1-idx], 1.0 ) );
            }
        } else {
            for( size_t idx = 1; idx < ed.size(); ++idx ) {
                shape.push_back( TT * glm::dvec3( ed[idx], 1.0 ) );
            }
        }
    }
}

void Tiling2D::getTilingEdges(const csk::IsohedralTiling& tiling,
        const Vector<T, 4>& eij,
        std::vector<std::vector<dvec2>>& edges)
{
    using namespace csk;
    for( U8 idx = 0; idx < tiling.numEdgeShapes(); ++idx ) {
        std::vector<dvec2> ej;

        ej.push_back( dvec2( 0, 0.0 ) );
        ej.push_back( dvec2( eij[0], eij[1] ) );
        ej.push_back( dvec2( eij[2], eij[3] ) );
        ej.push_back( dvec2( 1.0, 0.0 ) );
        
        // Now, depending on the edge shape class, enforce symmetry 
        // constraints on edges.
        switch( tiling.getEdgeShape( idx ) ) {
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
    }
}

void Tiling2D::shapeToPolygon(ClipperLib::Paths& final_shape, 
    std::vector<std::vector<TV2>>& polygons, T mult)
{
    polygons.resize(final_shape.size());

    for(int i=0; i<final_shape.size(); ++i)
    {
        for(int j=0; j<final_shape[i].size(); ++j)
        {
            TV2 cur_point = TV2(final_shape[i][j].X/mult, final_shape[i][j].Y/mult);

            if(j==final_shape[i].size()-1)
            {
                if((cur_point-polygons[i].front()).norm()>1e-4)
                    polygons[i].push_back(cur_point);
            }
            else if(j>0)
            {
                if((cur_point-polygons[i].back()).norm()>1e-4)
                    polygons[i].push_back(cur_point);
            }
            else
                polygons[i].push_back(cur_point);	
        }
    }
}

void Tiling2D::periodicToBase(const Vector<T, 8>& periodic, std::vector<TV2>& eigen_base)
{
    eigen_base.resize(0);

    eigen_base.push_back(TV2(periodic[0], periodic[1]));
    eigen_base.push_back(TV2(periodic[2], periodic[3]));
    eigen_base.push_back(TV2(periodic[4], periodic[5]));
    eigen_base.push_back(TV2(periodic[6], periodic[7]));
}

void Tiling2D::getTranslationUnitPolygon(std::vector<std::vector<dvec2>>& polygons_v,
        const std::vector<dvec2>& shape, const csk::IsohedralTiling& tiling, 
        Vector<T, 4>& transf, int width, int depth, TV2& xy_shift)
{
    int min_y=10000, max_y=-10000, min_x=10000, max_x=-10000;
    
    int ii=0;
    int extension = 4;
    dmat3 M = centrePSRect( -width, -depth, width, depth );
    for( auto i : tiling.fillRegion( -extension * width, -extension * depth, extension * width, extension * depth ) ) 
    {
        dmat3 T = M * i->getTransform();

        std::vector<dvec2> outv = outShapeVec( shape, T );

        if(T[0][0]!=T[1][1])
            std::reverse(outv.begin(), outv.end());

        min_y = std::min(i->getT2(), min_y);
        max_y = std::max(i->getT2(), max_y);

        min_x = std::min(i->getT1(), min_x);
        max_x = std::max(i->getT1(), max_x);

        polygons_v.push_back(outv);
    }

    int chosen_x = (max_x+min_x)/2;
    int chosen_y = (max_y+min_y)/2;

    TV2 xy, x1y, xy1, x1y1;

    for( auto i : tiling.fillRegion( -extension * width, -extension * depth, extension * width, extension * depth ) ) {

        dmat3 T = M * i->getTransform();
        std::vector<dvec2> outv = outShapeVec( shape, T );
    
        if(T[0][0]!=T[1][1])
            std::reverse(outv.begin(), outv.end());

        if(i->getT1() == chosen_x && i->getT2() == chosen_y && i->getAspect()==0)
            xy << outv[0].x, outv[0].y;		

        if(i->getT1() == chosen_x+1 && i->getT2() == chosen_y && i->getAspect()==0)
            x1y << outv[0].x, outv[0].y;		

        if(i->getT1() == chosen_x && i->getT2() == chosen_y+1 && i->getAspect()==0)
            xy1 << outv[0].x, outv[0].y;		

        if(i->getT1() == chosen_x+1 && i->getT2() == chosen_y+1 && i->getAspect()==0)
            x1y1 << outv[0].x, outv[0].y;		

    }

    transf.setZero();
    transf.head<2>() = x1y - xy;
    transf.tail<2>() = xy1 - xy;
    
    T temp1 = transf[0];
    T temp2 = transf[3];

    transf[0] = transf[2];
    transf[3] = transf[1];

    transf[1] = temp2;
    transf[2] = temp1;

    if(transf[0]<0)
        transf.head<2>() *= -1;
    if(transf[3]<0)
        transf.tail<2>() *= -1;

    T temp = transf[2];
    transf[2] = transf[1];
    transf[1] = temp;
    xy_shift = xy;
}

void Tiling2D::saveClip(const ClipperLib::Paths& final_shape, 
        const Vector<T, 8>& periodic, T mult,
        const std::string& filename, bool add_box)
{
    std::ofstream out(filename);
    if (add_box)
        for (int i = 0; i < 4; i++)
            out << "v " <<  periodic.segment<2>(i * 2).transpose() << " 0" << std::endl;
    for (auto polygon : final_shape)
    {
        for (auto vtx : polygon)
        {
            out << "v " << vtx.X / mult << " " << vtx.Y / mult << " 0" << std::endl;
        }
    }
    int cnt = 5;
    if (add_box)
    {
        out << "l 1 2" << std::endl;
        out << "l 2 3" << std::endl;
        out << "l 3 4" << std::endl;
        out << "l 4 1" << std::endl;
    }
    else
        cnt = 1;
    for (auto polygon : final_shape)
    {
        for (int i = 0; i < polygon.size(); i++)
        {
            int j = (i + 1) % polygon.size();
            out << "l " << cnt + i << " " << cnt + j << std::endl;
        }
        cnt += polygon.size();
    }
    out.close();
}