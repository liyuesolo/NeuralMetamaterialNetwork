#ifndef POISSON_DISK_H
#define POISSON_DISK_H

#include <iostream>
#include <vector>
#include <string>

#include "cyVector.h"
#include "cySampleElim.h"

#include "VecMatDef.h"

namespace PoissonDisk
{
    using TV3 = Vector<T, 3>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;

    template<int dim>
    void sampleNDBox(const Vector<T, dim>& min_corner, const Vector<T, dim>& max_corner,
        int n_samples, VectorXT& samples)
    {
        cy::WeightedSampleElimination< cy::Vec<T, dim>, T, dim, int > wse;

        cy::Vec<T, dim> minimumBounds, maximumBounds;
        for (int d = 0; d < dim; d++)
        {
            minimumBounds[d] = min_corner[d];
            maximumBounds[d] = max_corner[d];
        }

        wse.SetBoundsMin( minimumBounds );
        wse.SetBoundsMax( maximumBounds );


        std::vector< cy::Vec<T, dim> > inputPoints(n_samples * 10);
        for ( size_t i=0; i<inputPoints.size(); i++ ) 
        {
            for (int d = 0; d < dim; d++)
                inputPoints[i][d] = minimumBounds[d] + (T) rand() / RAND_MAX * (maximumBounds[d] - minimumBounds[d]);
        }

        std::vector< cy::Vec<T, dim> > outputPoints(n_samples);

        float d_max = 2 * wse.GetMaxPoissonDiskRadius( dim, outputPoints.size() );
        wse.Eliminate( inputPoints.data(), inputPoints.size(), 
                    outputPoints.data(), outputPoints.size(),
                    true,
                    d_max );
        samples.resize(n_samples * dim);
        for (int i = 0; i < n_samples; i++)
        {
            for (int d = 0; d < dim; d++)
                samples[i * dim + d] = outputPoints[i][d];    

            // std::cout << samples.template segment<dim>(i * dim).transpose() << std::endl;
        }
    }
};

#endif // !POISSON_DISK_H