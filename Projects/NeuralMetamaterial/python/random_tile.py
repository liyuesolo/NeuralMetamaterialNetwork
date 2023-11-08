import random
import math

import p5

from tactile import IsohedralTiling, tiling_types, EdgeShape, mul, Point

RENDER_WIDTH = 1000
RENDER_HEIGHT = 1000


def make_random_tiling():
    # Construct a tiling
    # tiling = IsohedralTiling(random.choice(tiling_types))
    tiling = IsohedralTiling(21)

    # Randomize the tiling vertex parameters
    ps = tiling.parameters
    # for i in range(tiling.num_parameters):
        # ps[i] += (random.random()-.5)*0.2
    tiling.parameters = ps

    # Make some random edge shapes.  Note that here, we sidestep the 
    # potential complexity of using .shapes vs. .parts by checking
    # ahead of time what the intrinsic edge shape is and building
    # Bezier control points that have all necessary symmetries.
    # See https://github.com/isohedral/tactile-js/ for more info.

    edges = []
    for shp in tiling.edge_shapes:
        ej = []
        if shp == EdgeShape.I:
            # Must be a straight line.
            pass
        elif shp == EdgeShape.J:
            # Anything works for J
            ej.append(Point(random.random()*0.6, random.random() - 0.5))
            ej.append(Point(random.random()*0.6 + 0.4, random.random() - 0.5))
        elif shp == EdgeShape.S:
            # 180-degree rotational symmetry
            ej.append(Point(random.random()*0.6, random.random() - 0.5))
            ej.append(Point(1.0 - ej[0].x, -ej[0].y))
        elif shp == EdgeShape.U:
            # Symmetry after reflecting/flipping across length.
            ej.append(Point(random.random()*0.6, random.random() - 0.5))
            ej.append(Point(1.0 - ej[0].x, ej[0].y))

        edges.append( ej )

    return tiling, edges


def draw_random_tiling(tx=0, ty=0, scale=100):
        
    tiling, edges = make_random_tiling()

    # Make some random colors.
    cols = []
    for i in range(3):
        cols.append([random.randint(0,255), random.randint(0,255), random.randint(0,255)])

    # Define a world-to-screen transformation matrix that scales by 100x.
    ST = [scale, 0.0, tx, 0.0, scale, ty]

    for i in tiling.fill_region_bounds( -2, -2, 12, 12 ):
        T = mul( ST, i.T )
        p5.fill(*cols[ tiling.get_color( i.t1, i.t2, i.aspect ) ])

        start = True
        p5.begin_shape()

        for si in tiling.shapes:
            S = mul( T, si.T )
            # Make the edge start at (0,0)
            seg = [ mul( S, Point(0., 0.)) ]

            if si.shape != EdgeShape.I:
                # The edge isn't just a straight line.
                ej = edges[ si.id ]
                seg.append( mul( S, ej[0] ) )
                seg.append( mul( S, ej[1] ) )

            # Make the edge end at (1,0)
            seg.append( mul( S, Point(1., 0.)) )

            if si.rev:
                seg.reverse()

            if start:
                start = False
                p5.vertex(seg[0].x, seg[0].y )
            if len(seg) == 2:
                p5.vertex(seg[1].x, seg[1].y )
            else:
                p5.bezier_vertex(
                    seg[1].x, seg[1].y, 
                    seg[2].x, seg[2].y, 
                    seg[3].x, seg[3].y)

        p5.end_shape()


def setup():

    global RENDER_WIDTH
    global RENDER_HEIGHT

    p5.size(RENDER_WIDTH, RENDER_HEIGHT)
    # p5.no_loop()


def draw():
    # p5.stroke_weight(1)
    # p5.stroke(0,0,0)
    p5.no_stroke()
    p5.background(0)
    draw_random_tiling()
    # p5.save_frame()


if __name__ == '__main__':

    p5.run()