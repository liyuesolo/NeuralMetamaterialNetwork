import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.patches import Polygon
from scipy.interpolate import splev, splrep

def pol2cart(r,theta):
    '''
    Parameters:
    - r: float, vector amplitude
    - theta: float, vector angle
    Returns:
    - x: float, x coord. of vector end
    - y: float, y coord. of vector end
    '''

    z = r * np.exp(1j * theta)
    x, y = z.real, z.imag

    return x, y

def cart2pol(x, y):
    '''
    Parameters:
    - x: float, x coord. of vector end
    - y: float, y coord. of vector end
    Returns:
    - r: float, vector amplitude
    - theta: float, vector angle
    '''

    z = x + y * 1j
    r,theta = np.abs(z), np.angle(z)

    return r,theta

def dist(x, y):
    """
    Return the distance between two points.
    """
    d = x - y
    return np.sqrt(np.dot(d, d))


def dist_point_to_segment(p, s0, s1):
    """
    Get the distance of a point to a segment.
      *p*, *s0*, *s1* are *xy* sequences
    This algorithm from
    http://geomalgorithms.com/a02-_lines.html
    """
    v = s1 - s0
    w = p - s0
    c1 = np.dot(w, v)
    if c1 <= 0:
        return dist(p, s0)
    c2 = np.dot(v, v)
    if c2 <= c1:
        return dist(p, s1)
    b = c1 / c2
    pb = s0 + b * v
    return dist(p, pb)


def interpolate(x, y):
    x, y = x, y##self.poly.xy[:].T
    
    shift = np.min(y)
    # y -= shift
    _x, _y = pol2cart(y, x)
    i = np.arange(len(_x))
    
    interp_i = np.linspace(0, i.max(), 100 * i.max())
    
    xi = interp1d(i, _x, kind='cubic')(interp_i)
    yi = interp1d(i, _y, kind='cubic')(interp_i)

    r, theta =cart2pol(xi,yi)
    # r += shift
    # print(np.min(r))
    # print(xi, yi)
    # for i in range(len(theta)):
        
    #     if theta[i] < 0:
    #         theta[i] += np.pi * 2.0
        
    return theta, r

class MacroPropertyModifier:
    """
    A polygon editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

      'd' delete the vertex under point

      'i' insert a vertex at point.  You must be within epsilon of the
          line connecting two existing vertices

    """

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, ax, poly, theta, thetas_nn):
        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure '
                               'or canvas before defining the interactor')
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly
        x, y = zip(*self.poly.xy)
        
        _x, _y = pol2cart(np.array(y), np.array(x))
        self.center = 0.5 * (np.array([np.min(_x), np.min(_y)]) + np.array([np.max(_x), np.max(_y)]))

        self.theta = theta
        self.n_half = len(self.theta) // 2
        # print(len(self.theta))
        # print(self.n_half)
        
        self.line = Line2D(x, y, ls = '--',
                           marker='o', markerfacecolor='r',
                           linewidth = 3.0,
                           color = 'purple',
                           markersize = 10.0,
                           animated=True)
        self.ax.add_line(self.line)

        xs, ys = self.poly.xy[:].T
        xs, ys = interpolate(xs, ys)
        self.spline = Line2D(xs, ys, linewidth = 5.0, animated = True)
        self.ax.add_line(self.spline)
        self.thetas_nn = thetas_nn
        

        self.cid = self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        canvas.mpl_connect('draw_event', self.on_draw)
        canvas.mpl_connect('button_press_event', self.on_button_press)
        canvas.mpl_connect('key_press_event', self.on_key_press)
        canvas.mpl_connect('button_release_event', self.on_button_release)
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas = canvas

    def on_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.ax.draw_artist(self.spline)
        # do not need to blit here, this will fire before the screen is
        # updated

    def poly_changed(self, poly):
        """This method is called whenever the pathpatch object is called."""
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.hypot(xt - event.x, yt - event.y)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def on_button_press(self, event):
        """Callback for mouse button presses."""
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None

    def on_key_press(self, event):
        """Callback for key presses."""
        if not event.inaxes:
            return
        if event.key == 't':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        elif event.key == 'd':
            ind = self.get_ind_under_point(event)
            if ind is not None:
                self.poly.xy = np.delete(self.poly.xy,
                                         ind, axis=0)
                self.line.set_data(zip(*self.poly.xy))
        elif event.key == 'i':
            xys = self.poly.get_transform().transform(self.poly.xy)
            p = event.x, event.y  # display coords
            for i in range(len(xys) - 1):
                s0 = xys[i]
                s1 = xys[i + 1]
                d = dist_point_to_segment(p, s0, s1)
                if d <= self.epsilon:
                    self.poly.xy = np.insert(
                        self.poly.xy, i+1,
                        [event.xdata, event.ydata],
                        axis=0)
                    self.line.set_data(zip(*self.poly.xy))
                    xs, ys = self.poly.xy[:].T
                    xs, ys = interpolate(xs, ys)
                    self.spline.set_data(xs, ys)
                    break
        elif event.key == 'r':
            # plt.polar(self.thetas_nn, stiffness, zorder=5,linewidth=4)
            
            plt.savefig("current.png", dpi=300)
            # plt.show()        
        elif event.key == 'a':
            xs, ys = self.poly.xy[:].T
            xs, ys = interpolate(xs, ys)
            for i in range(len(xs)):
                if xs[i] < 0:
                    xs[i] += 2.0 * np.pi
            xs[-1] = 2.0 * np.pi
            
            i = np.arange(len(xs))
            xi = interp1d(xs, i, kind='linear')(self.thetas_nn)
            stiffness = interp1d(i, ys, kind='cubic')(xi)
            # spl = splrep(xs, ys)
            # stiffness = splev(self.thetas_nn, spl)
            f = open("stiffness_target.txt", "w+")
            f.write("np.array([")
            for i in range(len(stiffness) - 1):
                f.write(str(stiffness[i]) + ", ")
            f.write(str(stiffness[-1]) + "])\n")
            f.close()
            
            # plt.polar(self.thetas_nn, stiffness, zorder=5,linewidth=4)
            # plt.savefig("polar.png", dpi=300)
            # plt.show()        

        if self.line.stale:
            self.canvas.draw_idle()

    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata

        self.poly.xy[self._ind] = x, y

        oppo = (self._ind + self.n_half) % (2 * self.n_half)
        
        _x, _y = pol2cart(y, x)
        a = 2 * self.center - np.array([_x, _y])
        r, theta = cart2pol(a[0], a[1])
        
        self.poly.xy[oppo] = theta, r

        if self._ind == 0:
            self.poly.xy[-1] = x, y
        elif self._ind == len(self.poly.xy) - 1:
            self.poly.xy[0] = x, y
        
        self.line.set_data(zip(*self.poly.xy))

        xs, ys = self.poly.xy[:].T
        xs, ys = interpolate(xs, ys)
        self.spline.set_data(xs, ys)


        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.ax.draw_artist(self.spline)
        
        self.canvas.blit(self.ax.bbox)