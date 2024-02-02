import numpy as np
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt


def bezier(x, t=30):
    
    ts = np.linspace(0, 1, t)

    # 10.Keyframe Animation Smooth Curves.pdf page 77.
    # Keyframe interpolation.pdf page 25.

    # --- code here
    results = []
    mb = np.array(
    [
        [ -1. ,  3. , -3. , 1.],
        [  3. , -6. ,  3. , 0.],
        [ -3. ,  3. ,  0. , 0.],
        [  1. ,  0. ,  0. , 0.]
    ])
    for t in ts:
        tv = np.array([t**3, t**2, t, 1])
        result = np.dot(tv, mb)
        results.append(np.dot(result, x))

    # --- end
    return results

def b_spline(x):
    # Keyframe interpolation.pdf page 52.

    results = []
    for i in range(len(x)-3):
        # --- code here

        v0 = 1/6 * x[i]   + 2/3 * x[i+1] + 1/6 * x[i+2]
        v1 = 2/3 * x[i+1] + 1/3 * x[i+2]
        v2 = 1/3 * x[i+1] + 2/3 * x[i+2]
        v3 = 1/6 * x[i+1] + 2/3 * x[i+2] + 1/6 * x[i+3]
        # v = [0., 0., 0., 0.]
        v = [v0, v1, v2, v3]

        # end
        results.extend(bezier(v))

    return results


x_points = [-0.9, -0.3, 0.3, 0.9]
y_points = [-0.6, 0.6, 0.6, -0.6]

def draw():
    fig = plt.figure(figsize=(6,6))
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    
    line = plt.plot(x_points, y_points, 'o--')[0]
    bez_line = plt.plot(bezier(x_points), bezier(y_points))[0]
    b_sep_line = plt.plot(b_spline(x_points), b_spline(y_points))[0]
    eh = evt_hanlder(fig, line, b_sep_line)

    plt.show()

class evt_hanlder:
    def __init__(self, fig: plt.Figure, line, b_sep_line):
        self.fig = fig
        self.h_id = fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.line = line
        self.b_sep_line = b_sep_line
    
    def on_click(self, event):
        if event.inaxes!= self.line.axes: return
        if event.button == MouseButton.LEFT:
            x_points.append(event.xdata)
            y_points.append(event.ydata)
        elif event.button == MouseButton.RIGHT and len(x_points) > 4:
            x_points.pop()
            y_points.pop()
        self.line.set_data(x_points, y_points)
        self.b_sep_line.set_data(b_spline(x_points), b_spline(y_points))
        self.fig.canvas.draw()


if __name__ == "__main__":
    draw()