import numpy as np
import matplotlib.pyplot as plt

# Plot function
def plot_3d_surface(plot_args, X, Y, Z, colors):
    fig, nrows, ncols, idx, title = plot_args

    ax = fig.add_subplot(nrows, ncols, idx, projection='3d')
    rstride = X.shape[0] // 80
    cstride = X.shape[1] // 80

    surf = ax.plot_surface(X, Y, Z, facecolors=colors, edgecolor='k', linewidth=0.3, rstride=rstride, cstride=cstride)

    ax.set_xlabel('+ Long "')
    ax.set_ylabel('+ Lat "')
    ax.set_zlabel("Elevation (m)")
    ax.set_title(title)

def plot_elevation(plot_args, elevations: np.ndarray, colors: np.ndarray):
    dim0 = elevations.shape[0]
    dim1 = elevations.shape[1]
    assert colors.shape[0] == dim0 - 1
    assert colors.shape[1] == dim1 - 1
    x = np.linspace(-dim0 // 2, dim0 - dim0 // 2, dim0)
    y = np.linspace(-dim1 // 2, dim1 - dim1 // 2, dim1)
    X, Y = np.meshgrid(x, y)
    plot_3d_surface(plot_args, X, Y, elevations[::-1,:], colors[::-1,:,:])

def plot_all(plots, show=False):
    nrows = len(plots) // 2 + 1
    fig = plt.figure(figsize=(10, 5 * nrows))
    for i, (elevations, colors, title) in enumerate(plots, 1):
        subplot_args = fig, nrows, 2, i, title
        plot_elevation(subplot_args, elevations, colors)
    if show:
        plt.show()
    else:
        fig.savefig('out.pdf', format='pdf')
