import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import matplotlib.colors as mcolors

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

def plot_multi_elevation(plots, show=False):
    nrows = len(plots) // 2 + 1
    fig = plt.figure(figsize=(10, 5 * nrows))
    for i, (elevations, colors, title) in enumerate(plots, 1):
        subplot_args = fig, nrows, 2, i, title
        plot_elevation(subplot_args, elevations, colors)
    if show:
        plt.show()
    else:
        fig.savefig('out.pdf', format='pdf')

def random_mpl_color():
    colors = ['b', 'g', 'r', 'c', 'm', 'k', 'w', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:cyan']
    return random.choice(colors)

def plot_result(result_path: str):
    # Equalize the plot's height and with
    y_scale = 1 / 1000
    # Create a meshgrid to fill the area above/right of the curve
    X, Y = np.meshgrid(np.linspace(0.0, 4, 400), np.linspace(0.0, 3000, 400) * y_scale)

    # Plot the region boundary
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_xlabel('1/sin(bottom_angle)')
    ax.set_ylabel('Cone height (m)')
    ax.set_xlim(1.0, 3.5)
    ax.set_ylim(100 * y_scale, 3300 * y_scale)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y / y_scale:.0f}'))  # Scale labels

    y_full_range = np.linspace(300 * y_scale, 3300 * y_scale, 16)
    with open(result_path, 'r') as f:
        for line in f.readlines()[1:]:
            line = line.replace('\n', '')
            name, eth, true_elevation, obs_elevation, *sines = line.split(',')
            if float(eth) == 0:
                # Invalid result
                continue
            mx = []
            my = []
            for drop, sine in zip(y_full_range, sines):
                sine = float(sine)
                mx.append(1 / sine)
                my.append(drop)

            max_drop = my[-1]
            if max_drop < 1400 * y_scale:
                continue

            color = random_mpl_color()
            ax.plot(mx, my, color, linewidth=1)
            name = name.split('/')[0]
            ax.text(mx[-1] + random.randrange(-10, 10) / 100,
                    my[-1] + random.randrange(-100,100) * y_scale, name, fontsize=10, color=color, ha='center', va='bottom')

    # Plot classification regions
    mask0 = Y / y_scale > X * 1200
    ax.imshow(mask0, extent=(0, 4, 0, 3300 * y_scale), origin='lower', cmap='Purples', alpha=0.7)
    mask1 = np.logical_and(Y / y_scale > X * 900, Y / y_scale <= X * 1200)
    ax.imshow(mask1, extent=(0, 4, 0, 3300 * y_scale), origin='lower', cmap='Oranges', alpha=0.6)
    mask2 = np.logical_and(Y / y_scale > X * 600, Y / y_scale <= X * 900)
    ax.imshow(mask2, extent=(0, 4, 0, 3300 * y_scale), origin='lower', cmap='Blues', alpha=0.5)
    mask3 = np.logical_and(Y / y_scale > X * 300, Y / y_scale <= X * 600)
    ax.imshow(mask3, extent=(0, 4, 0, 3300 * y_scale), origin='lower', cmap='Greens', alpha=0.4)
    maskInf = Y / y_scale <= X * 300
    ax.imshow(maskInf, extent=(0, 4, 0, 3300 * y_scale), origin='lower', cmap='Grays', alpha=0.3)

    ax.text(1.5, 3000 * y_scale, "Alien\n1200+", fontsize=15, color='r')
    ax.text(2.5, 3000 * y_scale, "Ultra\n900+", fontsize=15, color='c')
    ax.text(3.2, 3000 * y_scale, "Superb\n600+", fontsize=15, color='m')
    ax.text(3, 1200 * y_scale, "Notable\n300+", fontsize=15, color='b')
    ax.text(3, 500 * y_scale, "Common", fontsize=15, color='w')

    ax.grid()
    plt.show()


if __name__ == '__main__':
    plot_result('output.csv')