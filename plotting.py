import numpy as np
import matplotlib.pyplot as plt


# Generate example elevation data
def generate_elevation_data(size=(10, 10)):
    x = np.linspace(-5, 5, size[0])
    y = np.linspace(-5, 5, size[1])
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X ** 2 + Y ** 2)) * 120  # Scale to get values around 100
    return X, Y, Z


# Generate color map based on Z values
def generate_color_data(Z, threshold):
    colors = np.zeros((Z.shape[0] - 1, Z.shape[1] - 1, 3))
    colors[:,:] = [0, 1, 0]
    strict_above = np.logical_and(
        np.logical_and(Z[1:,1:] >= threshold, Z[1:,:-1] >= threshold),
        np.logical_and(Z[:-1,1:] >= threshold, Z[:-1,:-1] >= threshold))
    just_above = np.logical_or(
        np.logical_or(Z[1:,1:] >= threshold, Z[1:,:-1] >= threshold),
        np.logical_or(Z[:-1,1:] >= threshold, Z[:-1,:-1] >= threshold))

    colors[just_above] = [0,0,1]
    colors[strict_above] = [1,0,0]
    return colors


# Plot function
def plot_3d_surface(X, Y, Z, colors):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    rstride = X.shape[0] // 80
    cstride = X.shape[1] // 80

    surf = ax.plot_surface(X, Y, Z, facecolors=colors, edgecolor='k', linewidth=0.3, rstride=rstride, cstride=cstride)

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Elevation")
    plt.show()

def plot_elevation(elevations: np.ndarray, colors: np.ndarray):
    dim0 = elevations.shape[0]
    dim1 = elevations.shape[1]
    assert colors.shape[0] == dim0 - 1
    assert colors.shape[1] == dim1 - 1
    x = np.linspace(0, dim0, dim0)
    y = np.linspace(0, dim1, dim1)
    X, Y = np.meshgrid(x, y)
    plot_3d_surface(X, Y, elevations, colors)

if __name__ == '__main__':
    # Generate data and plot
    size = (200, 200)  # Grid size
    X, Y, Z = generate_elevation_data(size)
    colors = generate_color_data(Z, 100)
    plot_3d_surface(X, Y, Z, colors)
