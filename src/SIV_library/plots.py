import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from IPython.display import HTML


def plot_flow(u, v, grid_spacing: int = 50, show_img: bool = True, fn: str | None = None):
    num_frames, height, width = u.shape
    u, v = u.cpu().numpy(), v.cpu().numpy()

    fig, ax = plt.subplots()

    abs_velocities = np.sqrt(u**2 + v**2)
    min_abs, max_abs = np.min(abs_velocities), np.max(abs_velocities)

    u0, v0 = u[0], v[0]

    # https://stackoverflow.com/questions/24116027/slicing-arrays-with-meshgrid-array-indices-in-numpy
    x, y = np.meshgrid(np.arange(grid_spacing, width - grid_spacing, grid_spacing),
                       np.arange(grid_spacing, height - grid_spacing, grid_spacing))
    xx, yy = x[...].astype(np.uint16), y[...].astype(np.uint16)

    vx0, vy0 = u0[yy, xx], v0[yy, xx]
    vectors = ax.quiver(x, y, vx0, -vy0, color='red', scale=1, scale_units='xy', angles='xy')

    image = ax.imshow(abs_velocities[0], vmin=min_abs, vmax=max_abs, cmap='magma')

    ax.set_axis_off()
    ax.set_title("Optical Flow")
    fig.colorbar(image, ax=ax)

    def update(idx):
        image.set_data(abs_velocities[idx])

        ui, vi = u[idx], v[idx]
        vx, vy = ui[yy, xx], vi[yy, xx]
        vectors.set_UVC(vx, -vy)

        ax.set_title(f"t = {idx/240:.3f} s")
        return image, vectors

    ani = animation.FuncAnimation(fig=fig, func=update, frames=u.shape[0], interval=1000/30)

    if fn is not None:
        writer = animation.PillowWriter(fps=10)
        ani.save(f'../Test Data/{fn}', writer=writer)

    HTML(ani.to_jshtml())

    return ani
    #
    # plt.show()
