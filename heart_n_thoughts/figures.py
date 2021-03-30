import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors


def subplot_pca(fig, axes, pattern, scree, i, name):
    # pca
    f = axes[i, 0].matshow(pattern, cmap="RdBu_r")
    axes[i, 0].set_title(name)
    axes[i, 0].set_xlabel("Components")
    axes[i, 0].set_xticklabels(range(5))
    axes[i, 0].set_yticks(range(len(pattern.index)))
    axes[i, 0].set_yticklabels(pattern.index)

    divider = make_axes_locatable(axes[i, 0])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(f, cax=cax, orientation="vertical")

    # scree plot
    axes[i, 1].plot(scree, "-o")
    axes[i, 1].set_xlabel("Components")
    axes[i, 1].set_xticks(range(len(pattern.index)))
    axes[i, 1].set_xticklabels(range(1, 14))
