import numpy as np
import matplotlib.pyplot as plt
from Data.Preprocess import join_path


def draw_heat_new(array, ax, xlabel, ylabel, vmin, vmax):
    im = ax.imshow(array, cmap='Blues', vmin=vmin, vmax=vmax)
    ax.tick_params(axis='x')
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(['2', '3', '4', '5'])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['2', '3', '4', '5'])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    thresh = array.max() / 2.
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            ax.text(j, i, str(array[i, j]),
                     ha="center", va="center", color="white" if array[i, j] > thresh else "black")
    return im


def draw_confusion_matrix(save_dir):
    data1 = np.array([[20, 2, 2, 2],
                      [7, 9, 5, 3],
                      [4, 10, 22, 10],
                      [0, 4, 8, 38]])
    data2 = np.array([[19, 5, 2, 0],
                      [3, 15, 4, 2],
                      [4, 14, 25, 3],
                      [0, 4, 11, 35]])

    xlabel = 'Label'

    f, ax = plt.subplots(1, 2, figsize=(10,4))

    draw_heat_new(data1, ax[0], xlabel=xlabel, ylabel='ResNet50 ', vmin=0, vmax=38)
    im = draw_heat_new(data2, ax[1], xlabel=xlabel, ylabel='ResNet50-UDM ', vmin=0, vmax=38)

    f.subplots_adjust(right=0.99)
    cbar_ax = f.add_axes([0.95, 0.145, 0.01, 0.8])
    plt.colorbar(im, cax=cbar_ax)
    f.tight_layout()

    plt.savefig(join_path(save_dir, 'confusion_matrix.png'), format='png', dpi=500)
    plt.show()


def draw_udm(data, pred, label, t, save_path=None):
    plt.scatter(data, (np.array(label)+2).astype(np.int), c=pred)

    y_lin = np.arange(0,5.3,0.1)
    plt.plot([t[0]]*y_lin.shape[0], y_lin, linestyle='--', color='blue')
    plt.plot([t[1]]*y_lin.shape[0], y_lin, linestyle='--', color='green')
    plt.plot([t[2]]*y_lin.shape[0], y_lin, linestyle='--', color='red')

    plt.xlabel('Model output')
    # plt.ylabel('Radiologists\' prediction')
    plt.ylabel('Label')
    plt.ylim(ymin=1)
    plt.legend(['$\lambda_{23}$', '$\lambda_{34}$', '$\lambda_{45}$'])

    if save_path is not None:
        plt.savefig(save_path, format='png', dpi=1000)

    plt.show()
