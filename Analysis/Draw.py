import numpy as np
import matplotlib.pyplot as plt
from Data.Preprocess import join_path
import pandas as pd
from Visualization.Statistics import add_roc


# 1. draw heat map
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
    data_ce = np.array([[22, 2, 0, 2],
                      [6, 11, 5, 4],
                      [8, 10, 31, 6],
                      [2, 5, 4, 38]])
    data_sr = np.array([[19, 6, 0, 1],
                      [5, 12, 7, 2],
                      [6, 13, 31, 5],
                      [0, 4, 10, 35]])
    data_encode = np.array([[18, 6, 1, 1],
                      [7, 11, 6, 2],
                      [5, 10, 32, 8],
                      [1, 3, 5, 40]])
    data_udm = np.array([[17, 8, 1, 0],
                      [5, 13, 6, 2],
                      [4, 7, 36, 8],
                      [0, 3, 9, 37]])

    xlabel = 'Label'

    f, ax = plt.subplots(2, 2, figsize=(12,10))

    draw_heat_new(data_ce, ax[0,0], xlabel=xlabel, ylabel='Cross Entropy ', vmin=0, vmax=38)
    draw_heat_new(data_sr, ax[0,1], xlabel=xlabel, ylabel='Soft Regression ', vmin=0, vmax=38)
    draw_heat_new(data_encode, ax[1,0], xlabel=xlabel, ylabel='Encoding ', vmin=0, vmax=38)
    im = draw_heat_new(data_udm, ax[1,1], xlabel=xlabel, ylabel='UDM ', vmin=0, vmax=38)

    f.subplots_adjust(right=0.99)
    cbar_ax = f.add_axes([0.95, 0.145, 0.01, 0.8])
    plt.colorbar(im, cax=cbar_ax)
    f.tight_layout()

    plt.savefig(join_path(save_dir, 'confusion_matrix.png'), format='png', dpi=500)
    plt.show()


# 2. draw udm
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


# 3. draw roc
def draw_roc(result_path, save_dir):
    df = pd.read_csv(result_path)
    gs = df['gs']
    radiologist = df['radiologist']
    model = df['pred']

    gs = [1 if x>0 else 0 for x in gs]
    radiologist = [(x-1)*0.25 for x in radiologist]
    model = [(x-1)*0.25 for x in model]


    add_roc(model, gs, label_name='ResNet-UDM')
    add_roc(radiologist, gs, label_name='Radiologist')
    plt.tight_layout()
    plt.savefig(join_path(save_dir, 'ROC.tiff'), format='tiff', dpi=500)
    plt.show()


if __name__ == '__main__':
    result_dir = r'/homes/rqyu/Projects/PI-RADS_Classification/Results'
    save_dir = r'/homes/rqyu/Projects/PI-RADS_Classification/figure'

    # draw_confusion_matrix(save_dir)
    draw_roc(join_path(result_dir, 'UDM loc select', 'test result.csv'), save_dir)
