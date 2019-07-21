import os
import cv2
from matplotlib import pyplot as plt
from matplotlib import gridspec as grid
from itertools import cycle
import numpy as np

"""Some handy tools for visualization"""


def plot_epoch(prog, alpha, epochs):
    """Plot generated images, gen-loss, discr-loss, plausibility, alpha"""

    n_inter = len(prog.history["gen"])

    fig = plt.figure(figsize=(15, 5))
    outer = grid.GridSpec(1, 2)

    # Generated images
    inner1 = grid.GridSpecFromSubplotSpec(2, 2, outer[0])
    for _ in range(4):
        ax = plt.Subplot(fig, inner1[_])
        ax.imshow(prog.generate(alpha=alpha)[0], cmap='magma')
        ax.set(xticks=[], yticks=[])
        fig.add_subplot(ax)

    # Discr loss
    inner2 = grid.GridSpecFromSubplotSpec(2, 1, outer[1])
    ax = plt.Subplot(fig, inner2[0])
    ax.plot(prog.history['discr'][-epochs:])
    ax.grid()
    ax.set(xticks=[], xlim=(0, epochs), ylabel='DISCR')
    fig.add_subplot(ax)

    # Gen loss
    ax = plt.Subplot(fig, inner2[1])
    ax.plot(prog.history['gen'][-epochs:], label='gen', color='orange')
    ax.grid(axis='y')
    ax.set(xlim=(0, epochs), ylabel='GEN')
    fig.add_subplot(ax)

    img = prog.generate(alpha=alpha)
    fig.suptitle(f'plausibility: {prog.get_proba(img, alpha)[0][0]:.3f}\n{n_inter}|{alpha:.3f}', size=15)

    plt.show()


def load_img_gen(path, img_shape, n_img):
    """Image cycle iterator for NN testing"""

    X = []
    for x in np.random.choice([file for file in os.listdir(os.path.join(path, 'train/'))\
                               if file.startswith('180.png')], n_img):
        img = cv2.imread(os.path.join(path, 'train/', x))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_shape, img_shape))
        X.append(img/255)
    X = np.asarray([X])

    return cycle(X)


def plot_samples(img, n_img=9):
    plt.figure(figsize=(5, 5))

    n_rows = int(np.sqrt(n_img))
    X = next(img)
    for _ in range(min(n_img, len(X))):
        plt.subplot(n_rows, n_rows, _ + 1)
        plt.imshow(X[_])
        plt.xticks([])
        plt.yticks([])
    plt.show()
