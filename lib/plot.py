"""
Functions for plotting various visualizations/graphs
"""

__author__ = 'clee'
# Python 3.6.1

# Standard lib
import itertools

# Additional lib
import matplotlib.pyplot as plt
import numpy as np


def plot_cm(cm, cls_names,
            normalize=True,
            title='Confusion matrix',
            cmap=None,
            path_save=None,
            show=False,
            figsize=(10, 10),
            true_cls_names=None,
            fontsize=28,
            true_label='True Label',
            pred_label='Predicted Label'):
    """
    Plot color coded confusion matrix

    :param cm:              2D numpy.ndarray    Output of sklearn.metrics.confusion_matrix
    :param cls_names:       list of str         List of class names ordered by corresponding class label
    :param normalize:       bool                Whether or not to display percentages or absolute numbers
    :param title:           str                 Title of confusion matrix plot
    :param cmap:            matplotlib.colors.LinearSegmentedColormap
    :param path_save:       str                 Output location to save png to (if desired)
    :param show             bool                Whether or not to display plot
    :param figsize          2-tuple of ints     Size of the plot
    :param true_cls_names   list of str         List of class names for true class axis
    :param fontsize         int                 Size of text font on CM
    :param true_label       str                 True axis name
    :param pred_label       str                 Predicted axis name

    :return:                None
    """

    if normalize:
        cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100
        cm = np.around(cm, decimals=2)

        non_diag_ind = np.where(~np.eye(cm.shape[0], dtype=bool))
        cm[non_diag_ind] = -cm[non_diag_ind]

    if cmap is None:
        # Define custom default CM colormap
        cdict = {'green': ((0.0, 0.0, 0.0),
                           (0.25, 0.0, 0.0),
                           (0.5, 0.30, 1.0),
                           (0.75, 1.0, 1.0),
                           (1.0, 0.4, 1.0)),

                 'blue': ((0.0, 0.0, 0.0),
                          (0.25, 0.0, 0.0),
                          (0.5, 0.2, 0.8),
                          (0.75, 0.0, 0.0),
                          (1.0, 0.0, 0.0)),

                 'red': ((0.0, 0.0, 0.0),
                         (0.25, 0.2, 0.2),
                         (0.5, 1.0, 1.0),
                         (0.75, 0.5, 0.5),
                         (1.0, 0.0, 0.0))
                 }
        plt.register_cmap(name='WeightedRdYlGn', data=cdict)
        cmap = plt.get_cmap('WeightedRdYlGn')

    fig = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=-100, vmax=100)
    plt.title(title, fontsize=fontsize)
    # plt.colorbar()
    tick_marks = np.arange(len(cls_names))
    plt.xticks(tick_marks, cls_names, rotation=45, fontsize=fontsize)
    if true_cls_names is None:
        plt.yticks(tick_marks, cls_names, fontsize=fontsize)
    else:
        true_tick_marks = np.arange(len(true_cls_names))
        plt.yticks(true_tick_marks, true_cls_names, fontsize=fontsize)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, abs(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if abs(cm[i, j]) > thresh else "black",
                 fontsize=fontsize)

    plt.tight_layout()
    plt.ylabel(true_label, fontsize=fontsize)
    plt.xlabel(pred_label, fontsize=fontsize)
    if path_save is not None:
        plt.savefig(path_save, bbox_inches='tight')
    if show:
        plt.show()


def plot_scores(scores, cls_names,
                plt_title="Scores",
                cls_title="Class",
                precision=3,
                path_save=None,
                show=False,
                figsize=(20,20),):
    """
    Plot color coded score vector

    :param scores:          (n, m)-shaped array-like            Score vector from Keras model predict(); n samples, m classes
    :param cls_names:       m-long list of ints                 List of class names ordered by corresponding class label
    :param plt_title:       str                                 Plot title
    :param cls_title:       str                                 x-axis title
    :param precision:       int                                 Decimal precision of scores
    :param path_save:       str                                 Output location to save png to (if desired)
    :param show             bool                                Whether or not to display plot
    :param figsize          2-tuple of ints                     Size of the plot

    :return:                None
    """
    plt.figure(figsize=figsize)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.Blues, aspect=0.25, clim=[0,1])
    plt.xticks(np.arange(len(cls_names)), cls_names)
    plt.title(plt_title)
    plt.tick_params(axis='y', left='off', right='off', labelleft='off')

    thresh = 0.5
    for i, j in itertools.product(range(scores.shape[0]), range(scores.shape[1])):
        plt.text(j, i, '{0:.{1}%}'.format(scores[i, j], precision),
                 horizontalalignment="center",
                 color="white" if scores[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel(cls_title)
    if path_save is not None:
        plt.savefig(path_save, bbox_inches='tight')
    if show:
        plt.show()


def plot_channels(channels,
                  cols=4,
                  rect=(0, 0, 1, 0.96),
                  plt_title="",
                  colorbar=True,
                  colorbar_right=0.81,
                  colorbar_ax=(0.81, 0.85, 0.015, 0.12),
                  path_save=None,
                  show=False,
                  figsize=(15, 30)):
    """
    Plot 2D array channels side-by-side

    :param channels:            (1, x, y, c)-shaped array-like          Where c is the number of channels
    :param cols:                int                                     Number of columns to organize the subplots into
    :param rect:                4-tuple of floats from 0 to 1           Edit last value if plt_title is clipping with plots
    :param plt_title:           str                                     Plot title
    :param colorbar:            bool                                    Whether or not to show the colorbar
    :param colorbar_right:      float                                   Controls space between plots and colorbar
    :param colorbar_ax:         4-tuple of floats from 0 to 1           Controls size and position of colorbar
    :param path_save:           str                                     Output location to save png to (if desired)
    :param show                 bool                                    Whether or not to display plot
    :param figsize              2-tuple of ints                         Size of the plot
    
    :return:                    None
    """

    num_channels = channels.shape[-1]
    fig = plt.figure(figsize=figsize)
    plt.suptitle(plt_title, fontsize=16)
    for i in range(num_channels):
        plt.subplot(num_channels/cols, cols, i+1)
        plt.title(i+1)
        plt.tight_layout(rect=rect)
        plt.imshow(channels[0, :, :, i])

    if colorbar:
        fig.subplots_adjust(right=colorbar_right)
        cbar_ax = fig.add_axes(colorbar_ax)
        plt.colorbar(cax=cbar_ax)
    if path_save is not None:
        plt.savefig(path_save, bbox_inches='tight')
    if show:
        plt.show()