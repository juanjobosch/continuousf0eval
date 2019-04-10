import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches


def get_dfbox(data_frame_list, metrics):

    boxdata = []
    n_groups = len(data_frame_list)
    for metric in metrics:
        for data_frame in data_frame_list:
            boxdata.extend([
                data_frame[metric],
            ])

    dfbox = pd.DataFrame(np.array(boxdata).T)
    return dfbox, n_groups


def add_plot(subplot_tuple, metrics, dfbox, n_algs, title,
             show_yaxis=True, legend_loc=None, xlim=None,
             legend_labels=None, colors=None):
    plt.subplot(subplot_tuple)
    plt.title(title, weight='bold')
    n_metrics = len(metrics)
    positions = []
    k = 1
    for _ in range(n_metrics):
        for _ in range(n_algs):
            positions.append(k)
            k = k + 1
        k = k + 1

    if colors is None:
        current_palette = ["#00B200", "#42CAFD", "#FFC145"]
    else:
        current_palette = [c for c in colors]

    colors = current_palette[:n_algs] * n_metrics

    box = plt.boxplot(
        dfbox.values, widths=0.8, positions=positions,
        patch_artist=True, showmeans=True,
        medianprops={'color': 'k'},
        meanprops=dict(marker='D', markeredgecolor='black',
                       markerfacecolor='k'),
        vert=False,
        showfliers=False
    )
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.xlabel('Score')
    yticks = np.arange(
        (n_algs + 1) / 2, ((n_algs + 1) * n_metrics), n_algs + 1)
    if show_yaxis:
        plt.yticks(yticks, metrics, rotation='horizontal', weight='bold')
    else:
        plt.yticks(yticks, [''] * len(metrics), rotation='horizontal')

    if xlim is not None:
        plt.xlim(xlim)

    if legend_labels is not None:
        legend_colors = []
        for color in colors[:n_algs]:
            h_color = patches.Rectangle((0, 0), 1, 1, facecolor=color)
            legend_colors.append(h_color)

        plt.legend(legend_colors, legend_labels, ncol=1, loc=legend_loc)
