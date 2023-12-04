import matplotlib

matplotlib.use('Agg')
# noinspection PyPep8
import matplotlib.pyplot as plt
import numpy as np


def create_scatterplot(df, output_file):
    xs = df['loes-score']
    ys = df['prediction']
    groups = df.groupby('subject')
    fig, ax = plt.subplots()
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group['loes-score'], group['prediction'], marker='o', linestyle='', ms=12, label=name)
    ax.legend()

    plt.title('Actual Loes score vs. predicted Loes score')
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against each other
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.axvline(x=10, color='b', ls='--')
    plt.axhline(y=10, color='b', ls='--')
    plt.xlabel("Actual Loes score")
    plt.ylabel("Predicted Loes score")

    plt.savefig(output_file)
