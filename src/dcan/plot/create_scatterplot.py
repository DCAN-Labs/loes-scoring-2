import ast
import sys

import matplotlib.pyplot as plt
import numpy as np


def main(input_file, output_file):
    with open(input_file) as f:
        data = f.read()

        d = ast.literal_eval(data)

        xs = []
        ys = []
        for x in d:
            vals = d[x]
            for y in vals:
                xs.append(x)
                ys.append(y)

        fig, ax = plt.subplots()
        plt.scatter(xs, ys)
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
        plt.xlabel("Actual Loes score")
        plt.ylabel("Predicted Loes score")

        plt.savefig(output_file)
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
