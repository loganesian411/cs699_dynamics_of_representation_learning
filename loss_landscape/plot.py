"""Plot the contours and trajectory give the corresponding files"""

import argparse
import logging
import os

import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action='store_true')
    parser.add_argument("--result_folder", "-r", required=True)
    parser.add_argument("--trajectory_file", required=False, default=None)
    parser.add_argument("--surface_file", required=False, default=None)
    parser.add_argument("--loss_accuracy_file", required=False, default=None)
    parser.add_argument("--contour_levels", type=str, required=False,
                        default="8:0.42:0.49")
    parser.add_argument("--num_contour_levels", type=int, required=False,
                        default=0)
    parser.add_argument("--plot_prefix", required=True, help="prefix for the figure names")

    args = parser.parse_args()

    # set up logging
    os.makedirs(f"{args.result_folder}", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.surface_file:
        # create a contour plot
        data = numpy.load(f"{args.surface_file}")

        xcoords = data["xcoordinates"]
        ycoords = data["ycoordinates"]
        losses = data["losses"]
        acc = data["accuracies"]

        X, Y = numpy.meshgrid(xcoords, ycoords, indexing="ij")
        Z = losses

        if args.num_contour_levels:
            contour_levels = args.num_contour_levels
        else:
            x_num, x_min, x_max = [float(i) for i in args.contour_levels.split(":")]
            contour_levels = numpy.linspace(x_min, x_max, int(x_num))

        fig = pyplot.figure()
        CS = pyplot.contour(X, Y, Z, cmap='summer', levels=contour_levels)
        pyplot.clabel(CS, inline=1, fontsize=8)
        fig.savefig(
            f"{args.result_folder}/{args.plot_prefix}_surface_2d_contour", dpi=300,
            bbox_inches='tight'
        )
        pyplot.close()

    if args.trajectory_file:
        # create a 2D plot of trajectory
        data = numpy.load(f"{args.trajectory_file}")

        xcoords = data["xcoordinates"]
        ycoords = data["ycoordinates"]

        fig = pyplot.figure()
        pyplot.plot(xcoords, ycoords, linewidth=0.5, alpha=0.3)
        pyplot.scatter(xcoords, ycoords, marker='.', c=numpy.arange(len(xcoords)), cmap='Reds')
        pyplot.colorbar()
        pyplot.tick_params('y', labelsize='x-large')
        pyplot.tick_params('x', labelsize='x-large')

        fig.savefig(
            f"{args.result_folder}/{args.plot_prefix}_trajectory_2d", dpi=300,
            bbox_inches='tight'
        )
        pyplot.close()

    if args.surface_file and args.trajectory_file:
        # create a contour plot
        data = numpy.load(f"{args.surface_file}")

        xcoords = data["xcoordinates"]
        ycoords = data["ycoordinates"]
        losses = data["losses"]
        acc = data["accuracies"]

        X, Y = numpy.meshgrid(xcoords, ycoords, indexing="ij")
        Z = losses
        fig = pyplot.figure()
        CS = pyplot.contour(X, Y, Z, cmap='summer', levels=contour_levels)
        pyplot.clabel(CS, inline=1, fontsize=8)

        data = numpy.load(f"{args.trajectory_file}")

        xcoords = data["xcoordinates"]
        ycoords = data["ycoordinates"]
        pyplot.plot(xcoords, ycoords, linewidth=0.5, alpha=0.3)
        pyplot.colorbar()
        pyplot.scatter(xcoords, ycoords, marker='.', c=numpy.arange(len(xcoords)), cmap='Reds')
        pyplot.tick_params('y', labelsize='x-large')
        pyplot.tick_params('x', labelsize='x-large')

        fig.savefig(
            f"{args.result_folder}/{args.plot_prefix}_trajectory+contour_2d", dpi=300,
            bbox_inches="tight"
        )
        pyplot.close()

    if args.loss_accuracy_file:
        data = numpy.load(f"{args.loss_accuracy_file}")

        fig, ax = pyplot.subplots(1, 2, figsize=(15,7))
        xvals = 1 + numpy.arange(len(data["train_losses"]))
        ax[0].plot(xvals, data["train_losses"], "g.--", label="Training")
        ax[0].plot(xvals, data["test_losses"], "r.--", label="Test")
        ax[0].set_ylabel("Loss")
        ax[0].set_xlabel("Epoch")
        ax[0].legend()

        ax[1].plot(xvals, data["train_accuracies"], "g.--", label="Training")
        ax[1].plot(xvals, data["test_accuracies"], "r.--", label="Test")
        ax[1].set_ylabel("Accuracy")
        ax[1].set_xlabel("Epoch")
        ax[1].legend()

        fig.savefig(
            f"{args.result_folder}/{args.plot_prefix}_accuracy_loss_vs_epoch",
            dpi=300, bbox_inches=None #"tight"
        )
        pyplot.close()
