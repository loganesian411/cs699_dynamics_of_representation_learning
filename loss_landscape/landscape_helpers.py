import numpy
from utils.evaluations import get_loss_value
from utils.nn_manipulation import set_weights_by_direction

# TODO(loganesian): generalize to paralellize over the larger number of dimensions
def compute_direction_loss(model, pretrained_weights, train_loader,
                           direction1, direction2, device, skip_bn_bias,
                           x, y_vals):
    y_num = numpy.size(y_vals)
    this_y_losses = numpy.zeros((1, y_num))
    this_y_accuracies = numpy.zeros((1, y_num))
    for idx_y, y in enumerate(y_vals):
        set_weights_by_direction(
            model, x, y, direction1, direction2, pretrained_weights,
            skip_bn_bias=skip_bn_bias
        )
        this_y_losses[idx_y], this_y_accuracies[idx_y] = get_loss_value(
            model, train_loader, device
        )
        print(f"x:{x: .4f}, y:{y: .4f}, loss:{this_y_losses[idx_y]:.4f}")
    print(f"Done with x:{x: .4f}")
    return this_y_losses, this_y_accuracies