"""Basic script to run LSD + EBM."""
import data.toy_data as toy_data
import utils.density as density_utils
import utils.lsd_wrapper as lsd_wrapper

if __name__ == '__main__':
	data = toy_data.generate_density("checkerboard")
	crop = (250, 450, 300, 500)
	# crop = (350, 450, 350, 450)
	density, energy = density_utils.prepare_image(data, crop=crop, white_cutoff=225,
																								gauss_sigma=3, background=0.01)

	lsd_wrapper.lsd(density, energy, "./results/lsd_test",
									eps=0.3, K=5, density_initialization='gaussian')
