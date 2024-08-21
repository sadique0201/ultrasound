import os

class Config:
    # Network meta params
    # python_path = '/home/yiqunm2/tensorflow/bin/python'
    scale_factors = [[4.0, 4.0]]  # List of pairs (vertical, horizontal) for gradual increments in resolution
    base_change_sfs = []  # List of scales after which the input is changed to be the output (recommended for high sfs)
    max_iters = 3000
    min_iters = 256
    min_learning_rate = 9e-6  # This tells the algorithm when to stop (specify lower than the last learning-rate)
    width = 64
    depth = 8
    output_flip = True  # Geometric self-ensemble (see paper)
    downscale_method = 'cubic'  # Interpolation method ('cubic', 'linear'...), has no meaning if kernel given
    upscale_method = 'cubic'  # Base interpolation from which we learn the residual (same options as above)
    downscale_gt_method = 'cubic'  # Method to shrink ground-truth to wanted size when intermediate scales tested
    learn_residual = True  # If true, only learn the residual from base interpolation
    init_variance = 0.1  # Variance of weight initializations, typically smaller when residual learning is on
    back_projection_iters = [10]  # For each scale, number of BP iterations (same length as scale_factors)
    random_crop = True
    crop_size = 64
    noise_std = 0.0  # Adding noise to lr-sons. Small for real images, bigger for noisy images and zero for ideal case
    init_net_for_each_sf = False  # For gradual SR - should we optimize from the last SF or initialize each time?
    cuda = True

    # Params concerning learning rate policy
    learning_rate = 0.001
    learning_rate_change_ratio = 1.5  # Ratio between STD and slope of linear fit, under which lr is reduced
    learning_rate_policy_check_every = 60
    learning_rate_slope_range = 256

    # Data augmentation related params
    augment_leave_as_is_probability = 0.05
    augment_no_interpolate_probability = 0.45
    augment_min_scale = 0.6
    augment_scale_diff_sigma = 0.25
    augment_shear_sigma = 0.1
    augment_allow_rotation = True  # Recommended false for non-symmetric kernels

    # Params related to test and display
    run_test = True
    run_test_every = 60
    display_every = 50
    name = 'test'
    plot_losses = False
    result_path = os.path.join(os.path.dirname(__file__), 'results')
    create_results_dir = True
    input_path = os.path.join(os.path.dirname(__file__), 'test_data')  # Updated path format
    create_code_copy = True  # Save a copy of the code in the results folder to easily match code changes to results
    display_test_results = True
    save_results = True

    def __init__(self):
        # Network meta params that by default are determined (by other params) by other params but can be changed
        self.filter_shape = ([[3, 3, 3, self.width]] +
                             [[3, 3, self.width, self.width]] * (self.depth-2) +
                             [[3, 3, self.width, 3]])
