import sys
import os
import configs
import ZSSR_Cycle

def main(input_img=None, ground_truth='0', kernels='0', gpu=None, conf_str=None, results_path='./results'):
    # Choose the wanted GPU
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    # Handle the ground truth and kernels inputs
    ground_truth = None if ground_truth == '0' else ground_truth
    kernels = None if kernels == '0' else kernels.split(';')[:-1]

    # Setup configuration and results directory
    conf = configs.Config()
    if conf_str is not None:
        exec(f'conf = configs.{conf_str}')
    conf.result_path = results_path

    # Run ZSSR on the image
    net = ZSSR_Cycle.ZSSR(input_img, conf, ground_truth, kernels)
    net.run()

    # Clean up
    del net

if __name__ == '__main__':
    if len(sys.argv) < 7:
        print("Usage: python run_ZSSR_Cycle_single_input.py <input_img> <ground_truth> <kernels> <gpu> <conf_str> <results_path>")
        sys.exit(1)

    # Fill missing arguments with None or default values
    args = sys.argv[1:] + [None] * (6 - len(sys.argv[1:]))
    main(*args)
