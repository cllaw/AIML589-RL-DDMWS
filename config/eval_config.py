from abc import *
import argparse
import yaml
import os

# logging.basicConfig(level=logging.INFO)


class EvalConfig(metaclass=ABCMeta):

    def __init__(self, *args):
        fr, log_path, wf_size, gamma, distributed_cloud_enabled, data_scaling_factor, latency_penalty_factor, \
            region_mismatch_penalty_factor = args
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default=f"{log_path}/profile.yaml")
        parser.add_argument("--policy_path", type=str, default=f'{log_path}/saved_models/ep_{fr}.pt',
                            help='saved model directory')
        parser.add_argument("--rms_path", type=str, default=f'{log_path}/saved_models/ob_rms_{fr}.pickle',
                            help='saved run-time mean and std directory')
        parser.add_argument('--eval_ep_num', type=int, default=1, help='Set evaluation number per iteration')
        parser.add_argument('--save_model_freq', type=int, default=20, help='Save model every a few iterations')
        parser.add_argument('--processor_num', type=int, default=1, help='Testing model only use 1 processor')
        # parser.add_argument("--log", action="store_true", help="Use log")
        parser.add_argument("--log", default=True, action="store_true", help="Use log")  # Ya added
        parser.add_argument("--save_gif", action="store_true")
        parser.add_argument('--wf_size', '-w', type=str, default=wf_size)  # Ya added
        parser.add_argument('--gamma', '-g', type=float, default=gamma)  # Ya added

        # Chuan added for DDMWS
        parser.add_argument('--distributed_cloud_enabled', '-d', type=bool, default=distributed_cloud_enabled)
        parser.add_argument('--data_scaling_factor', '-dsf', type=float, default=0.5)
        parser.add_argument('--latency_penalty_factor', '-lpf', type=float, default=0.5)
        parser.add_argument('--region_mismatch_penalty_factor', '-rmpf', type=float, default=0.5)
        args = parser.parse_args()

        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            f.close()

            if args.wf_size is not None:  # Ya added
                config['env']['wf_size'] = args.wf_size

            if args.gamma is not None:  # Ya added
                config['env']['gamma'] = args.gamma

            if args.distributed_cloud_enabled is not None:  # Chuan added
                config['env']['distributed_cloud_enabled'] = args.distributed_cloud_enabled

            if args.data_scaling_factor is not None:
                config['env']['data_scaling_factor'] = args.data_scaling_factor

            if args.latency_penalty_factor is not None:
                config['env']['latency_penalty_factor'] = args.latency_penalty_factor

            if args.region_mismatch_penalty_factor is not None:
                config['env']['region_mismatch_penalty_factor'] = args.region_mismatch_penalty_factor

        if args.save_gif:
            run_num = args.ckpt_path.split("/")[-3]
            save_dir = f"test_gif/{run_num}/"
            os.makedirs(save_dir)

        self.config = {}
        self.config["runtime-config"] = vars(args)
        self.config["yaml-config"] = config


