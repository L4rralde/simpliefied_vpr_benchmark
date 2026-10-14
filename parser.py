
import os
import torch
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datasets_folder", type=str, default='datasets', help="Path with all datasets")
    parser.add_argument("--save_dir", type=str, default="default",
                            help="Folder name of the current run (saved in ./logs/)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--infer_batch_size", type=int, default=32,
                            help="Batch size for inference (caching and testing)")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="_")

    return parser