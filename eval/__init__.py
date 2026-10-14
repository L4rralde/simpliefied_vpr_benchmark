
"""
With this script you can evaluate checkpoints or test models from two popular
landmark retrieval github repos.

Note that although the architectures are almost the same, Naver's
implementation does not use a l2 normalization before/after the GeM aggregation,
while Radenovic's uses it after (and we use it before, which shows better
results in VG)
"""

import sys
import os
from os.path import join
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Tuple, Callable

import torch
import torchvision.transforms as tfm
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import parser
import test
import commons
from mydataset import BaseDataset


class Evaluator:
    def __init__(self):
        self.args = self.get_parser().parse_args()
        self.model, self.model_name, self.features_dim = self.define_model()
        self.input_transform = self.define_input_transform()

    def get_parser(self):
        return parser.get_parser()

    @abstractmethod    
    def define_model(self) -> Tuple[torch.nn.Module, str, int]:
        raise NotImplementedError()

    @abstractmethod
    def define_input_transform(self) -> Callable:
        raise NotImplementedError()

    def run(self, test_datasets=["baidu","amstertime","pitts250k", "eynsham",  "nordland", "st_lucia", "pitts30k"]):
        start_time = datetime.now()
        self.args.save_dir = join(
            "test",
            self.args.save_dir,
            self.model_name, 
            start_time.strftime('%Y-%m-%d_%H-%M-%S')
        )
        commons.setup_logging(self.args.save_dir)
        commons.make_deterministic(self.args.seed)
        logging.info(f"Arguments: {self.args}")
        logging.info(f"The outputs are being saved in {self.args.save_dir}")

        self.model = self.model.eval().to(self.args.device)
        df = pd.DataFrame({"model": [self.model_name]})
        df_path = f"{self.args.save_dir}.csv"

        test_datasets_ds = [
            BaseDataset(
                self.args,
                self.args.datasets_folder,
                ds_name,
                "test",
                transform=self.input_transform
            )
            for ds_name in test_datasets
        ]
        for test_ds in test_datasets_ds:
            logging.info(f"Test set: {test_ds}")
            recalls, recalls_str = test.test(self.args, test_ds, self.model, self.features_dim)
            r1, r5, r10, r20 = recalls
            recalls_dict = {
                f"{test_ds.dataset_name}_{r_str}": [r]
                for r_str, r in zip(['r1', 'r5', 'r10'], [r1, r5, r10])    
            }
            recalls_dict['model'] = self.model_name
            recalls_df = pd.DataFrame(recalls_dict)
            df = pd.merge(df, recalls_df, on="model", how="left")
            logging.info("-"*80)
            logging.info(f"Recalls on {test_ds.dataset_name}: {recalls_str}")
            logging.info("-"*80)
            df.to_csv(df_path, index=False)
        
    
class EvalMegaloc(Evaluator):
    def define_model(self):
        model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
        model_name = "MegaLoc"
        features_dim = 8448
        return model, model_name, features_dim

    def define_input_transform(self):
        transform = tfm.Compose([
            tfm.ToTensor(),
            tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tfm.Resize(size=[322, 322], antialias=True),
        ])

        return transform
