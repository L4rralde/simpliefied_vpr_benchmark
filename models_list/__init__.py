from abc import ABC, abstractmethod
from typing import Callable
import sys
from collections import OrderedDict

import torch
import torchvision.transforms as T
import torchvision.transforms.v2 as T2
from torchvision.transforms import functional as F


class Model:
    @staticmethod
    @abstractmethod
    def get_model(**kwargs) -> torch.nn.Module:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_transform(**kwargs) -> Callable:
        raise NotImplementedError()

    DESC_DIM = 0



class MegaLoc(Model):
    @staticmethod
    def get_model(**kwargs):
        return torch.hub.load("gmberton/MegaLoc", "get_trained_model")

    @staticmethod
    def get_transform(**kwargs):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Resize(size=[322, 322], antialias=True),
        ])
        return transform

    DESC_DIM = 8448


class DinoSalad(Model):
    @staticmethod
    def get_model(**kwargs):
        repo_path = kwargs.get('repo_path', None)
        if repo_path is None:
            raise RuntimeError("--repo_path is required")
        sys.path.append(repo_path)
        return torch.hub.load("serizba/salad", "dinov2_salad")

    @staticmethod
    def get_transform(**kwargs):
        MEAN=[0.485, 0.456, 0.406]
        STD=[0.229, 0.224, 0.225]
        return T.Compose([
            T.Resize([322, 322],  interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])

    DESC_DIM = 8448


class BoQ(Model):
    @staticmethod
    def get_model(**kwargs):
        class _BoQ(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.vpr_model = torch.hub.load(
                    "amaralibey/bag-of-queries",
                    "get_trained_boq",
                    backbone_name="dinov2",
                    output_dim=12288
                )
            def forward(self, x):
                return self.vpr_model(x)[0]

        return _BoQ()

    @staticmethod
    def get_transform(**kwargs):
        img_size = (322, 322) # to be used with DinoV2 backbone

        input_transform = T2.Compose([
            T2.ToTensor(),
            T2.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return input_transform

    DESC_DIM = 12288


class MixVPR(Model):
    @staticmethod
    def get_model(**kwargs):
        repo_path = kwargs.get('repo_path', None)
        if repo_path is None:
            raise RuntimeError("repo_path must be provided")
        ckpt = kwargs.get('ckpt', None)
        if ckpt is None:
            raise RuntimeError("ckpt must be provided")

        sys.path.append(repo_path)
        from main import VPRModel

        model = VPRModel(backbone_arch='resnet50',
                    layers_to_crop=[4],
                    agg_arch='MixVPR',
                    agg_config={'in_channels': 1024,
                                'in_h': 20,
                                'in_w': 20,
                                'out_channels': 1024,
                                'mix_depth': 4,
                                'mlp_ratio': 1,
                                'out_rows': 4},
                    )

        state_dict = torch.load(ckpt)
        model.load_state_dict(state_dict)

        model.eval()
        print(f"Loaded model from {ckpt} Successfully!")
        return model

    @staticmethod
    def get_transform(**kwargs):
        input_transform = T.Compose([
            T.Resize((320, 320), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
        ])

        return input_transform

    DESC_DIM = 4096


class EDTformer(Model):
    @staticmethod
    def get_model(**kwargs):
        repo_path = kwargs.get('repo_path', None)
        if repo_path is None:
            raise RuntimeError("repo_path must be provided")
        ckpt = kwargs.get('ckpt', None)
        if ckpt is None:
            raise RuntimeError("ckpt must be provided")
        sys.path.append(repo_path)
        from network import VPRNet

        model = VPRNet()
        
        checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
            # the checkpoint is directly the state dict
            state_dict = checkpoint
        # if the model contains the prefix "module" which is appendend by
        # DataParallel, remove it to avoid errors when loading dict
        if list(state_dict.keys())[0].startswith('module'):
            state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
        model.load_state_dict(state_dict)
        model = torch.nn.DataParallel(model)
        return model

    @staticmethod
    def get_transform(**kwargs):
        base_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        def input_transform(img):
            img = base_transform(img)
            img = F.resize(img, [322, 322])
            return img

        return input_transform

    DESC_DIM = 4096


class ImAge(Model):
    @staticmethod
    def get_model(**kwargs):
        repo_path = kwargs.get('repo_path', None)
        if repo_path is None:
            raise RuntimeError("repo_path must be provided")
        ckpt = kwargs.get('ckpt', None)
        if ckpt is None:
            raise RuntimeError("ckpt must be provided")
        sys.path.append(repo_path)
        from network import VPRmodel

        class ImageArgs:
            backbone="dinov2"
            aggregator=None
            freeze_te=8
            num_learnable_aggregation_tokens=8
            resume=ckpt
            features_dim=6144 #8*768


        model = VPRmodel(ImageArgs)

        checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
            # the checkpoint is directly the state dict
            state_dict = checkpoint
        # if the model contains the prefix "module" which is appendend by
        # DataParallel, remove it to avoid errors when loading dict
        if list(state_dict.keys())[0].startswith('module'):
            state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
        model.load_state_dict(state_dict)
        model = torch.nn.DataParallel(model)
        return model

    @staticmethod
    def get_transform(**kwargs):
        base_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        def input_transform(img):
            img = base_transform(img)
            img = F.resize(img, [322, 322])
            return img

        return input_transform


    DESC_DIM = 6144


ALL_MODELS = [MegaLoc, BoQ,  DinoSalad, MixVPR, EDTformer, ImAge]

