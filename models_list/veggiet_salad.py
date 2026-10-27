import sys

import torch

from models_list import Model


class VggtVprModel(torch.nn.Module):
    def __init__(self, salad_model: object) -> None:
        super().__init__()
        self.salad_model = salad_model

    def forward(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            patch_tokens = self.salad_model.backbone.dino_forward(images)
            feats, cls = self.salad_model.backbone.prepare_tokens_for_salad(patch_tokens, images.shape)
            global_descriptor = self.salad_model.aggregator((feats, cls))
        return global_descriptor


class VggtoVprModel(torch.nn.Module):
    def __init__(self, salad_model: object) -> None:
        super().__init__()
        self.salad_model = salad_model

    def forward(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            patch_tokens = self.salad_model.backbone.dino_forward(images)
            feats, cls = self.salad_model.backbone.prepare_tokens_for_salad(patch_tokens, images.shape)
        global_descriptor = self.salad_model.aggregator((feats, cls))
        return global_descriptor
    

class VeggietModel(Model):
    @staticmethod
    def get_model(**kwargs):
        model_name: str = kwargs.get('model', None)
        if model_name is None:
            raise RuntimeError("--model is required")

        if model_name.startswith('vggto'):
            model_wrapper_type = VggtoVprModel
            vggto_ckpt = kwargs.get('ckpt', None)
            if vggto_ckpt is None:
                raise RuntimeError("--ckpt is required for vggt-omega")
            model_kwargs = {
                'vggt_omega_ckpt': vggto_ckpt
            }
        elif model_name.startswith('vggt'):
            model_wrapper_type = VggtVprModel
            model_kwargs = {}
        else:
            RuntimeError(f"{model_name} not supported")

        repo_path = kwargs.get('repo_path', None)
        if repo_path is None:
            raise RuntimeError("--repo_path must be passed")

        print(model_name, repo_path)
        model_flavor = torch.hub.load(
            'L4rralde/Visual_Place_Recognition',
            model_name,
            repo_path,
            **model_kwargs
        )

        return model_wrapper_type(model_flavor), model_name, 8448

    @staticmethod
    def get_transform(**kwargs):
        repo_path = kwargs.get('repo_path', None)
        if repo_path is None:
            raise RuntimeError("--repo_path must be provided")

        model_name: str = kwargs.get('model', None)
        if model_name is None:
            raise RuntimeError("--model is required")

        if model_name.startswith('vggto'):
            img_size = 512
            backbone = 'VGGT_OMEGA'
        elif model_name.startswith('vggt'):
            img_size = 518
            backbone = 'VGGT'
        else:
            RuntimeError(f"{model_name} not supported")

        sys.path.append(repo_path)
        from vpr.models.helper import get_transforms
        config = {
            'img_size': img_size,
        }
        _, val_transform = get_transforms(
            backbone,
            config
        )
        return val_transform
