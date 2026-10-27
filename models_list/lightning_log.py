import sys

from models_list import Model



class LightningLogModel(Model):
    @staticmethod
    def get_model(**kwargs):
        repo_path = kwargs.get('repo_path', None)
        if repo_path is None:
            raise RuntimeError("--repo_path must be provided")
        log = kwargs.get('log', None)
        if log is None:
            raise RuntimeError("--log must be provided")
        sys.path.append(repo_path)

        from vpr_model import VPRModel

        model = VPRModel.from_lightning_log(log)
        return model, model.encoder_arch, 8448
    
    @staticmethod
    def get_transform(**kwargs):
        repo_path = kwargs.get('repo_path', None)
        if repo_path is None:
            raise RuntimeError("--repo_path must be provided")

        log = kwargs.get('log', None)
        if log is None:
            raise RuntimeError("--log must be provided")
        sys.path.append(repo_path)

        from vpr.models.helper import get_transforms
        from utils.lightning_logs import LightningLog

        img_size = kwargs.get('img_size', None)
        if img_size is None:
            raise RuntimeError("--img_size is required")

        lightning_log = LightningLog(log)
        config = {
            'img_size': int(img_size),
        }
        _, val_transform = get_transforms(
            lightning_log.backbone_arch,
            config
        )
        return val_transform
