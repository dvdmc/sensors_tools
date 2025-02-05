###
#
# This file manages the bridges module.
#
###
from dataclasses import fields
from typing import Literal, Tuple, Union, TYPE_CHECKING

from torchvision import transforms
import torch

# We use dynamic imports to avoid not used requirements.
# For this to work, used types must be "forward declared" in quotes
# (see https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING)
# Then, if bridge is selected, we can import the required module
if TYPE_CHECKING:
    from .semantic import SemanticInference, SemanticInferenceConfig
    from .semantic_mcd import SemanticMCDInference, SemanticMCDInferenceConfig
    from .open_clip_semantic import OpenClipSemanticInference, OpenClipSemanticInferenceConfig
    from .open_trident_semantic import OpenTridentSemanticInference, OpenTridentSemanticInferenceConfig

InferenceConfig = Union["SemanticInferenceConfig", "SemanticMCDInferenceConfig", "OpenClipSemanticInferenceConfig", "OpenTridentSemanticInferenceConfig"]

InferenceType = Literal["deterministic", "mcd", "open-sim", "open-seg"]

Inference = Union["SemanticInference", "SemanticMCDInference", "OpenClipSemanticInference", "OpenTridentSemanticInference"]


def get_inference_config(model_name: str):

    if "deterministic" in model_name:
        from .semantic import SemanticInferenceConfig

        return SemanticInferenceConfig
    elif "mcd" in model_name:
        from .semantic_mcd import SemanticMCDInferenceConfig

        return SemanticMCDInferenceConfig
    elif "open-sim" in model_name:
        from .open_clip_semantic import OpenClipSemanticInferenceConfig

        return OpenClipSemanticInferenceConfig
    elif "open-seg" in model_name:
        from .open_trident_semantic import OpenTridentSemanticInferenceConfig

        return OpenTridentSemanticInferenceConfig
    else:
        raise NotImplementedError(f"Inference type {model_name} not implemented")


def get_inference(inference_cfg: InferenceConfig) -> Inference:
    """
    Get the inference type which determines the get_prediction function
    """
    if "deterministic" in inference_cfg.model_name:
        from .semantic import SemanticInference, SemanticInferenceConfig

        assert isinstance(
            inference_cfg, SemanticInferenceConfig
        ), "Inference cfg must be of type SemanticInferenceConfig"
        return SemanticInference(inference_cfg)
    elif "mcd" in inference_cfg.model_name:
        from .semantic_mcd import SemanticMCDInference, SemanticMCDInferenceConfig

        assert isinstance(
            inference_cfg, SemanticMCDInferenceConfig
        ), "Inference cfg must be of type SemanticMCDInferenceConfig"
        return SemanticMCDInference(inference_cfg)
    elif "open-sim" in inference_cfg.model_name:
        from .open_clip_semantic import OpenClipSemanticInference, OpenClipSemanticInferenceConfig

        assert isinstance(
            inference_cfg, OpenClipSemanticInferenceConfig
        ), "Inference cfg must be of type OpenClipSemanticInferenceConfig"
        return OpenClipSemanticInference(inference_cfg)
    elif "open-seg" in inference_cfg.model_name:
        from .open_trident_semantic import OpenTridentSemanticInference, OpenTridentSemanticInferenceConfig

        assert isinstance(
            inference_cfg, OpenTridentSemanticInferenceConfig
        ), "Inference cfg must be of type OpenTridentSemanticInferenceConfig"
        return OpenTridentSemanticInference(inference_cfg)
    else:
        raise NotImplementedError(f"Inference type: {inference_cfg.model_name} not implemented")


def get_model(
    inference_cfg: InferenceConfig, device: torch.device, pretrained: bool = False
) -> Tuple[torch.nn.Module, transforms.Compose]:
    """
    Get the DL model to be used. Load it from different libraries / repositories
    Distinguish between inference types. Returns model and preprocess in device.
    """
    model_name = inference_cfg.model_name

    if "deeplabv3_resnet50" in model_name:
        if "deterministic" in model_name:

            from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

            if pretrained:
                if inference_cfg.weights_path is not None:
                    from .models.deeplabv3.deeplab_classifier import deeplabv3_resnet50_MCD

                    weights = torch.load(str(inference_cfg.weights_path), weights_only=True)
                    print("Using loaded weights")
                    model = deeplabv3_resnet50_MCD(
                        num_classes=inference_cfg.num_classes
                    )  # We assume this at the moment for the active semantic paper
                    model.load_state_dict(weights, strict=True)
                else:
                    weights = DeepLabV3_ResNet50_Weights.DEFAULT
                    model = deeplabv3_resnet50(num_classes=inference_cfg.num_classes, weights=weights)
            else:
                model = deeplabv3_resnet50(num_classes=inference_cfg.num_classes, weights=None)
        elif "mcd" in model_name:

            from .models.deeplabv3.deeplab_classifier import deeplabv3_resnet50_MCD

            model = deeplabv3_resnet50_MCD(num_classes=inference_cfg.num_classes)
            if pretrained:
                weights = torch.load(str(inference_cfg.weights_path), weights_only=True)
                model.load_state_dict(weights, strict=True)

        else:
            raise NotImplementedError("Inference type not implemented")

        model.to(device)
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((inference_cfg.height, inference_cfg.width), antialias=True),  # type: ignore
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return model, preprocess

    elif "deeplabv3_resnet101" in model_name:
        if "deterministic" in model_name:
            from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

            if pretrained:
                if inference_cfg.weights_path is not None:
                    weights = torch.load(str(inference_cfg.weights_path), weights_only=True)
                    model = deeplabv3_resnet101(num_classes=inference_cfg.num_classes, weights=None)
                else:
                    weights = DeepLabV3_ResNet101_Weights.DEFAULT
            else:
                weights = None
            model = deeplabv3_resnet101(num_classes=inference_cfg.num_classes, weights=weights)
        elif "mcd" in model_name:
            raise NotImplementedError(f"Model not type not implemented for: {model_name}")
        else:
            raise NotImplementedError("Inference type not implemented")

        model.to(device)
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.cfg.height, self.cfg.width), antialias=True),  # type: ignore
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return model, preprocess

    elif "clip" in model_name:
        from sensors_tools.inference.models.clip import clip

        clip_model_name = model_name.split("_")[1]
        return clip.load(clip_model_name, device=device)  # Returns model and preprocess

    elif "dino" in model_name:

        BACKBONE_SIZE = "small"  # in ("small", "base", "large" or "giant")

        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_arch = backbone_archs[BACKBONE_SIZE]
        backbone_name = f"dinov2_{backbone_arch}"

        backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)  # type: ignore
        backbone_model.eval()
        backbone_model.to(device)

    elif "trident" in model_name:
        from sensors_tools.inference.models.trident.trident import Trident

        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
            ]
        )
        class_names = inference_cfg.class_names.split(',')
        print(class_names)
        model = Trident(clip_type='openai', clip_model_type='ViT-B/16', vfm_model='dino', class_names=class_names, device=device, sam_refinement=True,
                    coarse_thresh=inference_cfg.coarse_threshold, minimal_area=225, debug=False, sam_ckpt=inference_cfg.sam_checkpoint, sam_model_type=inference_cfg.sam_model_type)

        return model, preprocess
    else:
        raise NotImplementedError("Model not implemented")
