###
#
# This file manages the bridges module.
#
###
from typing import Literal, Tuple, Union, TYPE_CHECKING

from torchvision import transforms
import torch

# We use dynamic imports to avoid not used requirements.
# For this to work, used types must be "forward declared" in quotes
# (see https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING)
# Then, if bridge is selected, we can import the required module
if TYPE_CHECKING:
    from .semantic import ClassicSemanticSegmentation, ClassicSemanticSegmentationConfig
    from .open_clip_semantic import (
        OpenClipSemanticSegmentation,
        OpenClipSemanticSegmentationConfig,
    )
    from .open_trident_semantic import (
        OpenTridentSemanticSegmentation,
        OpenTridentSemanticSegmentationConfig,
    )

InferenceConfig = Union[
    "ClassicSemanticSegmentationConfig",
    "OpenClipSemanticSegmentationConfig",
    "OpenTridentSemanticSegmentationConfig",
]

InferenceType = Literal["classic", "open-sim", "open-seg"]

Inference = Union[
    "ClassicSemanticSegmentation",
    "OpenClipSemanticSegmentation",
    "OpenTridentSemanticSegmentation",
]


def get_inference_config(inference_type: str):
    if inference_type == "classic":
        from .semantic import ClassicSemanticSegmentationConfig

        return ClassicSemanticSegmentationConfig

    elif inference_type == "open-sim":
        from .open_clip_semantic import OpenClipSemanticSegmentationConfig

        return OpenClipSemanticSegmentationConfig
    elif inference_type == "open-seg":
        from .open_trident_semantic import OpenTridentSemanticSegmentationConfig

        return OpenTridentSemanticSegmentationConfig
    else:
        raise NotImplementedError(f"Inference type {inference_type} not implemented")


def get_inference(inference_cfg: InferenceConfig) -> Inference:
    """
    Get the inference type which determines the get_prediction function
    """
    if inference_cfg.inference_type == "classic":
        from .semantic import (
            ClassicSemanticSegmentation,
            ClassicSemanticSegmentationConfig,
        )

        assert isinstance(inference_cfg, ClassicSemanticSegmentationConfig), (
            "Inference cfg must be of type ClassicSemanticSegmentationConfig"
        )
        return ClassicSemanticSegmentation(inference_cfg)

    elif inference_cfg.inference_type == "open-sim":
        from .open_clip_semantic import (
            OpenClipSemanticSegmentation,
            OpenClipSemanticSegmentationConfig,
        )

        assert isinstance(inference_cfg, OpenClipSemanticSegmentationConfig), (
            "Inference cfg must be of type OpenClipSemanticSegmentationConfig"
        )
        return OpenClipSemanticSegmentation(inference_cfg)

    elif inference_cfg.inference_type == "open-seg":
        from .open_trident_semantic import (
            OpenTridentSemanticSegmentation,
            OpenTridentSemanticSegmentationConfig,
        )

        assert isinstance(inference_cfg, OpenTridentSemanticSegmentationConfig), (
            "Inference cfg must be of type OpenTridentSemanticSegmentationConfig"
        )
        return OpenTridentSemanticSegmentation(inference_cfg)
    else:
        raise NotImplementedError(
            f"Inference type: {inference_cfg.inference_type} not implemented"
        )


def get_model(
    inference_cfg: InferenceConfig, device: torch.device, pretrained: bool = False
) -> Tuple[torch.nn.Module, transforms.Compose]:
    """
    Get the DL model to be used. Load it from different libraries / repositories
    Distinguish between inference types. Returns model and preprocess in device.
    """
    model_name = inference_cfg.model_name
    inference_type = inference_cfg.inference_type

    if model_name == "deeplabv3":
        if inference_type == "classic":
            # TODO: Add support for different encoder_name
            from torchvision.models.segmentation.deeplabv3 import (
                deeplabv3_resnet50,
                DeepLabV3_ResNet50_Weights,
            )

            weights = None
            if pretrained:
                if inference_cfg.weights_path is not None:
                    weights = torch.load(
                        str(inference_cfg.weights_path), weights_only=True
                    )
                    print("Using loaded weights")
                else:
                    weights = DeepLabV3_ResNet50_Weights.DEFAULT

            model = deeplabv3_resnet50(
                num_classes=inference_cfg.num_classes, weights=weights
            )
        else:
            raise NotImplementedError("Inference type not implemented")

        model.to(device)
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (inference_cfg.height, inference_cfg.width), antialias=True
                ),  # type: ignore
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return model, preprocess

    elif model_name == "deeplabv3_resnet101":
        if inference_type == "classic":
            from torchvision.models.segmentation.deeplabv3 import (
                deeplabv3_resnet101,
                DeepLabV3_ResNet101_Weights,
            )

            weights = None

            if pretrained:
                if inference_cfg.weights_path is not None:
                    weights = torch.load(
                        str(inference_cfg.weights_path), weights_only=True
                    )
                else:
                    weights = DeepLabV3_ResNet101_Weights.DEFAULT

            model = deeplabv3_resnet101(
                num_classes=inference_cfg.num_classes, weights=weights
            )
        else:
            raise NotImplementedError("Inference type not implemented")

        model.to(device)
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (inference_cfg.height, inference_cfg.width), antialias=True
                ),  # type: ignore
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return model, preprocess

    elif model_name == "clip":
        from sensors_tools.inference.models.clip import clip

        clip_model_name = inference_cfg.encoder_name
        return clip.load(clip_model_name, device=device)  # Returns model and preprocess

    elif model_name == "dino":
        BACKBONE_SIZE = "small"  # in ("small", "base", "large" or "giant")

        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_arch = backbone_archs[BACKBONE_SIZE]
        backbone_name = f"dinov2_{backbone_arch}"

        backbone_model = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2", model=backbone_name
        )  # type: ignore
        backbone_model.eval()
        backbone_model.to(device)

    elif model_name == "trident":
        from sensors_tools.inference.models.trident.trident import Trident

        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        class_names = inference_cfg.class_names.split(",")
        print(class_names)
        model = Trident(
            clip_type="openai",
            clip_model_type="ViT-B/16",
            vfm_model="dino",
            class_names=class_names,
            device=device,
            sam_refinement=True,
            coarse_thresh=inference_cfg.coarse_threshold,
            minimal_area=225,
            debug=False,
            sam_ckpt=inference_cfg.sam_checkpoint_path,
            sam_model_type=inference_cfg.sam_model_type,
        )

        return model, preprocess
    else:
        raise NotImplementedError(f"Model: {model_name} not implemented")
