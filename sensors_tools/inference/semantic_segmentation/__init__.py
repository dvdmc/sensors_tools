###
#
# This file manages the inference module.
#
###
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
  from .semantic_segmentation_deep_lab_v3 import SemanticSegmentationDeepLabV3, SemanticSegmentationDeepLabV3Config
  from .semantic_segmentation_segformer import SemanticSegmentationSegformer, SemanticSegmentationSegformerConfig
  from .semantic_segmentation_clip import SemanticSegmentationCLIP, SemanticSegmentationCLIPConfig
  from .semantic_segmentation_trident import SemanticSegmentationTrident, SemanticSegmentationTridentConfig

InferenceConfig = Union[
    "SemanticSegmentationDeepLabV3Config",
    "SemanticSegmentationSegformerConfig",
    "SemanticSegmentationCLIPConfig",
    "SemanticSegmentationTridentConfig",
]

Inference = Union[
    "SemanticSegmentationDeepLabV3",
    "SemanticSegmentationSegformer",
    "SemanticSegmentationCLIP",
    "SemanticSegmentationTrident",
]

def get_semantic_segmentation_config(inference_type: str):
    if inference_type == "deep_lab_v3":
        from .semantic_segmentation_deep_lab_v3 import SemanticSegmentationDeepLabV3Config

        return SemanticSegmentationDeepLabV3Config
    elif inference_type == "segformer":
        from .semantic_segmentation_segformer import SemanticSegmentationSegformerConfig

        return SemanticSegmentationSegformerConfig
    elif inference_type == "clip":
        from .semantic_segmentation_clip import SemanticSegmentationCLIPConfig

        return SemanticSegmentationCLIPConfig
    elif inference_type == "trident":
        from .semantic_segmentation_trident import SemanticSegmentationTridentConfig

        return SemanticSegmentationTridentConfig
    
    else:
        raise NotImplementedError(f"Inference type {inference_type} not implemented")
    
def get_semantic_segmentation(inference_cfg: InferenceConfig) -> Inference:
    """
     Get the inference module
    """
    if inference_cfg.semantic_segmentation_method == "deep_lab_v3":
        from .semantic_segmentation_deep_lab_v3 import SemanticSegmentationDeepLabV3

        return SemanticSegmentationDeepLabV3(inference_cfg)
    elif inference_cfg.semantic_segmentation_method == "segformer":
        from .semantic_segmentation_segformer import SemanticSegmentationSegformer

        return SemanticSegmentationSegformer(inference_cfg)
    elif inference_cfg.semantic_segmentation_method == "clip":
        from .semantic_segmentation_clip import SemanticSegmentationCLIP

        return SemanticSegmentationCLIP(inference_cfg)
    elif inference_cfg.semantic_segmentation_method == "trident":
        from .semantic_segmentation_trident import SemanticSegmentationTrident

        return SemanticSegmentationTrident(inference_cfg)
    else:
        raise NotImplementedError(f"Inference type {inference_cfg.inference_type} not implemented")