###
#
# This file manages the bridges module.
#
###
from dataclasses import fields
from typing import Literal, Union, TYPE_CHECKING

import torch

# We use dynamic imports to avoid not used requirements.
# For this to work, used types must be "forward declared" in quotes 
# (see https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING)
# Then, if bridge is selected, we can import the required module
if TYPE_CHECKING:
    from .semantic import SemanticInference, SemanticInferenceConfig
    from .semantic_mcd import SemanticMCDInference, SemanticMCDInferenceConfig

InferenceConfig = Union['SemanticInferenceConfig', 'SemanticMCDInferenceConfig']

InferenceType = Literal["deterministic", "mcd"]

Inference = Union['SemanticInference', 'SemanticMCDInference']

def get_inference_config(inference_type: InferenceType):
    assert inference_type is not None, "Inference type must be specified"

    if inference_type == "deterministic":
        from .semantic import SemanticInferenceConfig
        return SemanticInferenceConfig
    elif inference_type == "mcd":
        from .semantic_mcd import SemanticMCDInferenceConfig
        return SemanticMCDInferenceConfig
    else:
        raise NotImplementedError("Inference type not implemented")

def get_inference(inference_type: InferenceType, inference_cfg: InferenceConfig) -> Inference:
    """
        Get the inference type which determines the get_prediction function
    """
    if inference_type == "deterministic":
        from .semantic import SemanticInference, SemanticInferenceConfig
        assert isinstance(inference_cfg, SemanticInferenceConfig), "Inference cfg must be of type SemanticInferenceConfig"
        return SemanticInference(inference_cfg)
    elif inference_type == "mcd":
        from .semantic_mcd import SemanticMCDInference, SemanticMCDInferenceConfig
        assert isinstance(inference_cfg, SemanticMCDInferenceConfig), "Inference cfg must be of type SemanticMCDInferenceConfig"
        return SemanticMCDInference(inference_cfg)
    else:
        raise NotImplementedError("Inference type not implemented")
    
def get_model(inference_type: InferenceType, inference_cfg: InferenceConfig, pretrained: bool = False) -> torch.nn.Module:
    """
        Get the DL model to be used. Load it from different libraries / repositories
        Distinguish between inference types
    """
    model_name = inference_cfg.model_name

    if model_name == "deeplabv3_resnet50":
        if inference_type == "deterministic":
        
            from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
            if pretrained:
                if inference_cfg.weights_path is not None:
                    from .models.deeplabv3.deeplab_classifier import deeplabv3_resnet50_MCD
                    weights = torch.load(str(inference_cfg.weights_path), weights_only=True)
                    model = deeplabv3_resnet50_MCD(num_classes=inference_cfg.num_classes) # We assume this at the moment for the active semantic paper
                    model.load_state_dict(weights, strict=True)
                else:
                    weights = DeepLabV3_ResNet50_Weights.DEFAULT
                    model = deeplabv3_resnet50(num_classes=inference_cfg.num_classes, weights=weights)
            else:
                model = deeplabv3_resnet50(num_classes=inference_cfg.num_classes, weights=None)
            return model
        
        elif inference_type == "mcd":
        
            from .models.deeplabv3.deeplab_classifier import deeplabv3_resnet50_MCD
            model = deeplabv3_resnet50_MCD(num_classes=inference_cfg.num_classes)
            if pretrained:
                weights = torch.load(str(inference_cfg.weights_path), weights_only=True)
                model.load_state_dict(weights, strict=True)
            return model
        
        else:
            raise NotImplementedError("Inference type not implemented")
    
    elif model_name == "deeplabv3_resnet101":
        if inference_type == "deterministic":
            from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
            if pretrained:
                if inference_cfg.weights_path is not None:
                    weights = torch.load(str(inference_cfg.weights_path), weights_only=True)
                    model = deeplabv3_resnet101(num_classes=inference_cfg.num_classes, weights=None)
                else:
                    weights = DeepLabV3_ResNet101_Weights.DEFAULT
            else:
                weights = None
            return deeplabv3_resnet101(num_classes=inference_cfg.num_classes, weights=weights) 
        elif inference_type == "mcd":
            raise NotImplementedError(f"Model not type not implemented for: {model_name}")
        else:
            raise NotImplementedError("Inference type not implemented")
    else:
        raise NotImplementedError("Model not implemented")
