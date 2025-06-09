from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Optional, Tuple
from matplotlib import pyplot as plt

import numpy as np
import torch
from PIL import Image

from torchvision import transforms

from sensors_tools.utils.semantics_utils import get_color_map, label2rgb

from . import get_model

@dataclass
class ClassicSemanticSegmentationConfig:
    """
        Configuration class for the semantic inference
    """
    inference_type: str = "classic"
    """ Type of inference."""

    model_name: str = "deeplabv3"
    """ Name of the model to be used."""

    encoder_name: str = "resnet50"
    """ Name of the encoder to be used."""

    num_classes: int = 21
    """ Number of classes for the semantic inference """

    labels_name: str = "coco_voc"
    """ Name reference the label map (dataset or modified map: coco, coco_voc) """

    weights_path: Optional[Path] = None
    """ Path to the weights of the model """

    width: int = 512
    """ Default model input width """

    height: int = 512
    """ Default model input height """

class ClassicSemanticSegmentation:
    def __init__(self, cfg: ClassicSemanticSegmentationConfig):
        self.cfg = cfg

    def setup(self):

        # NN configuration
        self.gpu_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Get the color map
        self.color_map = get_color_map(self.cfg.labels_name, bgr=False)

        # Setup the semantic inference model
        custom_weights: bool = self.cfg.weights_path is not None # TODO: Handle this correctly
        self.model, self.preprocess = get_model(self.cfg, self.gpu_device, pretrained = True)
        self.model.eval()

        self.softmax = torch.nn.Softmax(dim=1).to(self.gpu_device) # Assumes it is applied to batch
        
    def overlay_label_rgb(self, pred: np.ndarray, img: np.ndarray) -> np.ndarray:
        """
            Overlay the predicted label on top of the RGB image
            Args:
                pred: predicted label
                img: RGB image
            Returns:
                img_out: RGB image with the predicted label overlayed
        """
        # Get the predicted class as an RGB image
        r = pred.astype(np.uint8)
        # Filter by prob threshold
        # r = self.label2rgb(self.gt_label, self.color_map_gt)
        r = label2rgb(r, self.color_map)
        # r[self.pred_class_prob < 0.5] = 0
        r = Image.fromarray(r)
        # Convert the image to RGBA for merging with RGB image
        r = r.convert('RGBA')
        datas = r.getdata() # This returns an internal PIL sequence data type. We ignore its type below
        newData = []
        for item in datas: # type: ignore
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                newData.append((0, 0, 0, 0))
            else:
                newData.append(item)
        r.putdata(newData)

        # Merge predicted color image with RGB image
        input_alpha_image = Image.fromarray(img).convert('RGBA')
        img_out = Image.alpha_composite(input_alpha_image, r).convert("RGB")
        img_out = np.array(img_out)

        return img_out

    def get_prediction(self, img: np.ndarray) -> dict:
        """
            Get the prediction from the model assuming it is classic
            Args:
                img: image to be processed
            Returns:
                out: dictionary with the outputs. For this inference model: probs, img_out
        """
        prev_height = img.shape[0]
        prev_width = img.shape[1]
        recover_size = transforms.Resize((prev_height, prev_width), interpolation=transforms.InterpolationMode.NEAREST)
        # We force below to be a tensor
        img_t: torch.Tensor = self.preprocess(img) # type: ignore
        img_t = img_t.unsqueeze(0)
        img_t = img_t.to(self.gpu_device)
        with torch.no_grad():
            output_logs = self.model(img_t)['out'] # TODO: We have to change this to handle generic outputs instead of deeplabv3 outputs
            probs = self.softmax(output_logs)

        # Resize to original size
        probs = recover_size(probs[0]) # Remove batch dimension
        probs_np = probs.permute(1, 2, 0).cpu().numpy() # HxWxC

        # Get label prediction for visualization
        pred = torch.argmax(probs, dim=0).cpu().numpy()
        img_out = self.overlay_label_rgb(pred, img)
        
        out = {'probs': probs_np, 'img_out': img_out}

        return out
