from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Optional, Tuple
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from einops import rearrange

import numpy as np
import torch
from PIL import Image

from torchvision import transforms
from torchvision.transforms import CenterCrop, Compose

from sensors_tools.inference.semantic import SemanticInference, SemanticInferenceConfig
from sensors_tools.utils.semantics_utils import get_color_map, label2rgb

from . import get_model

@dataclass
class OpenTridentSemanticInferenceConfig(SemanticInferenceConfig):
    """
        Configuration class for the semantic inference
    """
    class_names: str = "sheep,human,grass,sky,house"
    """ Name of the classes to be predicted. Classes must be separated with comma and subclasses must be separated with ; """
    sam_checkpoint: Path = Path("/home/david/git/Trident/sam_vit_b_01ec64.pth")
    """ Path to the SAM checkpoint """
    sam_model_type: str = "vit_b"
    """ Type of the SAM model """
    coarse_threshold: float = 0.2
    """ Threshold for the SAM refinement """

class OpenTridentSemanticInference(SemanticInference):
    def __init__(self, cfg: OpenTridentSemanticInferenceConfig):
        super().__init__(cfg)
        self.cfg = cfg
        # assert self.cfg.labels_name == "ade20k", "Open trident semantic inference works for many classes, use ade20k map for now." # TODO: Use generic color map

    def setup(self):

        # NN configuration
        self.gpu_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Get the color map
        self.color_map = get_color_map(self.cfg.labels_name, bgr=False)

        # Setup the semantic inference model
        self.model, self.preprocess  = get_model(self.cfg, self.gpu_device, pretrained = True)
        self.softmax = torch.nn.Softmax(dim=0).to(self.gpu_device)
        
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
        r = r.convert('RGBA').resize((self.cfg.width, self.cfg.height), resample=Image.Resampling.NEAREST)
        datas = r.getdata() # This returns an internal PIL sequence data type. We ignore its type below
        newData = []
        for item in datas: # type: ignore
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                newData.append((0, 0, 0, 0))
            else:
                newData.append(item)
        r.putdata(newData)

        # Merge predicted color image with RGB image
        input_alpha_image = Image.fromarray(img).convert('RGBA').resize((self.cfg.width, self.cfg.height))
        img_out = Image.alpha_composite(input_alpha_image, r).convert("RGB")
        img_out = np.array(img_out)

        return img_out

    def get_prediction(self, img: np.ndarray) -> dict:
        """
            Get the prediction from the model assuming it is deterministic
            Args:
                img: image to be processed
            Returns:
                out: dictionary with the outputs. For this inference model: probs, img_out
        """
        recover_size = transforms.Resize((self.cfg.height, self.cfg.width), interpolation=transforms.InterpolationMode.NEAREST)
        # We force below to be a tensor
        img_pil = Image.fromarray(img) # Preprocess assumes a PIL image
        img_t: torch.Tensor = self.preprocess(img_pil) # type: ignore
        img_t = img_t.unsqueeze(0) # Batch the single image
        img_t = img_t.to(self.gpu_device)

        with torch.no_grad():
            # pred are the labels so the size is [1, h, w] and logits size is [num_classes, h, w]
            # outputs have no batch
            pred, logits = self.model.predict(img_pil, img_t)
            seg_probs = self.softmax(logits)

            probs_np = seg_probs.permute(1, 2, 0).cpu().numpy()

            # Get label prediction for visualization
            pred = pred.permute(1, 2, 0).cpu().numpy()[:,:,0]

            # Get the probability of the predicted label
            # pred_probs = probs_np[np.arange(probs_np.shape[0])[:, None],
            #                     np.arange(probs_np.shape[1])[None, :],
            #                     pred.astype(int)]  # Shape: [h, w]
            
            # cmap = plt.get_cmap("turbo")
            # img_out = cmap(pred_probs)  # Apply colormap
            # img_out = (img_out[:, :, :3] * 255).astype(np.uint8)  # Drop alpha channel and convert to uint8

            img_out = self.overlay_label_rgb(pred, img)

            out = {'probs': probs_np, 'img_out': img_out}

            return out
