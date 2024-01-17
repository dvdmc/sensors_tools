from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from matplotlib import pyplot as plt

import numpy as np
import torch
from PIL import Image
import cv2

from torchvision import transforms

from sensors_tools.utils.semantics_utils import get_color_map, label2rgb

from . import get_model
from sensors_tools.inference.semantic import SemanticInference, SemanticInferenceConfig

@dataclass
class SemanticMCDInferenceConfig(SemanticInferenceConfig):
    """
        Configuration class for the semantic inference
    """
    num_mc_samples: int = 10
    """ Number of Monte Carlo samples """

class SemanticMCDInference(SemanticInference):
    def __init__(self, cfg: SemanticMCDInferenceConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def setup(self):

        # NN configuration
        self.gpu_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Get the color map
        self.color_map = get_color_map(self.cfg.labels_name, bgr=False)

        # Setup the semantic inference model
        custom_weights: bool = self.cfg.weights_path is not None
        self.model = get_model("mcd", self.cfg, pretrained = not custom_weights)

        # Load the pretrained model
        if custom_weights:
            load_weights = False
            weights = torch.load(str(self.cfg.weights_path), map_location=self.gpu_device)
            self.model.load_state_dict(weights['model_state_dict'], strict=True)

        

        self.model.to(self.gpu_device)
        self.model.eval()
        #We set the dropout layers active during inference!
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                print('We found a dropout layer', m)
                m.train()
    
        self.transform_img = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((self.cfg.height, self.cfg.width), antialias=True), # type: ignore
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

        self.softmax = torch.nn.Softmax(dim=1).to(self.gpu_device) # Assumes it is applied to batch
        
    def get_prediction(self, img: np.ndarray) -> dict:
        """
            Get the prediction from the model assuming it is deterministic
            Args:
                img: image to be processed
            Returns:
                out: dictionary with the outputs. For this inference model: probs, img_out
        """
        prev_width = img.shape[1]
        prev_height = img.shape[0]
        img_t: torch.Tensor = self.transform_img(img)
        img_t = img_t.unsqueeze(0)
        img_t = img_t.to(self.gpu_device)
        accumulated_probs = torch.zeros((self.cfg.num_mc_samples, self.cfg.num_classes, img_t.shape[2], img_t.shape[3])).to(self.gpu_device)
        num_pass = 0
        with torch.no_grad():
            for n_pass in range(self.cfg.num_mc_samples):
                output_logs = self.model(img_t)['out']
                probs = self.softmax(output_logs)
                accumulated_probs[n_pass] = probs[0]
        
        average_probs = torch.mean(accumulated_probs, dim = 0)
        pred = torch.argmax(average_probs, dim=0).cpu().numpy()
        # TODO: Variance things
        epistemic_var = torch.var(accumulated_probs, dim = 0).cpu().numpy()
        average_pred_np = average_probs.cpu().numpy()
        accumulated_probs_np = accumulated_probs.cpu().numpy()
        img_out = self.overlay_label_rgb(pred, img)
        out = {
            'probs': average_pred_np,
            'acc_probs': accumulated_probs_np,
            'img_out': img_out,
            'epistemic_var': epistemic_var
        }
        return out
