from dataclasses import dataclass
import numpy as np
import torch

from sensors_tools.utils.semantics_utils import get_color_map
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
        self.setup()

    def setup(self):
        super().setup()

        #We set the dropout layers active during inference!
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                print('We found a dropout layer', m)
                m.train()
        
    def get_prediction(self, img: torch.Tensor) -> tuple:
        """
            Get the prediction from the model assuming it is deterministic
            Args:
                img: image to be processed
            Returns:
                probs: probability vectors from the model
                img_out: image with the prediction
        """
        img = self.transform_img(img)
        img = img.unsqueeze(0)
        img = img.to(self.gpu_device)
        accumulated_pred = torch.zeros((self.cfg.num_mc_samples, self.cfg.num_classes, img.shape[2], img.shape[3])).to(self.gpu_device)
        num_pass = 0
        with torch.no_grad():
            for n_pass in range(self.cfg.num_mc_samples):
                output_logs = self.model(img)
                probs = self.softmax(output_logs)
                accumulated_pred[n_pass] = probs.squeeze(0)
        
        # TODO: Variance things
        epistemic_var = torch.var(accumulated_pred, dim = 0).cpu().numpy()

        average_pred = torch.mean(accumulated_pred, dim = 0)
        pred_class_prob, pred_class = torch.max(average_pred, dim = 1) # Check dims
        pred_class = np.squeeze(pred_class).cpu()
        pred_class_prob = np.squeeze(pred_class_prob).cpu()
        img_out = self.overlay_label_rgb(pred_class_prob, img)

        return average_pred, epistemic_var, img_out
    
