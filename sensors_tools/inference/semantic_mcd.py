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

    def setup(self):
        super().setup()

        #We set the dropout layers active during inference!
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                print('We found a dropout layer', m)
                m.train()
        
    def get_prediction(self, img: np.ndarray) -> dict:
        """
            Get the prediction from the model assuming it is deterministic
            Args:
                img: image to be processed
            Returns:
                out: dictionary with the outputs. For this inference model: probs, img_out
        """
        img_t: torch.Tensor = self.transform_img(img)
        img_t = img_t.unsqueeze(0)
        img_t = img_t.to(self.gpu_device)
        accumulated_probs = torch.zeros((self.cfg.num_mc_samples, self.cfg.num_classes, img.shape[2], img.shape[3])).to(self.gpu_device)
        num_pass = 0
        with torch.no_grad():
            for n_pass in range(self.cfg.num_mc_samples):
                output_logs = self.model(img)['out']
                probs = self.softmax(output_logs)
                accumulated_probs[n_pass] = probs[0]
        
        average_probs = torch.mean(accumulated_probs, dim = 0)
        pred = torch.max(average_probs, dim = 0)
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
