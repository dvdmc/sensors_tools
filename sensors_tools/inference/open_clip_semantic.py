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
from sensors_tools.inference.models.clip import tokenize # TODO: This should be moved to get_tokenizer interface

from . import get_model

@dataclass
class OpenClipSemanticInferenceConfig(SemanticInferenceConfig):
    """
        Configuration class for the semantic inference
        default model_name: clip_ViT-L/14@336px_open
    """
    classes_text: str = "cables"
    """ Name of the classes to be predicted """
    skip_center_crop: bool = True
    """ Whether to skip center crop """

class OpenClipSemanticInference(SemanticInference):
    def __init__(self, cfg: OpenClipSemanticInferenceConfig):
        super().__init__(cfg)
        self.cfg = cfg
        assert self.cfg.num_classes == 2, "Open semantic inference only works with 2 classes"
        assert self.cfg.labels_name == "binary", "Open semantic inference only works with binary labels"

    def setup(self):

        # NN configuration
        self.gpu_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Get the color map
        self.color_map = get_color_map(self.cfg.labels_name, bgr=False)

        # Setup the semantic inference model
        self.model, self.preprocess = get_model(self.cfg, self.gpu_device, pretrained = True)
        self.model.eval()
        print(f"Detecting: {self.cfg.classes_text}")
        self.tokens = tokenize(self.cfg.classes_text).to(self.gpu_device)
        self.text_embs = self.model.encode_text(self.tokens)
        self.text_embs /= self.text_embs.norm(dim=-1, keepdim=True)

        # Patch the preprocess if we want to skip center crop
        if self.cfg.skip_center_crop:
            # Check there is exactly one center crop transform
            is_center_crop = [isinstance(t, CenterCrop) for t in self.preprocess.transforms]
            assert (
                sum(is_center_crop) == 1
            ), "There should be exactly one CenterCrop transform"
            # Create new preprocess without center crop
            self.preprocess = Compose(
                [t for t in self.preprocess.transforms if not isinstance(t, CenterCrop)]
            )
            print("Skipping center crop")
            
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
        pil_image = Image.fromarray(img) # Preprocess assumes a PIL image
        img_t: torch.Tensor = self.preprocess(pil_image) # type: ignore
        img_t = img_t.unsqueeze(0) # Batch the single image
        img_t = img_t.to(self.gpu_device)

        # We use a batch inference approach for using the implementation in f3rm
        embeddings = []
        with torch.no_grad():
            embeddings.append(self.model.get_patch_encodings(img_t))

            embeddings = torch.cat(embeddings, dim=0)

            # Reshape embeddings from flattened patches to patch height and width
            h_in, w_in = img_t.shape[-2:]
            if "ViT" in self.cfg.model_name:
                h_out = h_in // self.model.visual.patch_size
                w_out = w_in // self.model.visual.patch_size
            elif "RN" in self.cfg.model_name:
                h_out = max(h_in / w_in, 1.0) * self.model.visual.attnpool.spacial_dim
                w_out = max(w_in / h_in, 1.0) * self.model.visual.attnpool.spacial_dim
                h_out, w_out = int(h_out), int(w_out)
            else:
                raise ValueError(f"Unknown CLIP model name: {self.cfg.model_name}")
            embeddings = rearrange(embeddings, "b (h w) c -> b h w c", h=h_out, w=w_out)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.squeeze(0)
            # Compute similarities
            sims = embeddings @ self.text_embs.T
            # print(f"RANGE: {sims.min()} - {sims.max()}")
            # sims_norm = (sims - sims.min()) / (sims.max() - sims.min())

            probs = recover_size(sims.permute(2,0,1)) * 3

            # Extend the classes channel to include 1-p for class 0
            probs_2 = torch.cat((1-probs, probs), dim=0)
            probs_np = probs_2.permute(1, 2, 0).cpu().numpy()
            # Get label prediction for visualization
            pred = torch.argmax(probs_2, dim=0).cpu().numpy()

            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
            probs_normalized = norm(probs.squeeze().cpu().numpy())  # Normalize probs
            cmap = plt.get_cmap("turbo")
            img_out = cmap(probs_normalized)  # Apply colormap
            img_out = (img_out[:, :, :3] * 255).astype(np.uint8)  # Drop alpha channel and convert to uint8
            
            # img_out = self.overlay_label_rgb(pred, img)

            out = {'probs': probs_np, 'img_out': img_out}

            return out
