from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from matplotlib import pyplot as plt

import numpy as np
import torch
from PIL import Image

from torchvision import transforms
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50

from utils.semantics_utils import get_color_map, label2rgb

@dataclass
class SemanticInferenceConfig:
    """
        Configuration class for the semantic inference
    """
    num_classes: int = 7
    """ Number of classes for the semantic inference """

    labels_name: str = "coco_voc"
    """ Name reference the label map (dataset or modified map: coco, coco_voc) """

    pretrained_model_path: Optional[Path] = None
    """ Path to the pretrained model """

    width: int = 512
    """ Default model input width """

    height: int = 512
    """ Default model input height """

class SemanticInference:
    def __init__(self, cfg: SemanticInferenceConfig):
        self.cfg = cfg

    def setup(self):

        # NN configuration
        self.gpu_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Get the color map
        self.color_map = get_color_map(self.cfg.labels_name, bgr=True)

        # Setup the semantic inference model
        # Load the pretrained model TODO: Optional pretrain?
        if self.cfg.pretrained_model_path is not None:
            pretrained_model = torch.load(self.cfg.pretrained_model_path)
            self.model = self.model = deeplabv3_resnet50(num_classes=self.cfg.num_classes)
            self.model.load_state_dict(pretrained_model['model_state_dict'])
        else:
            self.model = deeplabv3_resnet50(num_classes=self.cfg.num_classes, weights='DEFAULT')
            print("Loaded pretrained model")

        self.model.to(self.gpu_device)
        self.model.eval()
    
        self.transform_img = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((self.cfg.height, self.cfg.width)),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

        self.softmax = torch.nn.Softmax(dim=1).to(self.gpu_device)
        
    def overlay_label_rgb(self, pred: np.ndarray, img: np.ndarray) -> Image.Image:
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
        datas = r.getdata()
        newData = []
        for item in datas:
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                newData.append((0, 0, 0, 0))
            else:
                newData.append(item)
        r.putdata(newData)

        # Merge predicted color image with RGB image
        input_alpha_image = Image.fromarray(img).convert('RGBA').resize((self.cfg.width, self.cfg.height))
        img_out = Image.alpha_composite(input_alpha_image, r)
        return img_out

    # TODO: Debug this
    def get_prediction(self, img: Image.Image) -> tuple:
        """
            Get the prediction from the model assuming it is deterministic
            Args:
                img: image to be processed
            Returns:
                probs: probability vectors from the model
                img_out: image with the prediction
        """
        img_t: torch.Tensor = self.transform_img(img)
        # Change order to C,W,H
        img_t = img_t.unsqueeze(0)
        img_t = img_t.to(self.gpu_device)
        with torch.no_grad():
            output_logs = self.model(img_t)['out'][0]
            probs = self.softmax(output_logs)

        probs_np = probs.cpu().numpy()
        pred = np.argmax(probs_np, axis=0)
        img_np = np.array(img)
        img_out = self.overlay_label_rgb(pred, img_np)
        plt.imshow(pred)
        plt.show()
        return pred, img_out
    

    
