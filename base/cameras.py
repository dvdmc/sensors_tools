import numpy as np

from dataclasses import dataclass

@dataclass
class CameraInfo:
    """
        Stores camera parameters in a unified way
    """

    width: float
    """ Image width """

    height: float
    """ Image height """

    fx: float
    """ Focal length in x """

    fy: float
    """ Focal length in y """

    cx: float
    """ Principal point in x """

    cy: float
    """ Principal point in y """

    k1: float = 0.0
    """ Radial distortion coefficient k1 """

    k2: float = 0.0
    """ Radial distortion coefficient k2 """

    k3: float = 0.0
    """ Radial distortion coefficient k3 """

    p1: float = 0.0
    """ Tangential distortion coefficient p1 """

    p2: float = 0.0
    """ Tangential distortion coefficient p2 """
    
    @classmethod
    def from_fov_h(cls, width: float, height: float, fov_h: float) -> "CameraInfo":
        """
            Create camera info from fov_h

            Args:
                width: Image width
                height: Image height
                fov_h: Horizontal field of view

            Returns:
                CameraInfo: Camera info
        """
        cx = float(width) / 2
        cy = float(height) / 2
        fov_h_rad = fov_h * np.pi / 180.0
        fx = cx / (np.tan(fov_h_rad / 2))
        fy = fx * height / width
        return cls(width, height, fov_h, fx, fy, cx, cy)