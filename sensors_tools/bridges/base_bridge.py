from dataclasses import dataclass
from typing import List, Literal
from abc import ABC, abstractmethod

@dataclass
class BaseBridgeConfig:
    """
        Base class for bridge configuration
    """

    data_types: List[Literal["image", "depth", "semantic"]]
    """
        List of data types that the bridge will provide
        Each bridge can specialize on this
    """

class BaseBridge(ABC):
    """
        Any bridge class should implement this interface
    """
    def __init__(self, cfg):
        """

        """
        self.cfg = cfg

    @abstractmethod
    def setup(self):
        """
            Setup the bridge
        """
        pass

    @abstractmethod
    def get_data(self) -> dict:
        """
            Get data from the bridge
        """
        pass