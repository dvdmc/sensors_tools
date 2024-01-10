###
#
# This file manages the bridges module.
#
###
from typing import Literal, Union, TYPE_CHECKING

# We use dynamic imports to avoid not used requirements.
# For this to work, used types must be "forward declared" in quotes 
# (see https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING)
# Then, if bridge is selected, we can import the required module
if TYPE_CHECKING:
    from .airsim_bridge import AirsimBridge, AirsimBridgeConfig
    from .ros_bridge import ROSBridge, ROSBridgeConfig
    from .scannet_bridge import ScanNetBridge, ScanNetBridgeConfig
    from .test_bridge import TestBridge, TestBridgeConfig

BridgeConfig = Union['ScanNetBridgeConfig', 'TestBridgeConfig', 'AirsimBridgeConfig', 'ROSBridgeConfig']
BridgeType = Literal["airsim", "ros", "scannet", "test"]
Bridges = Union['ScanNetBridge', 'TestBridge', 'AirsimBridge', 'ROSBridge']

def get_bridge(bridge_type: BridgeType, bridge_cfg: BridgeConfig) -> Bridges:
    assert bridge_type is not None, "Bridge type must be specified"
    assert bridge_cfg is not None, "Bridge cfg must be specified"

    if bridge_type == "airsim":
        from .airsim_bridge import AirsimBridge, AirsimBridgeConfig
        assert isinstance(bridge_cfg, AirsimBridgeConfig), "Bridge cfg must be of type AirsimBridgeConfig"
        return AirsimBridge(bridge_cfg)
    if bridge_type == "scannet":
        from .scannet_bridge import ScanNetBridge, ScanNetBridgeConfig
        assert isinstance(bridge_cfg, ScanNetBridgeConfig), "Bridge cfg must be of type ScanNetBridgeConfig"
        return ScanNetBridge(bridge_cfg)
    elif bridge_type == "ros":
        from .ros_bridge import ROSBridge, ROSBridgeConfig
        assert isinstance(bridge_cfg, ROSBridgeConfig), "Bridge cfg must be of type ROSBridgeConfig"
        return ROSBridge(bridge_cfg)
    elif bridge_type == "test":
        from .test_bridge import TestBridge, TestBridgeConfig
        assert isinstance(bridge_cfg, TestBridgeConfig), "Bridge cfg must be of type TestBridgeConfig"
        return TestBridge(bridge_cfg)
    else:
        raise NotImplementedError("Bridge type not implemented")
    