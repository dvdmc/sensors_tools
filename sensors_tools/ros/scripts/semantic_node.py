#!/usr/bin/env python3
import rospy

from sensors_tools.sensor import SensorConfig

from .semantic_ros import SemanticNode

if __name__ == '__main__':
    rospy.init_node('semantic_node', anonymous=True)
    # Load dummy cfg
    cfg = SensorConfig()
    node = SemanticNode(cfg)
    rospy.spin()