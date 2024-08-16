#!/usr/bin/env python3

import rospy
from semantic_ros import SemanticNode

if __name__ == '__main__':
    rospy.init_node('semantic_node', anonymous=True)
    node = SemanticNode()

    rospy.spin()