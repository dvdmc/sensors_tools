#!/usr/bin/env python3

import rospy
from semantic_ros import SemanticNode

if __name__ == '__main__':
    rospy.init_node('semantic_node', anonymous=True)
    #TODO: Continue here
    node = SemanticNode()

    rospy.spin()