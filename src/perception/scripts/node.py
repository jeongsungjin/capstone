#!/usr/bin/env python3
import rospy
import rospack

from sensor_msgs.msg import CompresssedImage, Image

from cv_bridge import CvBridge

from taf import TAFtrt

class PerceptionNode:
    def __init__(self):
        rospy.init_node('perception')

        rospkg = rospack.RosPkg()
        path = rospkg.get_path('perception')
        self.taf = TAFtrt(path)

        self.bridge = CvBridge()

        rospy.Subscriber('/camera?', Image, self.image_callback)
        self.perception_result_pub = rospy.Publisher('/img', Image, queue_size=1)

    def image_callback(self, msg: Image):
        cv2_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        perception_result = self.taf.process(cv2_img)
        perception_result_msg = self.bridge.cv2_to_imgmsg(perception_result)
        self.perception_result_pub(perception_result_msg)
    
if __name__ == '__main__':
    node = PerceptionNode()
    rospy.spin()

