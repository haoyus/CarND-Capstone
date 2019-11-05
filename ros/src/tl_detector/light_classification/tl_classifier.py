from styx_msgs.msg import TrafficLight

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
from io import StringIO
from collections import defaultdict
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import time
from scipy.stats import norm
import cv2

cwd = os.path.dirname(os.path.realpath(__file__))

class TLClassifier(object):
    def __init__(self, thresh):
        self.thresh = thresh
        print('Initializing Classifier with threshold: ',self.thresh)
        print('Tensorflow version: ',tf.__version__)

        os.chdir(cwd)

        self.DETECT_MODEL_NAME = 'mymodel'
        self.PATH_TO_CKPT = self.DETECT_MODEL_NAME + '/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()

        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True

        #load frozen graph into memory
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.sess = tf.compat.v1.Session(graph=self.detection_graph)
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        self.cmap = ImageColor.colormap
        print("Number of colors =", len(self.cmap))
        self.COLOR_LIST = sorted([c for c in self.cmap.keys()])


    def get_classification(self, image, visualize = False):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        """
        MyModel is trained using Kaggle LISA dataset, and the output labels will be:
        1-Red
        2-Yellow
        3-Green
        4-Unknown
        """
        state = TrafficLight.UNKNOWN

        with self.detection_graph.as_default():
            image_np = self.load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})

            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)

            boxes, scores, classes = self.filter_boxes(self.thresh, boxes, scores, classes)
            det_num = len(boxes)
            if det_num == 0:
                #print('No valid detection')
                a = 1
            else:
                #print(classes)
                for i in range(det_num):
                    if classes[i] == 1:
                        state = TrafficLight.RED
                        break

            if visualize:
                # The current box coordinates are normalized to a range between 0 and 1.
                # This converts the coordinates actual location on the image.
                height,width,channels  = image.shape
                box_coords = self.to_image_coords(boxes, height, width)

                # Each class with be represented by a differently colored box
                #self.draw_boxes(image, box_coords, classes)
                for i in range(len(box_coords)):
                    cv2.rectangle(image, (box_coords[i][1], box_coords[i][0]),
                                         (box_coords[i][3], box_coords[i][2]), (255, 0, 0), 2)

                cv2.imshow('detection', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                #cv2.imshow('detection', image)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()


        return state

    def load_image_into_numpy_array(self, image):
        #(im_width, im_height, im_depth)= image.shape
        #print(image.shape)
        return np.asarray(image, dtype="uint8")
        # or
        #(im_width, im_height) = image.size
        #return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


    # Get the part of image that's of a specific class from a frame
    def get_detected_boxes(self, target_cl, boxes, scores, classes):
        boxes_slc = []
        scores_slc = []
        classes_slc = []

        for i, cl in enumerate(classes):
            if cl == target_cl:
                boxes_slc.append(boxes[i].tolist())
                scores_slc.append(scores[i])
                classes_slc.append(classes[i])
        return np.array(boxes_slc)

    #
    # helper funcs
    #

    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes


    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].

        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes, dtype=int)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width

        return box_coords


    def draw_boxes(self, image, boxes, classes, thickness=4):
        """Draw bounding boxes on the image"""
        draw = ImageDraw.Draw(image)
        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            class_id = int(classes[i])
            color = self.COLOR_LIST[class_id]
            draw.line([(left, top), (left, bot), (right, bot),
                       (right, top), (left, top)], width=thickness, fill=color)


    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph
