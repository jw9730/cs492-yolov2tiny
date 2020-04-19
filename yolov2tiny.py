import os
import sys
import pickle
import numpy as np
import tensorflow as tf


class YOLO_V2_TINY(object):

    def __init__(self, in_shape, weight_pickle, proc="cpu"):
        self.g = tf.Graph()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config, graph=self.g)
        self.proc = proc
        self.weight_pickle = weight_pickle
        self.input_tensor, self.tensor_list = self.build_graph(in_shape)

    def build_graph(self, in_shape):
        #
        # This function builds a tensor graph. Once created,
        # it will be used to inference every frame.
        #
        # Your code from here. You may clear the comments.
        # #
        # print('build_graph is not yet implemented')
        # sys.exit()

        # Load weight parameters from a pickle file.
        with open(self.weight_pickle, 'rb') as h:
            w = pickle.load(h, encoding='latin1')

        print("Type: {}".format(type(w)))
        print("Length of list:: {}".format(type(len(w))))
        print("Type of list element: {}".format(type(w[0])))
        for i in range(len(w)):
            print("Conv{}".format(i))
            for k in w[i].keys():
                print("\tConv{}[{}]: {}".format(i, k, w[i][k].shape))

        raise NotImplementedError

        bn_epsilon = 1e-5
        n_input_imgs = 1

        # Create an empty list for tensors.
        tensor_list = []
        with self.g.as_default():
            with tf.name_scope('input'):
                input_tensor = tf.placeholder(tf.float32, shape=[n_input_imgs, in_shape[0], in_shape[1], in_shape[2]])
                # labels = tf.placeholder(tf.float32, shape=[n_input_imgs, 1])

                # 1 conv1     16  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  16
                w1 = tf.Variable(tf.truncated_normal([3, 3, 3, 16], stddev=0.1))
                b1 = tf.Variable(tf.constant(0.0, shape=[16]))
                h1 = tf.nn.conv2d(input_tensor, w1, strides=[1, 1, 1, 1], padding='SAME') + b1
                o1 = tf.maximum(0.01 * h1, h1)
                num_params = 3 * 3 * 3 * 16 + 16 * 4

                # 2 max1          2 x 2 / 2   416 x 416 x  16   ->   208 x 208 x  16
                mp1 = tf.nn.max_pool(o1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

                # 3 conv2     32  3 x 3 / 1   208 x 208 x  16   ->   208 x 208 x  32
                w2 = tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=0.1))
                b2 = tf.Variable(tf.constant(0.0, shape=[32]))
                h2 = tf.nn.conv2d(mp1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2
                o2 = tf.maximum(0.01 * h2, h2)
                num_params = num_params + 3 * 3 * 16 * 32 + 32 * 4

                # 4 max2          2 x 2 / 2   208 x 208 x  16   ->   104 x 104 x  32
                mp2 = tf.nn.max_pool(o2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

                # 5 conv     64  3 x 3 / 1   104 x 104 x  32   ->   104 x 104 x  64
                w3 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
                b3 = tf.Variable(tf.constant(0.0, shape=[64]))
                h3 = tf.nn.conv2d(mp2, w3, strides=[1, 1, 1, 1], padding='SAME') + b3
                o3 = tf.maximum(0.01 * h3, h3)
                num_params = num_params + 3 * 3 * 32 * 64 + 64 * 4

                # 6 max3          2 x 2 / 2   104 x 104 x  64   ->    52 x  52 x  64
                mp3 = tf.nn.max_pool(o3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

                # 7 conv4    128  3 x 3 / 1    52 x  52 x  64   ->    52 x  52 x 128
                w4 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
                b4 = tf.Variable(tf.constant(0.0, shape=[128]))
                h4 = tf.nn.conv2d(mp3, w4, strides=[1, 1, 1, 1], padding='SAME') + b4
                o4 = tf.maximum(0.01 * h4, h4)
                num_params = num_params + 3 * 3 * 64 * 128 + 128 * 4

                # 8 max4          2 x 2 / 2    52 x  52 x 128   ->    26 x  26 x 128
                mp4 = tf.nn.max_pool(o4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

                # 9 conv5    256  3 x 3 / 1    26 x  26 x 128   ->    26 x  26 x 256
                w5 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1))
                b5 = tf.Variable(tf.constant(0.0, shape=[256]))
                h5 = tf.nn.conv2d(mp4, w5, strides=[1, 1, 1, 1], padding='SAME') + b5
                o5 = tf.maximum(0.01 * h5, h5)
                num_params = num_params + 3 * 3 * 128 * 256 + 256 * 4

                # 10 max5          2 x 2 / 2    26 x  26 x 256   ->    13 x  13 x 256
                mp5 = tf.nn.max_pool(o5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

                # 11 conv6   512  3 x 3 / 1    13 x  13 x 256   ->    13 x  13 512
                w6 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.1))
                b6 = tf.Variable(tf.constant(0.0, shape=[512]))
                h6 = tf.nn.conv2d(mp5, w6, strides=[1, 1, 1, 1], padding='SAME') + b6
                o6 = tf.maximum(0.01 * h6, h6)
                num_params = num_params + 3 * 3 * 256 * 512 + 512 * 4

                # 12 max6          2 x 2 / 1    13 x  13 x 512   ->    13 x  13 x 512
                mp6 = tf.nn.max_pool(o6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

                # 13 conv7    1024  1 x 1 / 1    13 x  13 x512   ->    13 x  13 x 1024
                w7 = tf.Variable(tf.truncated_normal([3, 3, 512, 1024], stddev=0.1))
                b7 = tf.Variable(tf.constant(0.0, shape=[1024]))
                h7 = tf.nn.conv2d(mp6, w7, strides=[1, 1, 1, 1], padding='SAME') + b7
                o7 = tf.maximum(0.01 * h7, h7)
                num_params = num_params + 3 * 3 * 512 * 1024 + 1024 * 4

                # 14 conv8   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024
                w8 = tf.Variable(tf.truncated_normal([3, 3, 1024, 1024], stddev=0.1))
                b8 = tf.Variable(tf.constant(0.0, shape=[1024]))
                h8 = tf.nn.conv2d(o7, w8, strides=[1, 1, 1, 1], padding='SAME') + b8
                o8 = tf.maximum(0.01 * h8, h8)
                num_params = num_params + 3 * 3 * 1024 * 1024 + 1024 * 4

                # 15 conv9   125  1 x 1 / 1    13 x  13 x 1024   ->    13 x  13 x125
                w9 = tf.Variable(tf.truncated_normal([1, 1, 1024, 125], stddev=0.1))
                b9 = tf.Variable(tf.constant(0.0, shape=[125]))
                h9 = tf.nn.conv2d(o8, w9, strides=[1, 1, 1, 1], padding='SAME') + b9
                o9 = h9
                num_params = num_params + 1 * 1 * 1024 * 125 + 125

        # Use self.g as a default graph. Refer to this API.
        ## https://www.tensorflow.org/api_docs/python/tf/Graph#as_default

        # Then you need to declare which device to use for tensor computation. The device info
        # is given from the command line argument and stored somewhere in this object.
        # In this project, you may choose CPU or GPU. Consider using the following API.
        ## https://www.tensorflow.org/api_docs/python/tf/Graph#device

        # Then you are ready to add tensors to the graph. According to the Yolo v2 tiny model,
        # build a graph and append the tensors to the returning list for computing intermediate
        # values. One tip is to start adding a placeholder tensor for the first tensor.
        # (Use 1e-5 for the epsilon value of batch normalization layers.)

        # Return the start tensor and the list of all tensors.
        return input_tensor, tensor_list

    def inference(self, img):
        feed_dict = {self.input_tensor: img}
        out_tensors = self.sess.run(self.tensor_list, feed_dict)
        return out_tensors


#
# Codes belows are for postprocessing step. Do not modify. The postprocessing 
# function takes an input of a resulting tensor as an array to parse it to
# generate the label box positions. It returns a list of the positions which
# composed of a label, two coordinates of left-top and right-bottom of the box
# and its color.
#

def postprocessing(predictions, w0, h0):
    """
    :param predictions: (1, 125, 13, 13) numpy array
        Each grid cell corresponds to 125 channels, made up of the 5 bounding boxes predicted by the grid cell
        and the 25 data elements that describe each bounding box.
    :param w0: integer
        Original video width
    :param h0: integer
        Original video height
    :return: bbox_list: list of tuples (x, y, w, h, text), representing bbox of confidence > 0.25
        x, y: bbox position in global coordinate, in respect to original image
        w, h: bbox size in global coordinate, in respect to original image
        text: box representative text (i.e. semantic category)
    """

    n_classes = 20
    n_grid_cells = 13
    n_b_boxes = 5
    n_b_box_coord = 4

    # Names and colors for each class
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    colors = [(254.0, 254.0, 254), (239.88888888888889, 211.66666666666669, 127),
              (225.77777777777777, 169.33333333333334, 0), (211.66666666666669, 127.0, 254),
              (197.55555555555557, 84.66666666666667, 127), (183.44444444444443, 42.33333333333332, 0),
              (169.33333333333334, 0.0, 254), (155.22222222222223, -42.33333333333335, 127),
              (141.11111111111111, -84.66666666666664, 0), (127.0, 254.0, 254),
              (112.88888888888889, 211.66666666666669, 127), (98.77777777777777, 169.33333333333334, 0),
              (84.66666666666667, 127.0, 254), (70.55555555555556, 84.66666666666667, 127),
              (56.44444444444444, 42.33333333333332, 0), (42.33333333333332, 0.0, 254),
              (28.222222222222236, -42.33333333333335, 127), (14.111111111111118, -84.66666666666664, 0),
              (0.0, 254.0, 254), (-14.111111111111118, 211.66666666666669, 127)]

    # Pre-computed YOLOv2 shapes of the k=5 B-Boxes
    """ [p_w, p_h] for first bbox, [p_w, p_h] for second bbox, ... (distance in output cell space)
    """
    anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

    """ Image pixel distance per one output cell width / height
    """

    thresholded_predictions = []

    # IMPORTANT: reshape to have shape = [ 13 x 13 x (5 B-Boxes) x (4 Coords + 1 Obj score + 20 Class scores ) ]
    predictions = np.reshape(predictions, (13, 13, 5, 25))

    # IMPORTANT: Compute the coordinates and score of the B-Boxes by considering the parametrization of YOLOv2
    for row in range(n_grid_cells):
        for col in range(n_grid_cells):
            for b in range(n_b_boxes):

                tx, ty, tw, th, tc = predictions[row, col, b, :5]

                # IMPORTANT: (416 img size) / (13 grid cells) = 32!
                # YOLOv2 predicts parametrized coordinates that must be converted to full size
                # final_coordinates = parametrized_coordinates * 32.0 ( You can see other EQUIVALENT ways to do this...)
                """ Position in cell space -> Position in original video pixel space
                """
                center_x = (float(col) + sigmoid(tx)) * 32.0 * (w0 / 416)
                center_y = (float(row) + sigmoid(ty)) * 32.0 * (h0 / 416)

                """ Size in cell space -> Size in original video pixel space
                """
                roi_w = np.exp(tw) * anchors[2 * b + 0] * 32.0 * (w0 / 416)
                roi_h = np.exp(th) * anchors[2 * b + 1] * 32.0 * (h0 / 416)

                final_confidence = sigmoid(tc)

                # Find best class
                class_predictions = predictions[row, col, b, 5:]
                class_predictions = softmax(class_predictions)

                class_predictions = tuple(class_predictions)
                best_class = class_predictions.index(max(class_predictions))
                best_class_score = class_predictions[best_class]

                # Flip the coordinates on both axes
                left = int(center_x - (roi_w / 2.))
                right = int(center_x + (roi_w / 2.))
                top = int(center_y - (roi_h / 2.))
                bottom = int(center_y + (roi_h / 2.))

                if (final_confidence * best_class_score) > 0.3:
                    thresholded_predictions.append(
                        [[left, top, right, bottom], final_confidence * best_class_score, classes[best_class]])

    # Sort the B-boxes by their final score
    thresholded_predictions.sort(key=lambda tup: tup[1], reverse=True)

    # Non maximal suppression
    if len(thresholded_predictions) > 0:
        nms_predictions = non_maximal_suppression(thresholded_predictions, 0.3)
    else:
        nms_predictions = []

    label_boxes = []
    # Append B-Boxes
    for i in range(len(nms_predictions)):
        best_class_name = nms_predictions[i][2]
        lefttop = tuple(nms_predictions[i][0][0:2])
        rightbottom = tuple(nms_predictions[i][0][2:4])
        color = colors[classes.index(nms_predictions[i][2])]

        label_boxes.append((best_class_name, lefttop, rightbottom, color))

    return label_boxes


def iou(boxA, boxB):
    # boxA = boxB = [x1,y1,x2,y2]

    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    intersection_area = (xB - xA + 1) * (yB - yA + 1)

    # Compute the area of both rectangles
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the IOU
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

    return iou


def non_maximal_suppression(thresholded_predictions, iou_threshold):
    nms_predictions = []

    # Add the best B-Box because it will never be deleted
    nms_predictions.append(thresholded_predictions[0])

    # For each B-Box (starting from the 2nd) check its iou with the higher score B-Boxes
    # thresholded_predictions[i][0] = [x1,y1,x2,y2]
    i = 1
    while i < len(thresholded_predictions):
        n_boxes_to_check = len(nms_predictions)
        # print('N boxes to check = {}'.format(n_boxes_to_check))
        to_delete = False

        j = 0
        while j < n_boxes_to_check:
            curr_iou = iou(thresholded_predictions[i][0], nms_predictions[j][0])
            if (curr_iou > iou_threshold):
                to_delete = True
            # print('Checking box {} vs {}: IOU = {} , To delete = {}'.format(thresholded_predictions[i][0],nms_predictions[j][0],curr_iou,to_delete))
            j = j + 1

        if to_delete == False:
            nms_predictions.append(thresholded_predictions[i])
        i = i + 1

    return nms_predictions


def sigmoid(x):
    return 1 / (1 + np.e ** -x)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
