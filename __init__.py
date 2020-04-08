import sys
import numpy as np
import cv2
import time
import math
from random import randint
import yolov2tiny


def open_video_with_opencv(in_video_path='sample.mp4', out_video_path='output.mp4'):

    # Open an object of input video using cv2.VideoCapture.
    vcap = cv2.VideoCapture(in_video_path)

    # Get video properties
    if vcap.isOpened():
        width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
        height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
        fps = vcap.get(cv2.CAP_PROP_FPS)  # frames per second
        n_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))  # number of frames
        print("Input video w: %d,"
              " h: %d,"
              " fps: %d frames/s,"
              " total %d frames" % (width, height, fps, n_frames))
    else:
        raise RuntimeError("open_video_with_opencv: Could not open input video")

    # Open an object of output video using cv2.VideoWriter.
    # Same encoding, size, and fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height), True)

    # Return the video objects and anything you want for further process.
    return vcap, out, (width, height, n_frames)


def resize_input(im):
    # im: (h0, w0, 3) numpy array
    imsz = np.asarray(cv2.resize(im, (416, 416)), dtype=np.float32)
    imsz = imsz / 255.
    # imsz = imsz[:, :, ::-1]
    imsz = imsz.transpose((2, 0, 1))
    return imsz


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def postprocess(output, w0, h0):
    """
    :param output: (1, 125, 13, 13) numpy array
        Each grid cell corresponds to 125 channels, made up of the 5 bounding boxes predicted by the grid cell
        and the 25 data elements that describe each bounding box.

    :return: bbox_list: list of tuples (x, y, w, h, text), representing bbox of confidence > 0.25
        x, y: bbox position in global coordinate, in respect to original image
        w, h: bbox size in global coordinate, in respect to original image
        text: box representative text (i.e. semantic category)
    """
    output = output.squeeze(0)  # (125, 13, 13)
    bbox_dim = 25

    # Image pixel distance per one output cell width / height
    x_ratio = 32 * (w0 / 416)
    y_ratio = 32 * (h0 / 416)

    # Pre-defined anchors: Height and width of the 5 anchors defined by YOLOv2
    # Source: https://github.com/simo23/tinyYOLOv2
    # [p_w, p_h] for first bbox, [p_w, p_h] for second bbox, ...
    # (distance in output cell space)
    anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

    bbox_list = list()
    # Iterate over grid cells
    for i in range(13):  # y-axis
        for j in range(13):  # x-axis
            # 125-dim feature = 5 bbox * 25 bbox feature = 5 bbox * (t_x, t_y, t_w, t_h, t_o, logit(category_0...20))
            o = output[:, i, j]
            for bbox_idx in range(5):
                # Parse prediction prior t_x, t_y, t_w, t_h, t_o, logit[0...19]
                bbox_feat = o[bbox_dim*bbox_idx:bbox_dim*(bbox_idx+1)]
                t_x = bbox_feat[0]
                t_y = bbox_feat[1]
                t_w = bbox_feat[2]
                t_h = bbox_feat[3]
                t_o = bbox_feat[4]
                logits = bbox_feat[5:]

                # If box confidence >= 0.2, decode x, y, w, h and bbox category
                p_bbox = sigmoid(t_o)
                if p_bbox < 0.2:
                    continue

                # Decode position in image pixel space
                del_x = sigmoid(t_x)
                del_y = sigmoid(t_y)
                x = (j+del_x) * x_ratio
                y = (i+del_y) * y_ratio

                # Decode size in image pixel space
                p_w = anchors[bbox_idx] * x_ratio
                p_h = anchors[bbox_idx+1] * y_ratio
                w = math.exp(t_w) * p_w
                h = math.exp(t_h) * p_h

                # Decode category
                category_idx = np.argmax(logits) + 1

                # Add to decoded bounding box list
                bbox_list.append((x, y, w, h, category_idx))

    bbox_list = [(50, 40, 100, 80, 0), (100, 50, 70, 90, 1), (100, 100, 70, 90, 10)]
    return bbox_list


def randcolors(n=20):
    # Color palette for categories
    color_list = list()
    for i in range(n):
        color_list.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    return color_list


def video_object_detection(in_video_path, out_video_path, proc="cpu"):
    # This function runs the inference for each frame and creates the output video.

    # Mark current time for end-to-end performance check
    start_time = time.time()

    # Open video using open_video_with_opencv.
    vcap, out, (w0, h0, n_frames) = open_video_with_opencv(in_video_path, out_video_path)

    # Check if video is opened. Otherwise, exit.
    if not vcap.isOpened():
        print('video_object_detection: Input video not opened')
        sys.exit()
    if not out.isOpened():
        print('video_object_detection: Output video not opened')
        sys.exit()

    # TODO: Create an instance of the YOLO_V2_TINY class. Pass the dimension of the input, a path to weight file, and which device you will use as arguments.

    # Start the main loop. For each frame of the video, the loop must do the followings:
    # 1. TODO: Do the inference.
    # 2. Run postprocessing using the inference result, accumulate them through the video writer object.
    #    The coordinates from postprocessing are calculated according to resized input; you must adjust
    #    them to fit into the original video.
    # 3. Measure the end-to-end time and the time spent only for inference.
    # 4. TODO: Save the intermediate values for the first layer.
    # Note that your input must be adjusted to fit into the algorithm,
    # including resizing the frame and changing the dimension.

    # Generate a random color list
    color_list = randcolors(20)

    # Main loop
    inference_time = 0
    for t in range(n_frames):
        inference_start_time = time.time()

        # Get an input frame as a (h0, w0, 3) numpy array
        ret, frame = vcap.read()

        # Pre-processing steps: Resize the input image to a (3, 416, 416) array of type float32.
        input_img = resize_input(frame)  # (3, 416, 416)

        # TODO: Do the inference.
        # Input: (3, 416, 416) numpy array
        # Output: (1, 125, 13, 13) numpy array
        output = np.ones((1, 125, 13, 13))

        # Postprocess
        bbox_list = postprocess(output, w0, h0)

        # Layout on
        for x, y, w, h, category_idx in bbox_list:
            assert (0 < x < w0) and (0 < y < h0) and (0 < w < w0) and (0 < h < h0)
            # x, y: box center
            start = (int(x-w/2), int(y-h/2))
            end = (int(x+w/2), int(y+h/2))
            cv2.rectangle(frame, start, end, color_list[category_idx], 1)

        inference_time += (time.time() - inference_start_time)
        # Accumulate final output frame to VideoWriter object
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Main loop terminated after processing %d frames, %d expected" % (t+1, n_frames))
            break

    # Check the inference performance; end-to-end elapsed time and inference time.
    # Check how many frames are processed per second respectively.
    elapsed_time = time.time() - start_time  # End-to-end elapsed time, including overhead
    inference_time /= n_frames  # Average inference (model forward + postprocessing) time taken per frame
    frames_per_second = 1/inference_time
    print("End-to-end elapsed time %f sec, processed %f frame/sec in average" % (elapsed_time, frames_per_second))

    # Release the opened videos.
    vcap.release()
    out.release()
    

def main():
    if len(sys.argv) < 3:
        print ("Usage: python3 __init__.py [in_video.mp4] [out_video.mp4] ([cpu|gpu])")
        sys.exit()

    in_video_path = sys.argv[1] 
    out_video_path = sys.argv[2] 

    if len(sys.argv) == 4:
        proc = sys.argv[3]
    else:
        proc = "cpu"

    video_object_detection(in_video_path, out_video_path, proc)


if __name__ == "__main__":
    main()
