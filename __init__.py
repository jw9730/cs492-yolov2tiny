import sys
import numpy as np
import cv2 as cv2
import time
import yolov2tiny


def open_video_with_opencv(in_video_path, out_video_path):
    #
    # This function takes input and output video path and open them.
    #
    # Your code from here. You may clear the comments.
    #

    # Open an object of input video using cv2.VideoCapture.
    vcap = cv2.VideoCapture(in_video_path)
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height), True)

    # Return the video objects and anything you want for further process.
    return vcap, out, (width, height, n_frames)


def resize_input(im):
    imsz = cv2.resize(im, (416, 416), interpolation=cv2.INTER_CUBIC)
    imsz = imsz / 255.
    imsz = imsz[:, :, ::-1]
    return np.asarray(imsz, dtype=np.float32)


def video_object_detection(in_video_path, out_video_path, proc="cpu"):
    #
    # This function runs the inference for each frame and creates the output video.
    #
    # Your code from here. You may clear the comments.
    #

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

    # Create an instance of the YOLO_V2_TINY class. Pass the dimension of
    # the input, a path to weight file, and which device you will use as arguments.
    model = yolov2tiny.YOLO_V2_TINY(in_shape=(1, 416, 416, 3), weight_pickle="y2t_weights.pickle", proc=proc)

    # Start the main loop. For each frame of the video, the loop must do the followings:
    # 1. Do the inference.
    # 2. Run postprocessing using the inference result, accumulate them through the video writer object.
    #    The coordinates from postprocessing are calculated according to resized input; you must adjust
    #    them to fit into the original video.
    # 3. Measure the end-to-end time and the time spent only for inferencing.
    # 4. Save the intermediate values for the first layer.
    # Note that your input must be adjusted to fit into the algorithm,
    # including resizing the frame and changing the dimension.
    inference_time = 0
    for t in range(n_frames):
        # Get an input frame as a (h0, w0, 3) numpy array
        ret, frame = vcap.read()

        # First-end
        inference_start_time = time.time()

        # Pre-processing steps: Resize the input image to a (1, 416, 416, 3) array of type float32.
        input_img = resize_input(frame)  # (1, 416, 416, 3)

        # Input: (1, 416, 416, 3) numpy array
        # Output: (1, 125, 13, 13) numpy array
        out_tensors = model.inference(input_img)
        output = out_tensors[-1].transpose((0, 3, 1, 2))

        if t == 0:
            for idx, out_tensor in enumerate(out_tensors):
                np.save(file='intermediate/layer_{}'.format(idx), arr=out_tensor)

        # Postprocess
        label_boxes = yolov2tiny.postprocessing(output, w0, h0)

        # Layout on unresized video
        for best_class_name, lefttop, rightbottom, color in label_boxes:
            try:
                cv2.rectangle(frame, lefttop, rightbottom, color, 1)
            except TypeError:
                print("TypeError: Input coordinates {}, {}".format(lefttop, rightbottom))
                continue

            text = best_class_name
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)[0]
            box_coords = ((lefttop[0], rightbottom[1]), (lefttop[0] + text_width + 2, rightbottom[1] - text_height - 2))

            cv2.rectangle(frame, box_coords[0], box_coords[1], color, cv2.FILLED)
            cv2.putText(frame, text, (lefttop[0], rightbottom[1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 255, 255), thickness=1)

        # Accumulate final output frame to VideoWriter object
        out.write(frame)

        inference_time += (time.time() - inference_start_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Main loop terminated after processing %d frames, %d expected" % (t + 1, n_frames))
            break

    # Check the inference peformance; end-to-end elapsed time and inferencing time.
    # Check how many frames are processed per second respectivly.
    elapsed_time = time.time() - start_time  # End-to-end elapsed time, including overhead
    inference_time /= n_frames  # Average inference (model forward + postprocessing) time taken per frame
    frames_per_second = 1 / inference_time
    print("End-to-end elapsed time %f sec, processed %f frame/sec in average" % (elapsed_time, frames_per_second))

    # Release the opened videos.
    vcap.release()
    out.release()


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 __init__.py [in_video.mp4] [out_video.mp4] ([cpu|gpu])")
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
