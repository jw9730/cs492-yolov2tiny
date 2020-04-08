import sys
import numpy as np
import cv2 as cv2
import time
import yolov2tiny


def open_video_with_opencv(in_video_path, out_video_path):

    # Open an object of input video using cv2.VideoCapture.
    vcap = cv2.VideoCapture('sample.mp4')

    # Get video properties
    if vcap.isOpened():
        width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = vcap.get(cv2.CAP_PROP_FPS)  # frames per second
        n_frames = vcap.get(cv2.CAP_PROP_FRAME_COUNT)  # number of frames
        codec = vcap.get(cv2.CAP_PROP_FOURCC)  # 4-character codec code
        print(f"Input video w: {width}, h: {height}, fps: {fps}frames/s, total {n_frames} frames")
    else:
        raise RuntimeError("open_video_with_opencv: Could not open input video")

    # Open an object of output video using cv2.VideoWriter.
    # Same encoding, size, and fps
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.ViewoWriter('output.mp4', fourcc, fps, (width, height))

    # Return the video objects and anything you want for further process.
    return vcap, out, (width, height, n_frames)


def resize_input(im):
    imsz = cv2.resize(im, (416, 416))
    imsz = imsz / 255.
    imsz = imsz[:,:,::-1]
    return np.asarray(imsz, dtype=np.float32)


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

    # TODO: Create an instance of the YOLO_V2_TINY class. Pass the dimension of
    # the input, a path to weight file, and which device you will use as arguments.

    # Start the main loop. For each frame of the video, the loop must do the followings:
    # 1. Do the inference.
    # 2. TODO: Run postprocessing using the inference result, accumulate them through the video writer object.
    #    The coordinates from postprocessing are calculated according to resized input; you must adjust
    #    them to fit into the original video.
    # 3. Measure the end-to-end time and the time spent only for inference.
    # 4. TODO: Save the intermediate values for the first layer.
    # TODO: Note that your input must be adjusted to fit into the algorithm,
    # including resizing the frame and changing the dimension.

    # Main loop
    inference_time = 0
    for t in range(n_frames):
        inference_start_time = time.time()

        # Get an input frame
        ret, in_frame = vcap.read()

        # TODO: Resize input frame to fit YOLOv2_tiny input layer

        # TODO: Do the inference.

        # TODO: Run postprocessing and get an output frame (h0 x w0 x 3 numpy array)
        out_frame = in_frame

        # TODO: Adjust bbox x, y, w, h and write on resized output frame
        x, y, w, h = (20, 40, 60, 80)
        cv2.rectangle(out_frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

        inference_time += (time.time() - inference_start_time)
        # Accumulate final output frame to VideoWriter object
        out.write(out_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"Main loop terminated after processing {t+1} frames, {n_frames} expected")
            break

    # Check the inference performance; end-to-end elapsed time and inference time.
    # Check how many frames are processed per second respectively.
    elapsed_time = time.time() - start_time  # End-to-end elapsed time, including overhead
    inference_time /= n_frames  # Average inference (model forward + postprocessing) time taken per frame
    frames_per_second = 1/inference_time
    print(f"End-to-end elapsed time {elapsed_time} sec, processed {frames_per_second} frame/sec in average")

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
