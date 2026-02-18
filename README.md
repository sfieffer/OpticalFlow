## To use:
1. Place the video file to analyze in the 'input_videos' folder.
2. Edit the top lines of 'Farneback.py' to point to the video file. Adjust sampling rate to desired rate.
3. Run Farneback.py.
4. Output csv and video visualization file will be created in 'Output' folder with similar name to the input file name.
<br>

## CSV output columns:

time_sec - the timestamp in the video of the sample

frame_idx - frame number of the sample

dt_frames - how many frames have passed since the last sample

mean_mag - average motion magnitude across the entire image

median_mag - median value of motion magnitude across the entire image

p90_mag - 90th percentile of motion magnitude (90% of pixels move less than this value)

max_mag - Max (fastest motion) in the frame

mean_u - Average horizontal motion (positive = rightward, negative = leftward)

mean_v - Average vertical motion (positive = downward, negative = upward)

All magnitude values are in pixels/frame units across the sampling time gap defined. If the framerate of the video is 60FPS and is sampled at 4Hz, the sampling will occur every 15 frames.
