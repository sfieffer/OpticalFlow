import cv2
import numpy as np
import pandas as pd

## OLD CODE FROM AMIN

# Load the video
video = cv2.VideoCapture("C:/Users/xboxs/Desktop/videoplayback.mp4")

# Get FPS
fps = video.get(cv2.CAP_PROP_FPS)
print(f"Frames per second: {fps}")

# Read the first frame
success, frame = video.read()

# Set the initial previous frame
prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Initialize lists to store the average optic flow vectors for each frame
fx_list = []
fy_list = []
while True:
    # Read the next frame
    success, frame = video.read()

    # If we reached the end of the video, break the loop
    if not success:
        break

    # Convert the frame to grayscale
    curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute the optic flow vectors using the Farneback/Dense method
    flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3,
    15, 3, 5, 1.2, 0)

    # Draw the optic flow vectors on the frame
    h, w = frame.shape[:2]
    y, x = np.mgrid[0:h:20, 0:w:20].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(prev_frame, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))

    # Display the frame with the optic flow vectors
    cv2.imshow('Optic Flow', vis)
    if cv2.waitKey(1) == 27:
        break

    # Update the previous frame and store the average optic flow vectors for this frame
    prev_frame = curr_frame
    fx_list.append(np.mean(fx))
    fy_list.append(np.mean(fy))

# Release the video capture and close all windows
video.release()
cv2.destroyAllWindows()

# Create a DataFrame with the optic flow vectors data
df = pd.DataFrame({'fx': fx_list, 'fy': fy_list})
df["magnitude"] = np.sqrt(df["fx"]**2 + df["fy"]**2)

# Using the FPS, create the vectors as pixels/sec and magnitude/sec
df['fx_per_sec'] = df['fx'] * fps
df['fy_per_sec'] = df['fy'] * fps
df['magnitude_per_sec'] = df['magnitude'] * fps


# Save the DataFrame to an Excel workbook
df.to_excel('optic_flow_vectors.xlsx', index=False)