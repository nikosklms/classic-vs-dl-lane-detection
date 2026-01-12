import cv2
import os

# Settings
video_path = 'inference/videos/1.mp4'
output_folder = 'inference/video_frames_temp'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)
count = 0

print("Extracting frames...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Save frame as JPG
    cv2.imwrite(os.path.join(output_folder, f"frame_{count:06d}.jpg"), frame)
    count += 1

cap.release()
print(f"Done! Extracted {count} frames to {output_folder}")