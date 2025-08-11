import cv2
import os

input_videos_dir = r"C:\Users\DELL\lion_model\normal\normal_videos"      
output_frames_dir = r"C:\Users\DELL\lion_model\normal\normal_frames"     
fps_to_extract = 6                            


os.makedirs(output_frames_dir, exist_ok=True)

# Iterate over all MP4 videos
for filename in os.listdir(input_videos_dir):
    if filename.endswith(".mp4"):
        video_path = os.path.join(input_videos_dir, filename)
        video_name = os.path.splitext(filename)[0]

        # Create a subfolder for this video's frames
        frame_folder = os.path.join(output_frames_dir, video_name)
        os.makedirs(frame_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)      # Original video FPS
        frame_interval = int(video_fps / fps_to_extract)  # Sample every Nth frame

        frame_count = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(frame_folder, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"Saved {saved_count} frames for {video_name}")

print("✅ Frame extraction complete.")
