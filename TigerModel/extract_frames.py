import os
import cv2
from collections import Counter
from sklearn.model_selection import train_test_split

def get_all_videos_and_labels(dataset_path):
    categories = ['Tiger_Abnormal', 'Tiger_Normal']
    videos = []
    labels = []

    for category in categories:
        category_path = os.path.join(dataset_path, category)

        if category == 'Tiger_Normal':
            subfolders = ['']
        else:
            subfolders = os.listdir(category_path)

        for subfolder in subfolders:
            if subfolder == '':
                class_path = category_path
                class_label = 'Tiger_Normal'
            else:
                class_path = os.path.join(category_path, subfolder)
                class_label = subfolder

            if not os.path.isdir(class_path):
                continue

            for file_name in os.listdir(class_path):
                if file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(class_path, file_name)
                    videos.append(video_path)
                    labels.append(class_label)
    return videos, labels

def get_dynamic_fps(duration, label):
    if 'normal' in label.lower():
        if duration <= 8:
            return 4
        elif duration <= 20:
            return 4
        else:
            return 5
    else:
        if duration <= 8:
            return 8
        elif duration <= 20:
            return 3
        else:
            return 3

def extract_frames(video_path, output_dir, label, max_frames=220):
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        print(f"⚠️ Skipping {video_path} (FPS not detected)")
        cap.release()
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps
    desired_fps = get_dynamic_fps(duration, label)

    frame_interval = max(int(original_fps / desired_fps), 1)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = f"{video_name}_frame_{saved_count}.jpg"
            frame_filepath = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_filepath, frame)
            saved_count += 1

            if saved_count >= max_frames:
                break

        frame_count += 1

    cap.release()
    return saved_count

def filter_to_common_classes(train_videos, train_labels, test_videos, test_labels):
    train_classes = set(train_labels)
    test_classes = set(test_labels)
    common_classes = train_classes.intersection(test_classes)

    def filter_sets(videos, labels, common_cls):
        filtered_videos = []
        filtered_labels = []
        for v, l in zip(videos, labels):
            if l in common_cls:
                filtered_videos.append(v)
                filtered_labels.append(l)
        return filtered_videos, filtered_labels

    train_videos_filtered, train_labels_filtered = filter_sets(train_videos, train_labels, common_classes)
    test_videos_filtered, test_labels_filtered = filter_sets(test_videos, test_labels, common_classes)

    return train_videos_filtered, train_labels_filtered, test_videos_filtered, test_labels_filtered

def main():
    dataset_path = "Dataset"       
    output_path = "frames_dataset" 
    max_frames_per_video = 200
    test_size = 0.2  
    print("📥 Scanning dataset...")
    videos, labels = get_all_videos_and_labels(dataset_path)

    label_counts = Counter(labels)
    print("\n📊 Video count per class:")
    for label, count in label_counts.items():
        print(f"   {label}: {count} videos")

    too_small = [label for label, count in label_counts.items() if count < 3]
    if too_small:
        raise ValueError(f"❌ Classes with fewer than 3 videos detected: {too_small}\n"
                         "You need at least 3 videos per class for stratified splitting.")

    print("\n🔀 Performing stratified train/test split...")
    train_videos, test_videos, train_labels, test_labels = train_test_split(
        videos, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # Filter both sets to hae the same class st (common classes only)
    train_videos, train_labels, test_videos, test_labels = filter_to_common_classes(
        train_videos, train_labels, test_videos, test_labels
    )

    print(f"Classes in train: {set(train_labels)}")
    print(f"Classes in test: {set(test_labels)}")

    print(f"\n   Training videos after filtering: {len(train_videos)}")
    print(f"   Testing videos after filtering: {len(test_videos)}")

    print("\n🎥 Extracting frames from training set...")
    for video_path, label in zip(train_videos, train_labels):
        if label == 'Tiger_Normal':
            output_dir = os.path.join(output_path, "train", label, os.path.splitext(os.path.basename(video_path))[0])
        else:
            output_dir = os.path.join(output_path, "train", "Tiger_Abnormal", label, os.path.splitext(os.path.basename(video_path))[0])

        num_frames = extract_frames(video_path, output_dir, label, max_frames_per_video)
        print(f"      ✔ {video_path} → {num_frames} frames")

    print("\n🎥 Extracting frames from testing set...")
    for video_path, label in zip(test_videos, test_labels):
        if label == 'Tiger_Normal':
            output_dir = os.path.join(output_path, "test", label, os.path.splitext(os.path.basename(video_path))[0])
        else:
            output_dir = os.path.join(output_path, "test", "Tiger_Abnormal", label, os.path.splitext(os.path.basename(video_path))[0])

        num_frames = extract_frames(video_path, output_dir, label, max_frames_per_video)
        print(f"      ✔ {video_path} → {num_frames} frames")

    print("\n✅ Frame extraction complete!")

if __name__ == "__main__":
    main()
