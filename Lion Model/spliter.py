import os
import json
import shutil
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split
import random
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LionBehaviorDatasetSplitter:
    """
    Create train/validation/test splits for hybrid TimeSformer + Pose Transformer model
    Handles both visual frames and normalized keypoints simultaneously
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Dataset pahs
        self.frames_paths = {
            'normal': 'normal/normal_frames_resized_padded_imagenet_normalized',
            'abnormal': 'abnormal/abnormal_frames_resized_padded_imagenet_normalized'
        }
        
        self.keypoints_path = 'normalized_keypoints'
        
        # Split ratios
        self.train_ratio = 0.70
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        
        # Normalization methods to use (you can modify this)
        self.pose_normalization_methods = ['global', 'per_keypoint', 'temporal']
        
    def scan_dataset(self, base_path: str) -> Dict[str, Dict[str, List[str]]]:
        """
        Scan dataset and create inventory of all videos
        Returns: {category: {data_type: [video_names]}}
        """
        base_path = Path(base_path)
        dataset_inventory = {'normal': {}, 'abnormal': {}}
        
        logger.info("Scanning dataset structure...")
        
        # Scan visual frames
        for category in ['normal', 'abnormal']:
            frames_path = base_path / self.frames_paths[category]
            if frames_path.exists():
                video_folders = [f.name for f in frames_path.iterdir() if f.is_dir()]
                dataset_inventory[category]['frames'] = sorted(video_folders)
                logger.info(f"Found {len(video_folders)} {category} videos with frames")
            else:
                logger.warning(f"Frames path not found: {frames_path}")
                dataset_inventory[category]['frames'] = []
        
        # Scan keypoints for each normalization method
        keypoints_path = base_path / self.keypoints_path
        if keypoints_path.exists():
            for method in self.pose_normalization_methods:
                method_path = keypoints_path / method
                if method_path.exists():
                    for category in ['normal', 'abnormal']:
                        category_path = method_path / category
                        if category_path.exists():
                            video_folders = [f.name for f in category_path.iterdir() if f.is_dir()]
                            if f'keypoints_{method}' not in dataset_inventory[category]:
                                dataset_inventory[category][f'keypoints_{method}'] = []
                            dataset_inventory[category][f'keypoints_{method}'] = sorted(video_folders)
                            logger.info(f"Found {len(video_folders)} {category} videos with {method} keypoints")
        
        return dataset_inventory
    
    def verify_data_consistency(self, dataset_inventory: Dict) -> Dict[str, List[str]]:
        """
        Verify that each video has both frames and keypoints data
        Returns valid videos for each category
        """
        logger.info("Verifying data consistency...")
        
        valid_videos = {'normal': [], 'abnormal': []}
        
        for category in ['normal', 'abnormal']:
            frames_videos = set(dataset_inventory[category].get('frames', []))
            
            # Check consistency across all keypoint normalization methods
            keypoint_videos_intersection = None
            for method in self.pose_normalization_methods:
                key = f'keypoints_{method}'
                if key in dataset_inventory[category]:
                    method_videos = set(dataset_inventory[category][key])
                    if keypoint_videos_intersection is None:
                        keypoint_videos_intersection = method_videos
                    else:
                        keypoint_videos_intersection = keypoint_videos_intersection.intersection(method_videos)
            
            if keypoint_videos_intersection is None:
                logger.error(f"No keypoint data found for {category} category")
                continue
            
            # Find videos that have both frames and all keypoint normalizations
            consistent_videos = frames_videos.intersection(keypoint_videos_intersection)
            valid_videos[category] = sorted(list(consistent_videos))
            
            logger.info(f"{category.title()} category:")
            logger.info(f"  Videos with frames: {len(frames_videos)}")
            logger.info(f"  Videos with keypoints: {len(keypoint_videos_intersection)}")
            logger.info(f"  Consistent videos: {len(consistent_videos)}")
            
            if len(consistent_videos) < len(frames_videos):
                missing_keypoints = frames_videos - keypoint_videos_intersection
                logger.warning(f"  Videos missing keypoints: {missing_keypoints}")
            
            if len(consistent_videos) < len(keypoint_videos_intersection):
                missing_frames = keypoint_videos_intersection - frames_videos
                logger.warning(f"  Videos missing frames: {missing_frames}")
        
        return valid_videos
    
    def analyze_video_characteristics(self, base_path: str, valid_videos: Dict[str, List[str]]) -> Dict:
        """
        Analyze characteristics of videos for informed splitting
        """
        logger.info("Analyzing video characteristics...")
        
        base_path = Path(base_path)
        video_characteristics = {'normal': {}, 'abnormal': {}}
        
        for category in ['normal', 'abnormal']:
            frames_base = base_path / self.frames_paths[category]
            
            for video_name in valid_videos[category]:
                video_path = frames_base / video_name
                
                characteristics = {
                    'name': video_name,
                    'category': category,
                    'num_frames': 0,
                    'has_npy': False,
                    'keypoint_shapes': {}
                }
                
                # Count frames
                if video_path.exists():
                    frame_files = [f for f in video_path.iterdir() 
                                 if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                    characteristics['num_frames'] = len(frame_files)
                    
                    # Check for .npy file
                    npy_files = list(video_path.glob('*.npy'))
                    characteristics['has_npy'] = len(npy_files) > 0
                
                # Get keypoint shapes
                keypoints_base = base_path / self.keypoints_path
                for method in self.pose_normalization_methods:
                    method_path = keypoints_base / method / category / video_name
                    if method_path.exists():
                        # Try to load one keypoint file to get shape
                        for file_path in method_path.glob('*.npy'):
                            try:
                                data = np.load(file_path)
                                characteristics['keypoint_shapes'][method] = data.shape
                                break
                            except Exception as e:
                                logger.warning(f"Could not load {file_path}: {e}")
                
                video_characteristics[category][video_name] = characteristics
        
        return video_characteristics
    
    def create_stratified_split(self, valid_videos: Dict[str, List[str]], 
                              video_characteristics: Dict) -> Dict[str, Dict[str, List[str]]]:
        """
        Create stratified train/val/test splits
        """
        logger.info("Creating stratified splits...")
        
        splits = {
            'train': {'normal': [], 'abnormal': []},
            'val': {'normal': [], 'abnormal': []},
            'test': {'normal': [], 'abnormal': []}
        }
        
        for category in ['normal', 'abnormal']:
            videos = valid_videos[category]
            
            if len(videos) == 0:
                logger.warning(f"No valid videos found for {category} category")
                continue
            
            # For abnormal videos, try to group by condition type for balanced splitting
            if category == 'abnormal':
                # Group videos by condition type (rough heuristic)
                condition_groups = defaultdict(list)
                for video in videos:
                    # Extract condition type from video name
                    if 'Lion_' in video:
                        condition = video.split('_')[1] if len(video.split('_')) > 1 else 'unknown'
                    else:
                        condition = video.split('_')[0] if '_' in video else video
                    condition_groups[condition].append(video)
                
                logger.info(f"Abnormal video condition distribution:")
                for condition, vids in condition_groups.items():
                    logger.info(f"  {condition}: {len(vids)} videos")
                
                # Split each condition group proportionally
                train_videos, val_videos, test_videos = [], [], []
                
                for condition, cond_videos in condition_groups.items():
                    if len(cond_videos) == 1:
                        # Single video goes to train
                        train_videos.extend(cond_videos)
                    elif len(cond_videos) == 2:
                        # Two videos: one to train, one to test
                        train_videos.append(cond_videos[0])
                        test_videos.append(cond_videos[1])
                    else:
                        # Multiple videos: proper split
                        temp_train, temp_test = train_test_split(
                            cond_videos, test_size=0.3, random_state=self.seed
                        )
                        if len(temp_test) >= 2:
                            temp_val, temp_test = train_test_split(
                                temp_test, test_size=0.5, random_state=self.seed
                            )
                        else:
                            temp_val = temp_test[:len(temp_test)//2] if len(temp_test) > 1 else []
                            temp_test = temp_test[len(temp_test)//2:] if len(temp_test) > 1 else temp_test
                        
                        train_videos.extend(temp_train)
                        val_videos.extend(temp_val)
                        test_videos.extend(temp_test)
                
                splits['train'][category] = train_videos
                splits['val'][category] = val_videos
                splits['test'][category] = test_videos
                
            else:  # Normal videos
                # Simple stratified split for normal videos
                train_val, test = train_test_split(
                    videos, test_size=self.test_ratio, random_state=self.seed
                )
                train, val = train_test_split(
                    train_val, test_size=self.val_ratio/(1-self.test_ratio), 
                    random_state=self.seed
                )
                
                splits['train'][category] = train
                splits['val'][category] = val
                splits['test'][category] = test
        
        # Log split statistics
        logger.info("\n" + "="*60)
        logger.info("DATASET SPLIT SUMMARY")
        logger.info("="*60)
        
        total_normal = len(valid_videos['normal'])
        total_abnormal = len(valid_videos['abnormal'])
        total_videos = total_normal + total_abnormal
        
        for split_name in ['train', 'val', 'test']:
            normal_count = len(splits[split_name]['normal'])
            abnormal_count = len(splits[split_name]['abnormal'])
            total_count = normal_count + abnormal_count
            
            logger.info(f"\n{split_name.upper()} SET:")
            logger.info(f"  Normal: {normal_count:2d} ({normal_count/total_normal*100:.1f}% of normal)")
            logger.info(f"  Abnormal: {abnormal_count:2d} ({abnormal_count/total_abnormal*100:.1f}% of abnormal)")
            logger.info(f"  Total: {total_count:2d} ({total_count/total_videos*100:.1f}% of dataset)")
            logger.info(f"  Balance: {normal_count/(total_count or 1)*100:.1f}% normal, {abnormal_count/(total_count or 1)*100:.1f}% abnormal")
        
        return splits
    
    def create_split_directories(self, base_path: str, output_path: str, 
                               splits: Dict[str, Dict[str, List[str]]]):
        """
        Create directory structure for splits and copy/link data
        """
        logger.info("Creating split directories...")
        
        base_path = Path(base_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        for split_name in ['train', 'val', 'test']:
            split_dir = output_path / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Create frames directories
            frames_dir = split_dir / 'frames'
            frames_dir.mkdir(exist_ok=True)
            (frames_dir / 'normal').mkdir(exist_ok=True)
            (frames_dir / 'abnormal').mkdir(exist_ok=True)
            
            # Create keypoints directories for each normalization method
            keypoints_dir = split_dir / 'keypoints'
            keypoints_dir.mkdir(exist_ok=True)
            for method in self.pose_normalization_methods:
                method_dir = keypoints_dir / method
                method_dir.mkdir(exist_ok=True)
                (method_dir / 'normal').mkdir(exist_ok=True)
                (method_dir / 'abnormal').mkdir(exist_ok=True)
        
        # Copy/link data to split directories
        for split_name, categories in splits.items():
            logger.info(f"Processing {split_name} split...")
            
            split_dir = output_path / split_name
            
            for category, videos in categories.items():
                logger.info(f"  Copying {len(videos)} {category} videos...")
                
                for video_name in videos:
                    # Copy frames
                    src_frames = base_path / self.frames_paths[category] / video_name
                    dst_frames = split_dir / 'frames' / category / video_name
                    
                    if src_frames.exists():
                        if dst_frames.exists():
                            shutil.rmtree(dst_frames)
                        shutil.copytree(src_frames, dst_frames)
                    
                    # Copy keypoints for each normalization method
                    for method in self.pose_normalization_methods:
                        src_keypoints = base_path / self.keypoints_path / method / category / video_name
                        dst_keypoints = split_dir / 'keypoints' / method / category / video_name
                        
                        if src_keypoints.exists():
                            if dst_keypoints.exists():
                                shutil.rmtree(dst_keypoints)
                            shutil.copytree(src_keypoints, dst_keypoints)
        
        # Copy normalization statistics
        stats_src = base_path / self.keypoints_path / 'normalization_statistics.json'
        if stats_src.exists():
            for split_name in ['train', 'val', 'test']:
                stats_dst = output_path / split_name / 'keypoints' / 'normalization_statistics.json'
                shutil.copy2(stats_src, stats_dst)
        
        logger.info("Split directories created successfully!")
    
    def save_split_metadata(self, output_path: str, splits: Dict[str, Dict[str, List[str]]], 
                          video_characteristics: Dict, dataset_inventory: Dict):
        """
        Save detailed metadata about the splits
        """
        output_path = Path(output_path)
        
        metadata = {
            'split_ratios': {
                'train': self.train_ratio,
                'val': self.val_ratio,
                'test': self.test_ratio
            },
            'random_seed': self.seed,
            'pose_normalization_methods': self.pose_normalization_methods,
            'splits': splits,
            'video_characteristics': video_characteristics,
            'dataset_inventory': dataset_inventory,
            'split_statistics': {}
        }
        
        # Calculate detailed statistics
        for split_name in ['train', 'val', 'test']:
            split_stats = {
                'total_videos': 0,
                'normal_videos': len(splits[split_name]['normal']),
                'abnormal_videos': len(splits[split_name]['abnormal']),
                'total_frames': 0,
                'abnormal_conditions': {}
            }
            
            split_stats['total_videos'] = split_stats['normal_videos'] + split_stats['abnormal_videos']
            
            # Count frames and analyze abnormal conditions
            for category in ['normal', 'abnormal']:
                for video_name in splits[split_name][category]:
                    if video_name in video_characteristics[category]:
                        split_stats['total_frames'] += video_characteristics[category][video_name]['num_frames']
                    
                    if category == 'abnormal':
                        # Extract condition type
                        if 'Lion_' in video_name:
                            condition = video_name.split('_')[1] if len(video_name.split('_')) > 1 else 'unknown'
                        else:
                            condition = video_name.split('_')[0] if '_' in video_name else video_name
                        
                        split_stats['abnormal_conditions'][condition] = split_stats['abnormal_conditions'].get(condition, 0) + 1
            
            metadata['split_statistics'][split_name] = split_stats
        
        # Save metadata
        with open(output_path / 'dataset_splits_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Split metadata saved to {output_path / 'dataset_splits_metadata.json'}")
    
    def create_dataset_splits(self, base_path: str, output_path: str = 'dataset_splits'):
        """
        Main function to create complete dataset splits
        """
        logger.info("🦁 Starting Lion Behavior Dataset Splitting...")
        logger.info(f"Base path: {base_path}")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Split ratios: Train {self.train_ratio}, Val {self.val_ratio}, Test {self.test_ratio}")
        
        # Step 1: Scan dataset
        dataset_inventory = self.scan_dataset(base_path)
        
        # Step 2: Verify consistency
        valid_videos = self.verify_data_consistency(dataset_inventory)
        
        # Step 3: Analyze characteristics
        video_characteristics = self.analyze_video_characteristics(base_path, valid_videos)
        
        # Step 4: Create splits
        splits = self.create_stratified_split(valid_videos, video_characteristics)
        
        # Step 5: Create directories and copy data
        self.create_split_directories(base_path, output_path, splits)
        
        # Step 6: Save metadata
        self.save_split_metadata(output_path, splits, video_characteristics, dataset_inventory)
        
        logger.info("\n🎯 Dataset splitting completed successfully!")
        logger.info(f"Check results in: {output_path}")
        
        return splits, video_characteristics

# Usage example and configuration
def main():
    """
    Main function to create dataset splits for hybrid TimeSformer + Pose Transformer model
    """
    
    # Configuration
    BASE_PATH = r"C:\Users\DELL\lion_model"  # Adjust this to your actual base path
    OUTPUT_PATH = "dataset_splits"
    RANDOM_SEED = 42
    
    # You can customize which pose normalization methods to include
    # Available: ['global', 'per_keypoint', 'per_video', 'temporal']
    POSE_METHODS = ['global', 'per_keypoint', 'temporal']  # Recommended combination
    
    # Initialize splitter
    splitter = LionBehaviorDatasetSplitter(seed=RANDOM_SEED)
    splitter.pose_normalization_methods = POSE_METHODS
    
    # Create splits
    splits, characteristics = splitter.create_dataset_splits(BASE_PATH, OUTPUT_PATH)
    
    # Print final summary
    print("\n" + "="*80)
    print("🦁 LION BEHAVIOR DATASET SPLITS READY FOR HYBRID MODEL TRAINING")
    print("="*80)
    print(f"📁 Dataset location: {OUTPUT_PATH}")
    print(f"🎯 Model architecture: TimeSformer + Pose-based Transformer")
    print(f"📊 Pose normalization methods: {', '.join(POSE_METHODS)}")
    print()
    print("📋 Usage for training:")
    print("   - Load frames from: splits/[train|val|test]/frames/[normal|abnormal]/")
    print("   - Load keypoints from: splits/[train|val|test]/keypoints/[method]/[normal|abnormal]/")
    print("   - Use normalization_statistics.json for any denormalization needs")
    print("\n✅ Ready for hybrid model training!")

if __name__ == "__main__":
    main()