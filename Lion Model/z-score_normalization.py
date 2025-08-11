import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KeypointZScoreNormalizer:
    """
    Comprehensive Z-Score normalization for 3D keypoints with multiple strategies:
    1. Global normalization (across all videos and frames)
    2. Per-video normalization
    3. Per-frame normalization
    4. Per-keypoint normalization
    5. Temporal normalization (across time for each video)
    """
    
    def __init__(self):
        self.normalization_stats = {}
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'neck', 'left_front_shoulder', 'right_front_shoulder',
            'left_front_elbow', 'right_front_elbow', 'left_front_paw', 'right_front_paw',
            'spine_mid', 'left_back_hip', 'right_back_hip',
            'left_back_knee', 'right_back_knee', 'left_back_paw', 'right_back_paw',
            'tail_base'
        ]
    
    def load_keypoints_data(self, data_folder: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load all keypoint data from the extracted poses
        Returns: {category: {video_name: poses_array}}
        """
        data_folder = Path(data_folder)
        all_data = {'normal': {}, 'abnormal': {}}
        
        logger.info("Loading keypoint data...")
        
        for category in ['normal', 'abnormal']:
            category_path = data_folder / category
            if not category_path.exists():
                logger.warning(f"Category folder not found: {category_path}")
                continue
            
            video_folders = [f for f in category_path.iterdir() if f.is_dir()]
            logger.info(f"Found {len(video_folders)} videos in {category} category")
            
            for video_folder in video_folders:
                # Try to load poses from pickle file (preferred) or numpy file
                pickle_file = video_folder / f"{video_folder.name}_poses_3d.pkl"
                npy_file = video_folder / f"{video_folder.name}_poses_3d.npy"
                
                if pickle_file.exists():
                    with open(pickle_file, 'rb') as f:
                        data = pickle.load(f)
                        poses = data['poses']
                elif npy_file.exists():
                    poses = np.load(npy_file)
                else:
                    logger.warning(f"No pose data found for {video_folder.name}")
                    continue
                
                all_data[category][video_folder.name] = poses
                logger.info(f"Loaded {poses.shape[0]} frames from {video_folder.name}")
        
        return all_data
    
    def compute_global_statistics(self, all_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Compute global mean and std across all videos and frames
        """
        logger.info("Computing global statistics...")
        
        # Collect all poses
        all_poses = []
        for category in all_data.values():
            for video_poses in category.values():
                all_poses.append(video_poses)
        
        if not all_poses:
            raise ValueError("No pose data found!")
        
        # Concatenate all poses: (total_frames, num_keypoints, 3)
        all_poses_concat = np.concatenate(all_poses, axis=0)
        
        # Flatten to (total_frames * num_keypoints, 3) for coordinate-wise statistics
        all_coords = all_poses_concat.reshape(-1, 3)
        
        global_stats = {
            'mean': np.mean(all_coords, axis=0),  # Mean for [x, y, z]
            'std': np.std(all_coords, axis=0),    # Std for [x, y, z]
            'mean_per_keypoint': np.mean(all_poses_concat, axis=0),  # (num_keypoints, 3)
            'std_per_keypoint': np.std(all_poses_concat, axis=0),    # (num_keypoints, 3)
        }
        
        logger.info(f"Global statistics computed from {all_coords.shape[0]} coordinate points")
        logger.info(f"Global mean: {global_stats['mean']}")
        logger.info(f"Global std: {global_stats['std']}")
        
        return global_stats
    
    def normalize_global(self, poses: np.ndarray, global_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Apply global z-score normalization
        Args:
            poses: (num_frames, num_keypoints, 3)
            global_stats: Dictionary with global mean and std
        Returns:
            normalized_poses: (num_frames, num_keypoints, 3)
        """
        mean = global_stats['mean']
        std = global_stats['std']
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        # Apply normalization
        normalized_poses = (poses - mean) / std
        
        return normalized_poses
    
    def normalize_per_keypoint(self, poses: np.ndarray, global_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Apply per-keypoint z-score normalization
        Args:
            poses: (num_frames, num_keypoints, 3)
        Returns:
            normalized_poses: (num_frames, num_keypoints, 3)
        """
        mean_per_keypoint = global_stats['mean_per_keypoint']
        std_per_keypoint = global_stats['std_per_keypoint']
        
        # Avoid division by zero
        std_per_keypoint = np.where(std_per_keypoint == 0, 1, std_per_keypoint)
        
        # Apply normalization per keypoint
        normalized_poses = (poses - mean_per_keypoint) / std_per_keypoint
        
        return normalized_poses
    
    def normalize_per_video(self, poses: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Apply per-video z-score normalization
        Args:
            poses: (num_frames, num_keypoints, 3)
        Returns:
            normalized_poses: (num_frames, num_keypoints, 3)
            video_stats: Dictionary with video-specific mean and std
        """
        # Flatten for coordinate-wise statistics
        coords = poses.reshape(-1, 3)
        
        mean = np.mean(coords, axis=0)
        std = np.std(coords, axis=0)
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        normalized_poses = (poses - mean) / std
        
        video_stats = {'mean': mean, 'std': std}
        
        return normalized_poses, video_stats
    
    def normalize_per_frame(self, poses: np.ndarray) -> np.ndarray:
        """
        Apply per-frame z-score normalization
        Args:
            poses: (num_frames, num_keypoints, 3)
        Returns:
            normalized_poses: (num_frames, num_keypoints, 3)
        """
        normalized_poses = np.zeros_like(poses)
        
        for i in range(poses.shape[0]):
            frame = poses[i]  # (num_keypoints, 3)
            coords = frame.reshape(-1, 3)  # (num_keypoints, 3)
            
            mean = np.mean(coords, axis=0)
            std = np.std(coords, axis=0)
            
            # Avoid division by zero
            std = np.where(std == 0, 1, std)
            
            normalized_poses[i] = (frame - mean) / std
        
        return normalized_poses
    
    def normalize_temporal(self, poses: np.ndarray) -> np.ndarray:
        """
        Apply temporal z-score normalization (across time for each keypoint)
        Args:
            poses: (num_frames, num_keypoints, 3)
        Returns:
            normalized_poses: (num_frames, num_keypoints, 3)
        """
        normalized_poses = np.zeros_like(poses)
        
        # For each keypoint and coordinate
        for kp in range(poses.shape[1]):  # keypoints
            for coord in range(poses.shape[2]):  # x, y, z
                trajectory = poses[:, kp, coord]  # (num_frames,)
                
                mean = np.mean(trajectory)
                std = np.std(trajectory)
                
                # Avoid division by zero
                if std == 0:
                    normalized_poses[:, kp, coord] = trajectory - mean
                else:
                    normalized_poses[:, kp, coord] = (trajectory - mean) / std
        
        return normalized_poses
    
    def _convert_numpy_to_serializable(self, obj):
        """
        Recursively convert numpy arrays and other non-serializable objects to JSON-compatible format
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_to_serializable(item) for item in obj)
        else:
            return obj
    
    def normalize_dataset(self, data_folder: str, output_folder: str, 
                         methods: List[str] = ['global', 'per_keypoint', 'per_video', 'temporal']):
        """
        Normalize entire dataset with multiple methods
        Args:
            data_folder: Folder containing extracted poses
            output_folder: Folder to save normalized data
            methods: List of normalization methods to apply
        """
        # Load all data
        all_data = self.load_keypoints_data(data_folder)
        
        # Compute global statistics
        global_stats = self.compute_global_statistics(all_data)
        self.normalization_stats['global'] = global_stats
        
        # Create output directory
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each method
        for method in methods:
            logger.info(f"\n{'='*50}")
            logger.info(f"Applying {method} normalization")
            logger.info(f"{'='*50}")
            
            method_output = output_path / method
            method_output.mkdir(exist_ok=True)
            
            method_stats = {}
            
            # Process each category and video
            for category, videos in all_data.items():
                category_output = method_output / category
                category_output.mkdir(exist_ok=True)
                
                for video_name, poses in videos.items():
                    logger.info(f"Normalizing {category}/{video_name} with {method} method")
                    
                    # Apply normalization based on method
                    if method == 'global':
                        normalized_poses = self.normalize_global(poses, global_stats)
                        video_stats = None
                    
                    elif method == 'per_keypoint':
                        normalized_poses = self.normalize_per_keypoint(poses, global_stats)
                        video_stats = None
                    
                    elif method == 'per_video':
                        normalized_poses, video_stats = self.normalize_per_video(poses)
                    
                    elif method == 'per_frame':
                        normalized_poses = self.normalize_per_frame(poses)
                        video_stats = None
                    
                    elif method == 'temporal':
                        normalized_poses = self.normalize_temporal(poses)
                        video_stats = None
                    
                    else:
                        logger.error(f"Unknown normalization method: {method}")
                        continue
                    
                    # Save normalized data
                    video_output = category_output / video_name
                    video_output.mkdir(exist_ok=True)
                    
                    # Save as numpy array
                    np.save(video_output / f"{video_name}_normalized_{method}.npy", normalized_poses)
                    
                    # Save as pickle with metadata
                    pickle_data = {
                        'normalized_poses': normalized_poses,
                        'original_shape': poses.shape,
                        'normalization_method': method,
                        'keypoint_names': self.keypoint_names,
                        'video_stats': video_stats
                    }
                    
                    with open(video_output / f"{video_name}_normalized_{method}.pkl", 'wb') as f:
                        pickle.dump(pickle_data, f)
                    
                    # Store stats for summary
                    if video_stats:
                        if method not in method_stats:
                            method_stats[method] = {}
                        method_stats[method][f"{category}_{video_name}"] = video_stats
            
            # Save method-specific statistics
            if method_stats and method in method_stats:
                self.normalization_stats[method] = method_stats[method]
        
        # Save all normalization statistics
        try:
            with open(output_path / 'normalization_statistics.json', 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                stats_serializable = self._convert_numpy_to_serializable(self.normalization_stats)
                json.dump(stats_serializable, f, indent=2)
            logger.info("Normalization statistics saved successfully")
        except Exception as e:
            logger.error(f"Failed to save normalization statistics: {e}")
            # Save as pickle as fallback
            with open(output_path / 'normalization_statistics.pkl', 'wb') as f:
                pickle.dump(self.normalization_stats, f)
            logger.info("Saved normalization statistics as pickle file instead")
        
        logger.info(f"\nNormalization complete! Results saved to: {output_folder}")
    
    def visualize_normalization_effects(self, original_poses: np.ndarray, 
                                      normalized_poses_dict: Dict[str, np.ndarray],
                                      video_name: str, save_path: Optional[str] = None):
        """
        Visualize the effects of different normalization methods
        """
        methods = list(normalized_poses_dict.keys())
        n_methods = len(methods) + 1  # +1 for original
        
        # Create subplots
        fig = plt.figure(figsize=(20, 4 * n_methods))
        
        # Plot original
        ax = fig.add_subplot(n_methods, 1, 1, projection='3d')
        self._plot_pose_trajectory(ax, original_poses, f"Original - {video_name}")
        
        # Plot normalized versions
        for i, (method, poses) in enumerate(normalized_poses_dict.items()):
            ax = fig.add_subplot(n_methods, 1, i + 2, projection='3d')
            self._plot_pose_trajectory(ax, poses, f"{method.title()} Normalized - {video_name}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        return fig
    
    def _plot_pose_trajectory(self, ax, poses: np.ndarray, title: str):
        """Helper function to plot 3D pose trajectory"""
        # Plot first frame keypoints
        first_frame = poses[0]
        ax.scatter(first_frame[:, 0], first_frame[:, 1], first_frame[:, 2], 
                  c='red', s=50, alpha=0.7, label='First frame')
        
        # Plot trajectory of nose keypoint (index 0) across all frames
        nose_trajectory = poses[:, 0, :]  # (num_frames, 3)
        ax.plot(nose_trajectory[:, 0], nose_trajectory[:, 1], nose_trajectory[:, 2], 
               'b-', alpha=0.6, linewidth=2, label='Nose trajectory')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()
    
    def analyze_normalization_quality(self, output_folder: str):
        """
        Analyze the quality of different normalization methods
        """
        output_path = Path(output_folder)
        
        # Load normalization statistics
        with open(output_path / 'normalization_statistics.json', 'r') as f:
            stats = json.load(f)
        
        logger.info("\n" + "="*60)
        logger.info("NORMALIZATION QUALITY ANALYSIS")
        logger.info("="*60)
        
        # Analyze global statistics
        global_stats = stats['global']
        logger.info(f"\nGlobal Statistics:")
        logger.info(f"  Mean: {global_stats['mean']}")
        logger.info(f"  Std:  {global_stats['std']}")
        
        # Check if std is reasonable (not too small or too large)
        std_array = np.array(global_stats['std'])
        if np.any(std_array < 0.1):
            logger.warning("Some standard deviations are very small - may indicate limited motion")
        if np.any(std_array > 100):
            logger.warning("Some standard deviations are very large - may indicate outliers")
        
        # Analyze per-keypoint variations
        per_kp_std = np.array(global_stats['std_per_keypoint'])
        avg_std_per_keypoint = np.mean(per_kp_std, axis=1)  # Average std across x,y,z for each keypoint
        
        logger.info(f"\nKeypoint Motion Analysis (Average std across coordinates):")
        for i, (kp_name, std_val) in enumerate(zip(self.keypoint_names, avg_std_per_keypoint)):
            logger.info(f"  {kp_name:20}: {std_val:.3f}")
        
        # Identify most/least mobile keypoints
        most_mobile_idx = np.argmax(avg_std_per_keypoint)
        least_mobile_idx = np.argmin(avg_std_per_keypoint)
        
        logger.info(f"\nMost mobile keypoint:  {self.keypoint_names[most_mobile_idx]} (std: {avg_std_per_keypoint[most_mobile_idx]:.3f})")
        logger.info(f"Least mobile keypoint: {self.keypoint_names[least_mobile_idx]} (std: {avg_std_per_keypoint[least_mobile_idx]:.3f})")

def main():
    """Main function to demonstrate z-score normalization"""
    
    # Configuration
    DATA_FOLDER = r"C:\Users\DELL\lion_model\output\lion_poses_3d" # Folder with extracted poses
    OUTPUT_FOLDER = "output/normalized_keypoints"  # Output folder for normalized data
    
    # Normalization methods to apply
    METHODS = ['global', 'per_keypoint', 'per_video', 'temporal']
    
    # Initialize normalizer
    normalizer = KeypointZScoreNormalizer()
    
    # Perform normalization
    logger.info("Starting keypoint normalization process...")
    normalizer.normalize_dataset(DATA_FOLDER, OUTPUT_FOLDER, METHODS)
    
    # Analyze normalization quality
    normalizer.analyze_normalization_quality(OUTPUT_FOLDER)
    
    logger.info("\n🎯 Keypoint normalization complete!")
    logger.info(f"Check results in: {OUTPUT_FOLDER}")

# Example usage for loading normalized data
def load_normalized_data_example():
    """Example of how to load and use normalized keypoint data"""
    
    # Load normalized data
    video_path = "output/normalized_keypoints/global/normal/video_name"
    
    # Method 1: Load numpy array
    normalized_poses = np.load(f"{video_path}/video_name_normalized_global.npy")
    print(f"Normalized poses shape: {normalized_poses.shape}")
    
    # Method 2: Load pickle file with metadata
    with open(f"{video_path}/video_name_normalized_global.pkl", 'rb') as f:
        data = pickle.load(f)
        poses = data['normalized_poses']
        method = data['normalization_method']
        keypoints = data['keypoint_names']
        
    print(f"Normalization method: {method}")
    print(f"Number of keypoints: {len(keypoints)}")
    
    # Check normalization (should have mean ≈ 0, std ≈ 1 for global method)
    coords = poses.reshape(-1, 3)
    print(f"Normalized data mean: {np.mean(coords, axis=0)}")
    print(f"Normalized data std:  {np.std(coords, axis=0)}")

if __name__ == "__main__":
    main()