import os
import cv2
import numpy as np
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
from typing import Optional, Tuple, List, Dict, Any
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnimalPose3DEstimator(nn.Module):
    """
    Enhanced 3D pose estimator specifically designed for animal (lion) pose detection.
    Uses a more robust architecture with residual connections and attention mechanisms.
    """
    def __init__(self, num_keypoints: int = 20, input_size: int = 256):
        super(AnimalPose3DEstimator, self).__init__()
        self.num_keypoints = num_keypoints
        
        # Enhanced backbone with residual blocks
        self.backbone = nn.Sequential(
            # Initial conv layer
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Residual blocks
            self._make_layer(64, 128, stride=2),
            self._make_layer(128, 256, stride=2),
            self._make_layer(256, 512, stride=2),
            self._make_layer(512, 1024, stride=2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Enhanced pose regression head with dropout and batch norm
        self.pose_head = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(512, num_keypoints * 3)  # x, y, z for each keypoint
        )
        
        # Confidence head for keypoint visibility
        self.confidence_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_keypoints),
            nn.Sigmoid()  # Confidence scores between 0 and 1
        )
        
    def _make_layer(self, in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
        """Create a residual layer"""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride, downsample),
            ResidualBlock(out_channels, out_channels)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Get 3D poses and confidence scores
        poses = self.pose_head(features)
        confidence = self.confidence_head(features)
        
        poses = poses.view(-1, self.num_keypoints, 3)
        
        return poses, confidence

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class LionPoseExtractor:
    """Enhanced Lion Pose Extractor with better error handling and optimizations"""
    
    # Lion-specific keypoints (adapted for quadruped anatomy)
    LION_KEYPOINTS = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'neck', 'left_front_shoulder', 'right_front_shoulder',
        'left_front_elbow', 'right_front_elbow', 'left_front_paw', 'right_front_paw',
        'spine_mid', 'left_back_hip', 'right_back_hip',
        'left_back_knee', 'right_back_knee', 'left_back_paw', 'right_back_paw',
        'tail_base'
    ]
    
    # Skeleton connections for visualization
    SKELETON_CONNECTIONS = [
        # Head
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5),  # nose-eyes-ears-neck
        # Front legs
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 11),
        # Body
        (5, 12), (12, 13), (12, 14), (12, 19),  # neck-spine-hips-tail
        # Back legs
        (13, 15), (14, 16), (15, 17), (16, 18)
    ]
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = AnimalPose3DEstimator(num_keypoints=len(self.LION_KEYPOINTS))
        
        # Load pre-trained weights if available
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                    logger.info(f"Loaded model checkpoint from {model_path}")
                else:
                    self.model.load_state_dict(checkpoint)
                    logger.info(f"Loaded model weights from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {e}")
                logger.warning("Using randomly initialized model")
        else:
            logger.warning("No pre-trained model provided - using random initialization")
            logger.warning("Consider training the model or using pre-trained weights for better results")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Enhanced preprocessing with data augmentation capability
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),  # Ensure exact size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_frame(self, frame_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess a single frame with error handling"""
        try:
            frame = cv2.imread(frame_path)
            if frame is None:
                logger.error(f"Could not load image: {frame_path}")
                return None
            
            # Verify image dimensions
            h, w = frame.shape[:2]
            if h != 256 or w != 256:
                logger.warning(f"Frame {frame_path} is {w}x{h}, expected 256x256")
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            frame_tensor = self.transform(frame).unsqueeze(0).to(self.device)
            return frame_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing frame {frame_path}: {e}")
            return None
    
    def extract_pose_from_frame(self, frame_path: str) -> Optional[Dict[str, Any]]:
        """Extract 3D pose keypoints from a single frame with confidence scores"""
        frame_tensor = self.preprocess_frame(frame_path)
        if frame_tensor is None:
            return None
        
        try:
            with torch.no_grad():
                poses_3d, confidence = self.model(frame_tensor)
                poses_3d = poses_3d.cpu().numpy().squeeze()
                confidence = confidence.cpu().numpy().squeeze()
            
            # Filter low-confidence keypoints
            valid_keypoints = confidence > self.confidence_threshold
            
            return {
                'keypoints_3d': poses_3d,
                'confidence': confidence,
                'valid_keypoints': valid_keypoints,
                'num_valid_keypoints': int(np.sum(valid_keypoints))
            }
            
        except Exception as e:
            logger.error(f"Error extracting pose from {frame_path}: {e}")
            return None
    
    def process_video_folder(self, video_folder_path: str, output_folder: str) -> Optional[Tuple[np.ndarray, List[Dict]]]:
        """Process all frames in a video folder with enhanced error handling"""
        video_path = Path(video_folder_path)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        frame_files = sorted([
            f for f in video_path.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ])
        
        if not frame_files:
            logger.warning(f"No image files found in {video_folder_path}")
            return None
        
        video_poses = []
        video_confidences = []
        frame_info = []
        
        logger.info(f"Processing {len(frame_files)} frames from {video_path.name}")
        
        successful_extractions = 0
        
        for frame_file in tqdm(frame_files, desc=f"Processing {video_path.name}"):
            result = self.extract_pose_from_frame(str(frame_file))
            
            if result is not None:
                video_poses.append(result['keypoints_3d'])
                video_confidences.append(result['confidence'])
                
                frame_info.append({
                    'frame_name': frame_file.name,
                    'frame_path': str(frame_file),
                    'keypoints_3d': result['keypoints_3d'].tolist(),
                    'confidence': result['confidence'].tolist(),
                    'valid_keypoints': result['valid_keypoints'].tolist(),
                    'num_valid_keypoints': result['num_valid_keypoints']
                })
                successful_extractions += 1
            else:
                logger.warning(f"Failed to extract pose from {frame_file.name}")
        
        if successful_extractions == 0:
            logger.error(f"No valid poses extracted from {video_folder_path}")
            return None
        
        logger.info(f"Successfully extracted poses from {successful_extractions}/{len(frame_files)} frames")
        
        # Convert to numpy arrays
        poses_array = np.array(video_poses)
        confidences_array = np.array(video_confidences)
        
        # Save results in multiple formats
        self._save_results(poses_array, confidences_array, frame_info, video_path.name, output_path)
        
        return poses_array, frame_info
    
    def _save_results(self, poses: np.ndarray, confidences: np.ndarray, frame_info: List[Dict], 
                     video_name: str, output_path: Path):
        """Save results in multiple formats"""
        try:
            # Save numpy arrays
            np.save(output_path / f"{video_name}_poses_3d.npy", poses)
            np.save(output_path / f"{video_name}_confidences.npy", confidences)
            
            # Save comprehensive JSON
            json_data = {
                'video_name': video_name,
                'num_frames': len(frame_info),
                'num_keypoints': len(self.LION_KEYPOINTS),
                'keypoint_names': self.LION_KEYPOINTS,
                'confidence_threshold': self.confidence_threshold,
                'frames': frame_info,
                'statistics': {
                    'mean_confidence': float(np.mean(confidences)),
                    'std_confidence': float(np.std(confidences)),
                    'min_confidence': float(np.min(confidences)),
                    'max_confidence': float(np.max(confidences))
                }
            }
            
            with open(output_path / f"{video_name}_poses_3d.json", 'w') as f:
                json.dump(json_data, f, indent=2)
            
            # Save pickle for easy loading
            pickle_data = {
                'poses': poses,
                'confidences': confidences,
                'frame_names': [info['frame_name'] for info in frame_info],
                'keypoint_names': self.LION_KEYPOINTS,
                'skeleton_connections': self.SKELETON_CONNECTIONS
            }
            
            with open(output_path / f"{video_name}_poses_3d.pkl", 'wb') as f:
                pickle.dump(pickle_data, f)
                
            logger.info(f"Saved results for {video_name} to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results for {video_name}: {e}")
    
    def visualize_pose_3d(self, pose_3d: np.ndarray, confidence: np.ndarray, 
                         title: str = "3D Lion Pose", save_path: Optional[str] = None) -> plt.Figure:
        """Enhanced 3D pose visualization with confidence-based coloring"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract coordinates
        x, y, z = pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2]
        
        # Color keypoints based on confidence
        colors = plt.cm.RdYlGn(confidence)  # Red for low confidence, green for high
        
        # Plot keypoints with confidence-based colors and sizes
        scatter = ax.scatter(x, y, z, c=confidence, s=100 * confidence + 50, 
                           alpha=0.8, cmap='RdYlGn', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Confidence', rotation=270, labelpad=15)
        
        # Add keypoint labels for high-confidence points
        for i, (name, conf) in enumerate(zip(self.LION_KEYPOINTS, confidence)):
            if conf > self.confidence_threshold:
                ax.text(x[i], y[i], z[i], name, fontsize=8, alpha=0.7)
        
        # Draw skeleton connections
        for connection in self.SKELETON_CONNECTIONS:
            if (connection[0] < len(pose_3d) and connection[1] < len(pose_3d) and
                confidence[connection[0]] > self.confidence_threshold and 
                confidence[connection[1]] > self.confidence_threshold):
                
                ax.plot([x[connection[0]], x[connection[1]]], 
                       [y[connection[0]], y[connection[1]]], 
                       [z[connection[0]], z[connection[1]]], 
                       'b-', alpha=0.6, linewidth=2)
        
        # Set labels and title
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        ax.set_zlabel('Z (depth)', fontsize=12)
        ax.set_title(f"{title}\n(Confidence threshold: {self.confidence_threshold})", fontsize=14)
        
        # Equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x, mid_y, mid_z = (x.max()+x.min()) * 0.5, (y.max()+y.min()) * 0.5, (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved visualization to {save_path}")
        
        return fig
    
    def process_all_videos(self, main_folder_path: str, output_base_path: str):
        """Process all videos with comprehensive logging and error handling"""
        main_path = Path(main_folder_path)
        output_base = Path(output_base_path)
        
        if not main_path.exists():
            logger.error(f"Main folder does not exist: {main_folder_path}")
            return
        
        # Create output directory
        output_base.mkdir(parents=True, exist_ok=True)
        
        # Process statistics
        total_videos = 0
        successful_videos = 0
        
        # Process both normal and abnormal folders
        for category in ['normal', 'abnormal']:
            category_path = main_path / category
            if not category_path.exists():
                logger.warning(f"Category folder does not exist: {category_path}")
                continue
            
            # Look for the frames folder (e.g., normal_frames_resized_padded, abnormal_frames_resized_padded)
            frames_folder_name = f"{category}_frames_resized_padded"
            frames_folder_path = category_path / frames_folder_name
            
            if not frames_folder_path.exists():
                logger.warning(f"Frames folder does not exist: {frames_folder_path}")
                continue
            
            # Create output folder for this category
            category_output = output_base / category
            category_output.mkdir(exist_ok=True)
            
            # Get all video folders in the frames folder
            video_folders = [f for f in frames_folder_path.iterdir() if f.is_dir()]
            
            if not video_folders:
                logger.warning(f"No video folders found in {frames_folder_path}")
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {len(video_folders)} video folders in '{category}' category")
            logger.info(f"Frames folder: {frames_folder_name}")
            logger.info(f"{'='*60}")
            
            for video_folder in video_folders:
                total_videos += 1
                logger.info(f"\n--- Processing Video {total_videos}: {video_folder.name} ---")
                
                # Create output folder for this video
                video_output = category_output / video_folder.name
                
                # Process the video folder
                result = self.process_video_folder(str(video_folder), str(video_output))
                
                if result is not None:
                    poses, frame_info = result
                    
                    if len(poses) > 0:
                        # Load confidences for visualization
                        confidences = np.load(video_output / f"{video_folder.name}_confidences.npy")
                        
                        # Create visualization for the first frame
                        viz_path = video_output / f"{video_folder.name}_first_frame_pose.png"
                        fig = self.visualize_pose_3d(
                            poses[0], confidences[0],
                            title=f"3D Lion Pose - {video_folder.name} (Frame 1)",
                            save_path=str(viz_path)
                        )
                        plt.close(fig)
                        
                        # Create visualization for middle frame if available
                        if len(poses) > 1:
                            mid_idx = len(poses) // 2
                            viz_path_mid = video_output / f"{video_folder.name}_middle_frame_pose.png"
                            fig = self.visualize_pose_3d(
                                poses[mid_idx], confidences[mid_idx],
                                title=f"3D Lion Pose - {video_folder.name} (Frame {mid_idx+1})",
                                save_path=str(viz_path_mid)
                            )
                            plt.close(fig)
                        
                        successful_videos += 1
                        avg_confidence = np.mean(confidences)
                        logger.info(f"✓ Successfully processed {len(poses)} frames")
                        logger.info(f"  Average confidence: {avg_confidence:.3f}")
                        logger.info(f"  Output saved to: {video_output}")
                    else:
                        logger.error(f"✗ No valid poses extracted from {video_folder.name}")
                else:
                    logger.error(f"✗ Failed to process {video_folder.name}")
        
        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info("PROCESSING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total videos processed: {total_videos}")
        logger.info(f"Successful extractions: {successful_videos}")
        logger.info(f"Success rate: {(successful_videos/total_videos*100) if total_videos > 0 else 0:.1f}%")
        logger.info(f"Results saved to: {output_base}")
        logger.info(f"\nOutput files for each video:")
        logger.info(f"- *_poses_3d.npy: 3D pose coordinates")
        logger.info(f"- *_confidences.npy: Confidence scores")
        logger.info(f"- *_poses_3d.json: Human-readable format with statistics")
        logger.info(f"- *_poses_3d.pkl: Complete data for Python analysis")
        logger.info(f"- *_first_frame_pose.png: First frame visualization")
        logger.info(f"- *_middle_frame_pose.png: Middle frame visualization")

def main():
    """Main execution function with argument validation"""
    # Configuration - UPDATE THESE PATHS
    MAIN_FOLDER_PATH = r"C:\Users\DELL\lion_model"  # Your main project folder
    OUTPUT_FOLDER_PATH = "output/lion_poses_3d"  # Output directory
    MODEL_PATH = None  # Path to pre-trained model weights (optional)
    CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for keypoints
    
    # Validate paths
    if not os.path.exists(MAIN_FOLDER_PATH):
        logger.error(f"Main folder path does not exist: {MAIN_FOLDER_PATH}")
        logger.error("Please update MAIN_FOLDER_PATH in the main() function")
        return
    
    # Initialize the pose extractor
    logger.info("Initializing Lion Pose Extractor...")
    extractor = LionPoseExtractor(model_path=MODEL_PATH, confidence_threshold=CONFIDENCE_THRESHOLD)
    
    # Process all videos
    logger.info("Starting pose extraction process...")
    extractor.process_all_videos(MAIN_FOLDER_PATH, OUTPUT_FOLDER_PATH)
    
    logger.info("\n🦁 Lion Pose Extraction Complete! 🦁")

if __name__ == "__main__":
    main()