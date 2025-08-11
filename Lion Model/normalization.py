import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import torch
import torchvision.transforms as transforms

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])  # RGB
IMAGENET_STD = np.array([0.229, 0.224, 0.225])   # RGB

def normalize_imagenet_single_image(image_path, save_path=None):
    """
    Normalize a single image using ImageNet mean and std
    
    Args:
        image_path (str): Path to the input image
        save_path (str): Path to save normalized image (optional)
    
    Returns:
        numpy.ndarray: ImageNet normalized image array
    """
    
    # Load image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to float32 and normalize to [0, 1] first
    img_float = img_rgb.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization: (x - mean) / std
    normalized_img = (img_float - IMAGENET_MEAN) / IMAGENET_STD
    
    # Save normalized image if path provided (as numpy array)
    if save_path:
        # Save as .npy for exact values
        np.save(save_path.replace('.jpg', '.npy').replace('.png', '.npy'), normalized_img)
        
        # For visualization, convert back to [0,1] range
        vis_img = (normalized_img * IMAGENET_STD) + IMAGENET_MEAN
        vis_img = np.clip(vis_img, 0, 1)  # Ensure [0,1] range
        vis_img_uint8 = (vis_img * 255).astype(np.uint8)
        
        img_pil = Image.fromarray(vis_img_uint8)
        img_pil.save(save_path)
        print(f"ImageNet normalized image saved to: {save_path}")
    
    return normalized_img

def normalize_imagenet_folder(input_folder, output_folder=None):
    """
    Normalize all images in a folder using ImageNet mean and std
    
    Args:
        input_folder (str): Path to folder containing images
        output_folder (str): Path to save normalized images
    """
    
    if output_folder is None:
        output_folder = str(Path(input_folder).parent / f"{Path(input_folder).name}_imagenet_normalized")
    
    os.makedirs(output_folder, exist_ok=True)
    
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    total_processed = 0
    
    print(f"Using ImageNet normalization:")
    print(f"Mean: {IMAGENET_MEAN}")
    print(f"Std:  {IMAGENET_STD}")
    print("-" * 50)
    
    for root, dirs, files in os.walk(input_folder):
        rel_path = os.path.relpath(root, input_folder)
        
        if rel_path != '.':
            output_subdir = os.path.join(output_folder, rel_path)
        else:
            output_subdir = output_folder
        
        os.makedirs(output_subdir, exist_ok=True)
        
        for file in files:
            file_ext = Path(file).suffix.lower()
            
            if file_ext in supported_extensions:
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_subdir, file)
                
                try:
                    # Normalize the image with ImageNet stats
                    normalized_img = normalize_imagenet_single_image(input_path, output_path)
                    
                    total_processed += 1
                    
                    if total_processed % 50 == 0:
                        print(f"Processed {total_processed} images...")
                    
                except Exception as e:
                    print(f"Error processing {input_path}: {str(e)}")
    
    print(f"\nTotal images processed: {total_processed}")
    print(f"ImageNet normalized images saved to: {output_folder}")
    print(f"Both .npy (exact values) and .jpg/.png (visualization) files created")

def normalize_imagenet_numpy_array(img_array):
    """
    Normalize a numpy array using ImageNet mean and std
    
    Args:
        img_array (numpy.ndarray): Input image array (H, W, C) in [0, 255] or [0, 1]
    
    Returns:
        numpy.ndarray: ImageNet normalized array
    """
    
    # Convert to float32
    img_float = img_array.astype(np.float32)
    
    # Check if input is in [0, 255] range and normalize to [0, 1]
    if np.max(img_float) > 1.0:
        img_float = img_float / 255.0
    
    # Apply ImageNet normalization
    normalized = (img_float - IMAGENET_MEAN) / IMAGENET_STD
    
    return normalized

def normalize_imagenet_batch_numpy(batch_array):
    """
    Normalize a batch of images using ImageNet mean and std
    
    Args:
        batch_array (numpy.ndarray): Batch of images (N, H, W, C)
    
    Returns:
        numpy.ndarray: ImageNet normalized batch
    """
    
    batch_float = batch_array.astype(np.float32)
    
    # Normalize to [0, 1] if needed
    if np.max(batch_float) > 1.0:
        batch_float = batch_float / 255.0
    
    # Apply ImageNet normalization to each image
    normalized_batch = (batch_float - IMAGENET_MEAN) / IMAGENET_STD
    
    return normalized_batch

def create_pytorch_imagenet_transform():
    """
    Create PyTorch transform with ImageNet normalization
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL to tensor and scales to [0,1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet RGB mean
            std=[0.229, 0.224, 0.225]    # ImageNet RGB std
        )
    ])
    
    return transform

def create_tensorflow_imagenet_normalizer():
    """
    Create TensorFlow function for ImageNet normalization
    
    Returns:
        function: TensorFlow normalization function
    """
    
    def normalize_tf(image_tensor):
        # Convert to float32 and scale to [0,1]
        image_float = tf.cast(image_tensor, tf.float32) / 255.0
        
        # ImageNet normalization
        mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
        std = tf.constant(IMAGENET_STD, dtype=tf.float32)
        
        normalized = (image_float - mean) / std
        return normalized
    
    return normalize_tf

def denormalize_imagenet(normalized_img):
    """
    Reverse ImageNet normalization to get back original image
    
    Args:
        normalized_img (numpy.ndarray): ImageNet normalized image
    
    Returns:
        numpy.ndarray: Denormalized image in [0, 1] range
    """
    
    # Reverse normalization: x = (normalized * std) + mean
    denormalized = (normalized_img * IMAGENET_STD) + IMAGENET_MEAN
    
    # Clip to valid range
    denormalized = np.clip(denormalized, 0, 1)
    
    return denormalized

def visualize_imagenet_normalization(original_img, normalized_img, save_path=None):
    """
    Visualize ImageNet normalization effects
    
    Args:
        original_img (numpy.ndarray): Original image [0, 255]
        normalized_img (numpy.ndarray): ImageNet normalized image
        save_path (str): Path to save comparison plot
    """
    
    # Denormalize for visualization
    denormalized = denormalize_imagenet(normalized_img)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original_img.astype(np.uint8))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # ImageNet normalized (reconstructed)
    axes[0, 1].imshow(denormalized)
    axes[0, 1].set_title('After ImageNet Normalization\n(Reconstructed)')
    axes[0, 1].axis('off')
    
    # Difference
    if original_img.dtype == np.uint8:
        orig_float = original_img.astype(np.float32) / 255.0
    else:
        orig_float = original_img
    
    diff = np.abs(orig_float - denormalized)
    axes[0, 2].imshow(diff)
    axes[0, 2].set_title('Difference Map')
    axes[0, 2].axis('off')
    
    # Histograms for each channel
    colors = ['red', 'green', 'blue']
    for i in range(3):
        axes[1, i].hist(normalized_img[:, :, i].flatten(), bins=50, alpha=0.7, color=colors[i])
        axes[1, i].set_title(f'Normalized {colors[i].upper()} Channel')
        axes[1, i].set_xlabel('Normalized Value')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].axvline(0, color='black', linestyle='--', alpha=0.5, label='Zero')
        axes[1, i].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def load_normalized_batch(folder_path, batch_size=32):
    """
    Load a batch of ImageNet normalized images
    
    Args:
        folder_path (str): Path to folder with .npy normalized files
        batch_size (int): Size of batch to load
    
    Returns:
        numpy.ndarray: Batch of normalized images
    """
    
    npy_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    
    if not npy_files:
        raise ValueError(f"No .npy files found in {folder_path}")
    
    # Select random batch
    import random
    selected_files = random.sample(npy_files, min(batch_size, len(npy_files)))
    
    # Load images
    batch = []
    for file_path in selected_files:
        img = np.load(file_path)
        batch.append(img)
    
    return np.array(batch)

# Example usage and testing
if __name__ == "__main__":
    print("ImageNet Mean/Std Normalizer for Lion Frames")
    print("=" * 60)
    print(f"ImageNet RGB Mean: {IMAGENET_MEAN}")
    print(f"ImageNet RGB Std:  {IMAGENET_STD}")
    print("=" * 60)
    
    # Lion dataset path
    lion_folder = r"C:\Users\DELL\lion_model\abnormal\abnormal_frames_resized_padded"
    
    print(f"\n1. Processing Lion Dataset")
    if os.path.exists(lion_folder):
        print(f"✅ Found lion dataset: {lion_folder}")
        
        proceed = input("Start ImageNet normalization? (y/n) [default: y]: ").strip().lower()
        if proceed in ['', 'y', 'yes']:
            print("\n🚀 Starting ImageNet normalization...")
            normalize_imagenet_folder(lion_folder)
        else:
            print("Normalization skipped.")
    else:
        print(f"❌ Lion dataset not found: {lion_folder}")
        
        # Manual input
        manual_path = input("\nEnter folder path manually: ").strip()
        if manual_path:
            manual_path = manual_path.replace('"', '').replace("'", "").strip()
            if os.path.exists(manual_path):
                normalize_imagenet_folder(manual_path)
            else:
                print(f"❌ Path not found: {manual_path}")
    
    # Demo with sample array
    print(f"\n2. Sample Array Demo")
    sample_img = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    normalized_sample = normalize_imagenet_numpy_array(sample_img)
    
    print(f"✅ Sample image normalized:")
    print(f"   Original range: [0, 255]")
    print(f"   Normalized range: [{np.min(normalized_sample):.3f}, {np.max(normalized_sample):.3f}]")
    print(f"   Channel means: {np.mean(normalized_sample, axis=(0,1))}")
    print(f"   Channel stds:  {np.std(normalized_sample, axis=(0,1))}")
    
    # PyTorch transform demo
    print(f"\n3. PyTorch Transform Created")
    pytorch_transform = create_pytorch_imagenet_transform()
    print("✅ PyTorch ImageNet transform ready for use")
    
    # TensorFlow normalizer demo
    print(f"\n4. TensorFlow Normalizer Created")
    tf_normalizer = create_tensorflow_imagenet_normalizer()
    print("✅ TensorFlow ImageNet normalizer ready for use")
    
    print("\n" + "=" * 60)
    print("🎉 ImageNet normalization setup complete!")
    print("\nFiles created:")
    print("- .npy files: Exact normalized values for ML models")
    print("- .jpg/.png files: Visual representations")
    print("\nNext steps:")
    print("- Load .npy files for training")
    print("- Use PyTorch/TensorFlow transforms for data pipelines")
    input("\nPress Enter to exit...")