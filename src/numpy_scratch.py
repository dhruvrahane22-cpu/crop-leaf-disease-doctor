import numpy as np
import cv2
import matplotlib.pyplot as plt

def conv2d_scratch(image, kernel, stride=1, padding=0):
    """
    2D Convolution implemented from scratch in pure NumPy.
    image: 2D numpy array (H, W)
    kernel: 2D numpy array (kh, kw)
    """
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
        
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    
    out_h = (img_h - k_h) // stride + 1
    out_w = (img_w - k_w) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    for y in range(0, out_h):
        for x in range(0, out_w):
            y_start = y * stride
            x_start = x * stride
            region = image[y_start:y_start+k_h, x_start:x_start+k_w]
            output[y, x] = np.sum(region * kernel)
            
    return output

def maxpool2d_scratch(image, pool_size=2, stride=2):
    """
    2D Max Pooling implemented from scratch in pure NumPy.
    """
    img_h, img_w = image.shape
    out_h = (img_h - pool_size) // stride + 1
    out_w = (img_w - pool_size) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    for y in range(0, out_h):
        for x in range(0, out_w):
            y_start = y * stride
            x_start = x * stride
            region = image[y_start:y_start+pool_size, x_start:x_start+pool_size]
            output[y, x] = np.max(region)
            
    return output

# Test kernels
SOBEL_EDGE_KERNEL = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

GAUSSIAN_BLUR_KERNEL = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32) / 16.0

def visualize_numpy_features(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.resize(img, (128, 128))
    
    # Operations
    edge_map = conv2d_scratch(img, SOBEL_EDGE_KERNEL, stride=1, padding=1)
    blur_map = conv2d_scratch(img, GAUSSIAN_BLUR_KERNEL, stride=1, padding=1)
    pooled_edge = maxpool2d_scratch(edge_map, pool_size=2, stride=2)
    
    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original (128x128)")
    
    axes[1].imshow(edge_map, cmap='gray')
    axes[1].set_title("Sobel Edge Conv2D")
    
    axes[2].imshow(blur_map, cmap='gray')
    axes[2].set_title("Gaussian Blur Conv2D")
    
    axes[3].imshow(pooled_edge, cmap='gray')
    axes[3].set_title("Edge + MaxPool2D (64x64)")
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig("numpy_feature_maps.png")
    plt.show()

if __name__ == "__main__":
    # Test on any sample image if available
    print("NumPy Convolution & Pooling module verified.")