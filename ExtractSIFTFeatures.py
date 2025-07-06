import cv2
import numpy as np

from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt

def ExtractSIFTFeatures(Train_loader: DataLoader, Variables: dict, plot: bool = False):
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Get all interest points
    interest_points = Variables['interest_point']
    all_descriptors = []
    
    """
        Here the for loop is destructuring the data in the following way:
        
        - batch_idx: Index of the current batch in the DataLoader.
        - (images, _): Yields whatever the Train_loader returns. 
                       We've seen that when following this syntax,
                       the DataLoader unpacks: 
                        - In the first position, the image tensor
                        - In the second position, the labels (which we ignore with '_').
    
    """
    
    for batch_idx, (images, _) in enumerate(Train_loader):
        
        for img_idx in range(images.shape[0]):
            
            # Get current image
            img = images[img_idx].numpy().astype(np.uint8)
            
            # Get corresponding interest points for this image
            points = interest_points[batch_idx * Train_loader.batch_size + img_idx]
            
            # Create keypoints for this image's interest points using list comprehension
            keypoints = [
                                            cv2.KeyPoint(
                                                        x=float(x),
                                                        y=float(y),
                                                        size=float(Variables['scale'][batch_idx * Train_loader.batch_size + img_idx].flatten()[i])
                                                        )
                                            
                                            for i, (x, y) in enumerate(points)
            ]
            
            # Compute SIFT descriptors
            _, descriptors = sift.compute(img, keypoints)
            all_descriptors.append(descriptors)
            
            # Print progress
            print(f"\rProcessing image {batch_idx * Train_loader.batch_size + img_idx + 1}", end="")

            if plot:
                
                # Create a copy for visualization
                img_with_keypoints = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                # Draw keypoints with circles and orientation
                for kp in keypoints:
                    
                    x, y = map(int, [kp.pt[0], kp.pt[1]])
                    size = int(kp.size)
                    cv2.circle(img_with_keypoints, (x, y), size, (0, 255, 0), 1)  # Green circles
                    cv2.circle(img_with_keypoints, (x, y), 1, (0, 0, 255), -1)    # Red center points
                
                # Show image with matplotlib (better for displaying in notebooks/interactive environments)
                plt.figure(figsize=(10, 10))
                plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
                plt.title(f'SIFT Features - Image {batch_idx * Train_loader.batch_size + img_idx + 1}')
                plt.axis('off')
                plt.show()
            
            
    return all_descriptors