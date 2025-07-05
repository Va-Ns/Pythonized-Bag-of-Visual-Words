import cv2
import numpy as np

from torch.utils.data import DataLoader, random_split

def ExtractSIFTFeatures(Train_loader: DataLoader, Variables: dict):
    
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
            
            # Create keypoints for this image's interest points
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
    
    return all_descriptors