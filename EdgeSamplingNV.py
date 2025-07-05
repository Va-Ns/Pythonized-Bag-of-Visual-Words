import numpy as np
import torch
import torchvision
import cv2
import matplotlib.pyplot as plt 
import sys

from torch.utils.data import DataLoader
from ProcessingConfig import ProcessingConfig
from ImageProcessor import ImageProcessor
from vgg_xcv_segment import vgg_xcv_segment
from discrete_sampler import discrete_sampler
from numpy.random import rand
from pathlib import Path



class EdgeSamplingNV: 
    
    # --------------------------------Class Definition-----------------------------
    def __init__(self, DataLoader: DataLoader, Config: ProcessingConfig):
    
        """
            Initializes the EdgeSamplingNV object with the provided DataLoader and configuration.
        """
        
        self.XScale = Config.XScale
        self.Max_Points = Config.Max_points
        self.Scale = Config.Scale
        self.Weighted_Sampling = Config.Weighted_sampling
        self.WorkspaceDir = Config.Workspace_dir
        self.Plot = Config.Plot 
        

    def __str__(self):
        
        """
        String representation of the EdgeSamplingNV object.
        
        Returns:
        - str: A string describing the EdgeSamplingNV object.
        """
        
        return (f"EdgeSamplingNV(XScale={self.XScale},\n Max_Points={self.Max_Points},\n"
                f"Scale={self.Scale},\n Weighted_Sampling={self.Weighted_Sampling},\n "
                f"WorkspaceDir={self.WorkspaceDir},\n Plot={self.Plot})\n")
    
    def __repr__(self):
        
        """
        Representation of the EdgeSamplingNV object.
        
        Returns:
        - str: A string representation of the EdgeSamplingNV object.
        """
        
        return f"EdgeSamplingNV(XScale={self.XScale}, Max_Points={self.Max_Points}, " \
               f"Scale={self.Scale}, Weighted_Sampling={self.Weighted_Sampling}, " \
               f"WorkspaceDir={self.WorkspaceDir}, Plot={self.Plot})"
    
    def Edge_Sampling(self, Dataloader):

        x, y, scale, score, interest_point  = [], [], [], [], []
        xx = np.empty((1, 0))
        yy = np.empty((1, 0))
        strength = np.empty((1, 0))
        scale = []
        score = []
        interest_point = []

        total_images_processed = 0
        num_batches = len(Dataloader)
        
        for batch_idx, batch in enumerate(Dataloader):

            """
                Get the input image tensor. 
                Beware that this image tensor is of shape (B, R, C) where:
                 -B is the batch size,
                 -R is the rows, and
                 -C is the columns.

            """

            # Get the number of images in the batch
            num_imgs, _ = batch
            batch_size = num_imgs.shape[0]

            # Iterate through each image in the batch and apply to it the vgg_xcv_segment function.
            for i in range(batch_size):
                
                total_images_processed += 1
                
                progress_msg = (
                    f"\rBatch {batch_idx+1}/{num_batches} | Image {i+1}/{batch_size} | "
                    f"Total images processed: {total_images_processed}"
                )
                # Pad with spaces to clear previous content
                print(progress_msg + ' ' * 20, end='', flush=True)

                # The vgg_xcv_segment function is used to extract the canny edges from the image.
                # The outcome of the function is a list of edgel segments, each of which is of 
                # type float64.
                curves = vgg_xcv_segment(num_imgs[i], action = "canny_edges")

                # Check if curves is not None and has a length attribute. This ensures that
                # we can iterate over it safely.
                if curves is not None and hasattr(curves, '__len__'):
                    
                    # Concatenate all edgel segments together into one big array
                    for edgel in range(len(curves)):
                        
                        # Ensure each curve component is a 1D numpy array before reshaping
                        x_edgel = np.asarray(curves[edgel][0, :]).reshape(1, -1)
                        y_edgel = np.asarray(curves[edgel][1, :]).reshape(1, -1)
                        strength_edgel = np.asarray(curves[edgel][2, :]).reshape(1, -1)

                        xx = np.concatenate((xx, x_edgel), axis=1)
                        yy = np.concatenate((yy, y_edgel), axis=1)
                        strength = np.concatenate((strength, strength_edgel), axis=1)

                nEdgels = np.size(strength)

                if nEdgels > 0: # Check if there are any edgels
                    
                    if self.Weighted_Sampling:
                        
                        # Calculate the sample density based on the strength of the edgels
                        sample_density = strength / np.sum(strength)
                        
                    else: 
                        
                        # If not weighted sampling, set the sample density to uniform distribution
                        sample_density = np.ones_like(nEdgels) / nEdgels  
                        
                    nPoints_to_Sample = self.Max_Points
                    
                    # Create a vector of indices to sample from the edgels
                    samples = discrete_sampler(sample_density, nPoints_to_Sample, replacement_options = 1) 
                    
                    """
                        Lookup point corresponding to the samples. 
                        Here, instead of using xx[:, samples] and yy[:, samples], we use np.take
                        to ensure that the Buffer is used correctly. 
                    
                        Also, the use of .flatten() and .T are to ensure that the dimensions of the 
                        output (here interest_point) doesn't contain the depiction of the 
                        instance in the loop (e.g. the image that is being processed)."""
                    
                    x.append(np.take(xx, samples).flatten())
                    y.append(np.take(yy, samples).flatten())

                    # Here, using the -1 index, we're indexing to the last 
                    # item in the list, hence the current image being processed.
                    interest_point.append(np.vstack((x[-1], y[-1])).T)

                    scale.append(rand(1,nPoints_to_Sample) * 
                                (np.max(self.Scale) - np.min(self.Scale)) + np.min(self.Scale))
                    
                    score.append(np.take(strength, samples))
                    
                else:
                    
                    print("No edgels found in the image.")
                    continue

            if self.Plot: # Needs to be fixed
                
                for img_idx in range(num_imgs.shape[0]):
                    
                    # 1) Display the image using OpenCV
                    img = num_imgs[img_idx].numpy()
                    
                    if img.ndim == 2:
                        img_disp = img.astype(np.uint8)
                    else:
                        img_disp = img.transpose(1, 2, 0).astype(np.uint8)  # if CHW format

                    cv2.imshow(f'Image {img_idx+1}', img_disp)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    # 2) Create a second plot with xx and yy points using matplotlib
                    plt.figure()
                    plt.scatter(xx.flatten(), yy.flatten(), c='m', s=8)
                    plt.title(f'Edgel points for Image {img_idx+1}')
                    plt.gca().invert_yaxis()  # To match image coordinates
                    plt.show()
        
        save_dir = self.WorkspaceDir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        """" 
             By using the dtype = object option, we give numpy the flexibility
             to handle arrays of different shapes and sizes.
             This is particularly useful if a variable is a list of arrays 
             that may have different shapes or lengths. In this manner, 
             we can store a heterogenous collection of arrays in a single .npy file
        """
        np.save(save_dir / "x.npy", np.array(x, dtype = object))
        np.save(save_dir / "y.npy", np.array(y, dtype=object))
        np.save(save_dir / "scale.npy", np.array(scale, dtype=object))
        np.save(save_dir / "score.npy", np.array(score, dtype=object))
        np.save(save_dir / "interest_point.npy", np.array(interest_point, dtype=object))
        
        # Collect variables for the current image
        Variables = {
            "x": x,
            "y": y,
            "interest_point": interest_point,
            "scale": scale,
            "score": score
        }
        
        return Variables                                 