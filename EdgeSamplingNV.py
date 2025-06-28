import numpy as np
import torch
import torchvision
import cv2

from torch.utils.data import DataLoader
from ProcessingConfig import ProcessingConfig
from ImageProcessor import ImageProcessor
from vgg_xcv_segment import vgg_xcv_segment

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
    
    def Edge_sampling(self, Dataloader):
        
        x = []
        y = []
        xx = []
        yy = []
        strength = []
        scale = []
        score = []
        
        for batch in Dataloader:
            
            """
                Get the input image tensor. 
                Beware that this image tensor is of shape (B, R, C) where:
                 -B is the batch size,
                 -R is the rows, and
                 -C is the columns.

            """

            # Get the number of images in the batch
            num_imgs, _ = batch

            # Iterate through each image in the batch and apply to it the vgg_xcv_segment function.
            for i in range(num_imgs.shape[0]):

                curves = vgg_xcv_segment(num_imgs[i], action = "canny_edges")

                # Concatenate all edgel segments together into one big array
                
                for i in range(len(curves)):
                    
                  xx.append(curves[i][0,:])  
                    
                    
                

        
        
        
            