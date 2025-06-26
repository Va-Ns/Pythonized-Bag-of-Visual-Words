import torchvision
import torch
import cv2

from pathlib import Path 
from torchvision.transforms import v2 as transforms
from ProcessingConfig import ProcessingConfig
from typing import List, Optional
from torch.utils.data import DataLoader
from torchvision import datasets 


class ImageProcessor:
    
    """Handles image loading and processing pipeline"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.transform = self._transform_pipeline()

    def _transform_pipeline(self):
        
        """Creates the image transformation pipeline"""
        
        return transforms.Compose([
                                    transforms.ToImage(),
                                    transforms.Grayscale(num_output_channels = 1),
                                    transforms.Lambda(lambda img: img.squeeze()),
                                    transforms.Lambda(lambda img: torch.from_numpy(cv2.resize(img.numpy(), dsize = None,  
                                                                                fx = self.config.XScale / img.shape[0],
                                                                                fy = self.config.XScale / img.shape[0],
                                                                                interpolation = cv2.INTER_LINEAR)))
                            
                          ])
            
    def setup_subdirs(self, subdirs: List[str]) -> None:
        
        """Sets up workspace subdirectories"""
        
        items = list(self.config.Workspace_dir.iterdir())
        if items and all(item.is_dir() for item in items):
            
            return  # Workspace with subdirectories is already set up

        else:
            
            for subdir in subdirs:
                
                dir_path = self.config.Workspace_dir / subdir
                dir_path.mkdir(exist_ok = True)
            
    def create_dataloader(self, image_dir: Path) -> DataLoader:
        
        """Creates a DataLoader for the image dataset"""
        
        dataset = datasets.ImageFolder(
            root = image_dir,
            transform = self.transform
        )
        return DataLoader(dataset, batch_size = 20, shuffle = True)
