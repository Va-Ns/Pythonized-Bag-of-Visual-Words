import matplotlib.pyplot as plt

from pathlib import Path                                # We import Path to create the directory structure
from torchvision import datasets                        # We import datasets to load the dataset  
from torch.utils.data import DataLoader, random_split   # We import random_split and DataLoader to split the dataset into 
                                                        # training and validation sets and load the dataset
from torchvision.transforms import v2 as transforms     # We import transforms to apply transformations to the images
from EdgeSamplingNV import EdgeSamplingNV               # We import the EdgeSamplingNV class from the EdgeSamplingNV module
from ProcessingConfig import ProcessingConfig           # We import the ProcessingConfig class from the ProcessingConfig module
from ImageProcessor import ImageProcessor               # We import the ImageProcessor class from the ImageProcessor module



def main():
    
    # Setup configuration
    base_dir = Path()
    Workspace_dir = base_dir / "Workspace"
    Workspace_dir.mkdir(exist_ok = True)

    Config = ProcessingConfig(Workspace_dir = Workspace_dir)
    
    # Initialize processor with the configuration
    Processor = ImageProcessor(Config)

    # Create the subdirectory names and set them up 
    subdirs = [
        "Codebook",
        "Quantized_vector_descriptors",
        "SIFT_features_of_interest_points"
    ]
    Processor.setup_subdirs(subdirs)
    
    # Create the data loader
    Train_loader = Processor.create_dataloader(base_dir / "images")

    Edge_Sampling = EdgeSamplingNV(Train_loader, Config)
    
    Edge_Sampling.Edge_sampling(Train_loader)
    
    # Process a sample image
    image, _ = next(iter(Train_loader))
    img_display = image.squeeze(0).numpy()
    plt.imshow(img_display, cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()