import matplotlib.pyplot as plt


from pathlib import Path                                # We import Path to create the directory structure
from torchvision import datasets                        # We import datasets to load the dataset  
from torch.utils.data import DataLoader, random_split   # We import random_split and DataLoader to split the dataset into 
                                                        # training and validation sets and load the dataset
from torchvision.transforms import v2 as transforms     # We import transforms to apply transformations to the images
from EdgeSamplingNV import EdgeSamplingNV               # We import the EdgeSamplingNV class from the EdgeSamplingNV module
from ProcessingConfig import ProcessingConfig           # We import the ProcessingConfig class from the ProcessingConfig module
from ImageProcessor import ImageProcessor               # We import the ImageProcessor class from the ImageProcessor module
from timeit import default_timer as timer               # We import timer to measure the time taken for edge sampling
from ExtractSIFTFeatures import ExtractSIFTFeatures     # We import the ExtractSIFTFeatures function from the ExtractSIFTFeatures module

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
    Processor.Setup_subdirs(subdirs)
    
    # Create the data loader
    Train_loader = Processor.Create_dataloader(base_dir / "images")

    # Create EdgeSamplingNV instance
    Edge_Sampling = EdgeSamplingNV(Train_loader, Config)
    
    start = timer()
    # Assign the results of the edge sampling
    Variables = Edge_Sampling.Edge_Sampling(Train_loader)
    end = timer()
    print(f"Edge sampling took a mean time of {(end - start)/len(Train_loader.dataset):.2f} seconds.")

    start = timer()
    # Extract SIFT features (The data of the Train_loader are not resuffled)
    Features = ExtractSIFTFeatures(Train_loader, Variables, plot = True)
    end = timer()
    print(f"SIFT feature extraction took a mean time of {(end - start)/len(Train_loader.dataset):.2f} seconds.")

if __name__ == "__main__":
    main()