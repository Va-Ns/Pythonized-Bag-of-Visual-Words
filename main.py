import matplotlib.pyplot as plt
import numpy as np                                                           # We import numpy for numerical operations

from pathlib import Path                                                     # We import Path to create the directory structure
from torchvision import datasets                                             # We import datasets to load the dataset  
from torch.utils.data import DataLoader, random_split                        # We import random_split and DataLoader to split the dataset into 
                                                                             # training and validation sets and load the dataset
from torchvision.transforms import v2 as transforms                          # We import transforms to apply transformations to the images
from EdgeSamplingNV import EdgeSamplingNV                                    # We import the EdgeSamplingNV class from the EdgeSamplingNV module
from ProcessingConfig import ProcessingConfig                                # We import the ProcessingConfig class from the ProcessingConfig module
from ImageProcessor import ImageProcessor                                    # We import the ImageProcessor class from the ImageProcessor module
from timeit import default_timer as timer                                    # We import timer to measure the time taken for edge sampling
from ExtractSIFTFeatures import ExtractSIFTFeatures                          # We import the ExtractSIFTFeatures function from the ExtractSIFTFeatures module
from torch import Generator                                                  # We import Generator to set the random seed for reproducibility
from sklearn.cluster import KMeans                                           # We import KMeans for clustering the SIFT features
from sklearn.metrics import pairwise_distances_argmin_min, accuracy_score    # We import pairwise_distances_argmin_min to find the closest cluster center for each feature
from sklearn import svm                                                      # We import svm for building the classifier
from sklearn import preprocessing                                            # We import preprocessing for data standardization 
from sklearn.model_selection import GridSearchCV


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
    subdir_paths = Processor.Setup_subdirs(subdirs)
    
    # Create the data loader
    Loader = Processor.Create_dataloader(base_dir / "images")

    # Create EdgeSamplingNV instance
    Edge_Sampling = EdgeSamplingNV(Loader, Config)

    start = timer()
    # Assign the results of the edge sampling
    Variables = Edge_Sampling.Edge_Sampling(Loader)
    end = timer()
    print(f"\nEdge sampling took a mean time of {(end - start)/len(Loader.dataset):.2f} seconds.\n")

    start = timer()
    # Extract SIFT features (The data of the Loader are not resuffled)
    Features = ExtractSIFTFeatures(Loader, Variables, plot = False)
    end = timer()
    print(f"\nSIFT feature extraction took a mean time of {(end - start)/len(Loader.dataset):.2f} seconds.")

    # Initialize the generator for reproducibility
    generator = Generator().manual_seed(42)

    # Split the dataset into training and test sets
    # 75% for training and 25% for testing  
    [Train_dataset, Test_dataset] = random_split(Loader.dataset, [0.75, 0.25], generator=generator)
    
    # Instantiate the list to hold descriptors that will be used for training
    Descriptors = []

    for idx in range(len(Train_dataset)):

        Descriptors.append(Features[Train_dataset.indices[idx]])

    Descriptors = np.vstack(Descriptors)
    
    # Instantiate the KMeans clustering algorithm
    kmeans = KMeans(n_clusters = 300, init = 'k-means++', n_init='auto').fit(Descriptors)
    
    # Formate the Codebook
    Codebook = kmeans.cluster_centers_
    
    # Instantiate the lists of the quantized vector descriptors
    Training_vq_descriptors  = np.zeros((len(Train_dataset), Codebook.shape[0]))
    Testing_vq_descriptors  = np.zeros((len(Test_dataset), Codebook.shape[0]))

    for img_descriptor in range(len(Train_dataset)):

        index, _ = pairwise_distances_argmin_min(Features[Train_dataset.indices[img_descriptor]], Codebook)
        hist, _ = np.histogram(index, bins = Codebook.shape[0])

        Training_vq_descriptors[img_descriptor, :] = hist / len(index)

    for img_descriptor in range(len(Test_dataset)):

        index, _ = pairwise_distances_argmin_min(Codebook, Features[Test_dataset.indices[img_descriptor]])
        hist, _ = np.histogram(index, bins = Codebook.shape[0])

        Testing_vq_descriptors[img_descriptor, :] = hist / len(index)
    
    # Save the Codebook and the quantized vector descriptors
    np.save(subdir_paths[0] / "Codebook.npy", Codebook)
    np.save(subdir_paths[1] / "Training_vq_descriptors.npy", Training_vq_descriptors)
    np.save(subdir_paths[1] / "Testing_vq_descriptors.npy", Testing_vq_descriptors)
    np.save(subdir_paths[2] / "SIFT_features_of_interest_points.npy", Features)
    
    # Get the targets and indices for training and testing datasets
    Training_targets = np.array(Train_dataset.dataset.targets)
    Training_indices = np.array(Train_dataset.indices)
    
    # Get the targets and indices for testing dataset
    Testing_targets = np.array(Test_dataset.dataset.targets)
    Testing_indices = np.array(Test_dataset.indices)
    
    # Before the formation of the classifier, try to standardize the data
    scaler = preprocessing.MaxAbsScaler().fit(Training_vq_descriptors)
    
    Training_vq_descriptors = scaler.transform(Training_vq_descriptors)
    Testing_vq_descriptors = scaler.transform(Testing_vq_descriptors)
    
    # Define the parameter grid for the SVM classifier
    param_grid = {
    'kernel': ['linear','rbf'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale','auto']
    }
    
    # Build the classifier using GridSearchCV
    # This will perform a grid search over the specified parameters
    grid = GridSearchCV(svm.SVC(max_iter=5000, break_ties=True), param_grid,
                    cv=5, verbose=2, n_jobs=-1)
    grid.fit(Training_vq_descriptors, Training_targets[Training_indices])
    
    # Extract the best estimator from the grid search
    best_clf = grid.best_estimator_

    # Perform predictions on the test set
    y_pred = best_clf.predict(Testing_vq_descriptors)
    y_true = Testing_targets[Testing_indices]
    
    Accuracy = accuracy_score(y_true, y_pred)
    print(f"\nThe accuracy of the classifier is {Accuracy:.2f}.\n")


if __name__ == "__main__":
    main()