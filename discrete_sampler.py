from networkx.classes.function import is_empty
import numpy as np


def discrete_sampler(density, num_samples, replacement_options):
    
    samples_out = np.zeros((1, num_samples))
    
    cum_density = np.cumsum(density)

    uni_sample = np.random.rand(1, num_samples)
    
    a = 1
    
    while a <= num_samples:
        
        # Since uni_sample is a 1D array, we can directly index it
        binary = uni_sample[0, a - 1] > cum_density

        # Find where in the binary matrix the condition is True.
        # np.where returns two arrays, named 0 and 1. The 0 array 
        # represents the row indices, and the 1 array represents the column indices.
        highest = np.where(binary == True)[0]
        
        # Check if the highest variable is empty
        if np.any(highest) == False:
            
            samples_out[0, a - 1] = 1
            
        else:
            
            samples_out[0, a - 1] = highest[-1] + 1
            
        if not replacement_options and a > 1:
            
            if np.sum(samples_out[0, a-1] == samples_out[0, :a-1]) > 0:
                
                uni_sample[0, a - 1] = np.random.rand(1)
                
        a += 1
    
    # Use np.vectorize to convert the samples_out to integers
    vector = np.vectorize(np.int_)
    samples_out = vector(samples_out)
    
    return samples_out