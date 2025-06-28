import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import subprocess
import os


from pathlib import Path
from torchvision.utils import save_image

def parse_xcv_output(x, action):
        
        """
            Parse the numeric output from xcv_segment according to the action.
            Returns: s (parsed structure)
        """
        c = 0  
        s = None

        if action == 'corners':
            
            n = int(x[c])
            c += 1
            arr = np.array(x[c:c+2*n]).reshape((2, n))
            arr = arr + 1  # shift to MATLAB's 1-based frame
            s = arr
            c += 2 * n

        elif action in {'canny_edges', 'break_edges'}:
            
            # Take the number of segments as the integer form of the first 
            # element of the output data
            n_segments = int(x[c])

            # Increase the counter
            c += 1
            
            # Reform the output into a list
            s = []
            
            # Loop for n_segments times 
            for _ in range(n_segments):
                
                # The number of points is the integer form of the second element of the output data
                n_points = int(x[c])
                
                # Increase the counter
                c += 1
                
                """
                    The np.array(x[c:c+4*n_points]) format creates a (c:c+4*n_points,) vector
                    which is then reshaped into a (4, n_points) matrix. 
                """
                arr = np.array(x[c:c+4*n_points]).reshape((4, n_points))
                
                # Add 1 to all the columns of the first 3 rows
                arr[0:2, :] = arr[0:2, :] + 1  

                # Add the values to the output list
                s.append(arr)

                # Update the counter
                c += 4 * n_points

        elif action == 'lines':
            
            n = int(x[c])
            c += 1
            arr = np.array(x[c:c+4*n]).reshape((4, n))
            arr = arr + 1  # shift to MATLAB's 1-based frame
            s = arr
            c += 4 * n

        else:
            
            raise ValueError("Unknown action for parsing output")

        if len(x) + 1 != c + 1:
            
            raise ValueError("Wrong length of the output file")

        return s
    

def vgg_xcv_segment(Img: torch.Tensor, action: str, Options: str = ''):

    """
        VGG XCV segmentation using the specified action and options.
        Inputs: -Img (torch.Tensor): The input image tensor. The shape of the Tensor is (R, C) 
                                     where R is the rows, and C is the columns.

                -action (str): The action to perform on the image. Supported actions are 'corners',
                               'canny_edges', 'break_edges', and 'lines'.
                               
                -Options (str): Additional customizable options based on the action provided.

    """
    
    
    # Check if the input image is of type uint8
    if Img.dtype != torch.uint8:

        raise TypeError("Input image must be of type torch.uint8")
    
    # Check the values of the Options parameter
    if Options is None:
        
        Options = ''
    
    # Check the values of the action parameter        
    if action == 'corners':
        
        Options = (' -gauss_sigma 0.7'
                   ' -corner_count_max 300'
                   ' -relative_minimum 1e-5'
                   ' -scale_factor 0.04'
                   ' -adaptive 1'
                   f' {Options}')
    
    elif action in {'canny_edges', 'break_edges', 'lines'}:
        
        Options = (
                    ' -sigma 1'
                    ' -max_width 50'
                    ' -gauss_tail 0.0001'
                    ' -low 2'
                    ' -high 12'
                    ' -edge_min 60'
                    ' -min_length 10'
                    ' -border_size 2'
                    ' -border_value 0.0'
                    ' -scale 5.0'
                    ' -follow_strategy 2'
                    ' -join_flag 1'
                    ' -junction_option 0'
                    ' -bk_thresh 0.3'
                    ' -str_high 12'
                    ' -str_edge_min 60'
                    ' -str_min_length 60'
                    ' -min_fit_length 10'
                    f' {Options}'
                )
    
    else:
        
        raise ValueError("Unknown action required")   
    
    inname = 'vgg_xcv_in.png'
    outname = 'vgg_xcv_out'
    
    # Convert the image to PIL format. This is necessary
    # because the vgg_xcv_segment function expects a png image
    
    Img_pil = F.to_pil_image(Img)
    Img_pil.save(inname, format='PNG')
    
    # Get the path to the executable
    fname = str(Path(__file__).resolve().parent)
    
    # Define the path of the executable
    exec_path = os.path.join(fname, 'xcv_segment.exe')

    """
        Build the command. The aforementioned includes the following arguments:
        
            exec_path: Indicated the path to the executable.
            -i: Indicated the input image file.
            inname: The name of the input image file, as instantiated above.
            -f: Indicated the output file name.
            outname: The name of the output file, also instantiated above.
            f'-{action}': An f-string representing the action to be performed
                          on the image, as specified by the user.
    
    """
    
    cmd = [exec_path, '-i', inname, '-f', outname, f'-{action}']
    
    # Check if the Options variable is not an empty string
    if Options:

            # If it isn't, the Options variable is split into a list
            # and is added to the command.
            cmd += Options.split()

    """
        Call the executable. 
        
        The additional arguments are used to capture the return code 
    """
    result = subprocess.run(cmd, capture_output = True, text = True)

    # Check if running the executable was successful. If the exit status is 0,
    # the process was successful.   
    if result.returncode != 0: 
        
        raise RuntimeError("Calling the binary failed")
    
    os.remove(inname)
    
    try:
        
        # Open the created outname file in reading mode
        with open(outname, 'r') as f:
            
            # Read the content of the file
            content = f.read()
            
            # Convert all tokens to float
            x = [float(num) for num in content.split()]

            s = parse_xcv_output(x, action)

        # Remove the output file
        os.remove(outname)
        
        return s

        

    except:

        print("Error reading output file")
        content = None
