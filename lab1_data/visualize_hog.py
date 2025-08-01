from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np

from skimage.feature import hog
from int_to_image import int_to_image
import matplotlib.pyplot as plt       
        
def visualize_hog_orientations(X_train_int):
	# #YOUR_CODE_HERE
    
    num_of_examples = X_train_int.shape[0]
    fig = plt.figure(figsize=(16.0,20.0))
    
    for j in range(3):
        for i in range(num_of_examples):
       
            ax = fig.add_subplot(3, 4, j*num_of_examples + i+1)
            
            img = int_to_image(X_train_int[i])
            
            
            if j == 0:
                fd, X_train_hog = hog(img, orientations=4, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
            
            if j == 1:
                fd, X_train_hog = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    	
            if j == 2:
                fd, X_train_hog = hog(img, orientations=32, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
            
            ax.imshow((X_train_hog), cmap='gray')
            plt.axis("off")
            plt.savefig('images\hog_orient.png')
    plt.show()    
    
    return X_train_hog

def visualize_hog_pixels_per_cell(X_train_int):
    # #YOUR_CODE_HERE
    num_of_examples = X_train_int.shape[0]
    fig = plt.figure(figsize=(16.0,20.0))
    
    for j in range(3):
        for i in range(num_of_examples):
       
            ax = fig.add_subplot(3, 4, j*num_of_examples + i+1)
            
            img = int_to_image(X_train_int[i])
            
           
            if j == 0:
                fd, X_train_hog = hog(img, orientations=8, pixels_per_cell=(4,4), cells_per_block=(2, 2), visualize=True)
            
            if j == 1:
                fd, X_train_hog = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    	
            if j == 2:
                fd, X_train_hog = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
            
            ax.imshow((X_train_hog), cmap='gray')
            plt.axis("off")
            plt.savefig('images\hog_pix.png')
    plt.show()    
    
    
    return X_train_hog


    
    

"""
===============================================================
                      EXCERSISE 1.2 + 1.3
===============================================================
"""

"""
mat_file = loadmat("lab1_data/new_data_int.mat")

grab_test_int = np.array(mat_file['garb_test_int'])
grab_train_int = np.array(mat_file['garb_train_int'])
ped_test_int = np.array(mat_file['ped_test_int'])
ped_train_int = np.array(mat_file['ped_train_int'])
   
ped1 = ped_train_int[0]
ped2 = ped_train_int[1]

grab1 = grab_train_int[0]
grab2 = grab_train_int[1]

ped1_hog = visualize_hog_orientations(ped1)
ped2_hog = visualize_hog_orientations(ped2)
grab1_hog = visualize_hog_orientations(grab1)
grab2_hog = visualize_hog_orientations(grab2)

#argument 'orientation' was changed manually in the function

ped1_hog = visualize_hog_pixels_per_cell(ped1)
ped2_hog = visualize_hog_pixels_per_cell(ped2)
grab1_hog = visualize_hog_pixels_per_cell(grab1)
grab2_hog = visualize_hog_pixels_per_cell(grab2)

#argurment 'pixels_per_cell' was changed manually in the function
"""


