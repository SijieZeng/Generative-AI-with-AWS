from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np

def int_to_image(X):
# #YOUR_CODE_HERE

    X_img = (X.reshape(48, 96)).T
    
    return X_img


"""
===============================================================
                      EXCERSISE 1.1
===============================================================
mat_file = loadmat("lab1_data/new_data_int.mat")
grab_test_int = np.array(mat_file['garb_test_int'])
grab_train_int = np.array(mat_file['garb_train_int'])
ped_test_int = np.array(mat_file['ped_test_int'])
ped_train_int = np.array(mat_file['ped_train_int'])
   
grab_train_img = []
grab_test_img = []
ped_train_img = []
ped_test_img = []

for i in range(len(grab_test_int)):
    grab_test_img.append( int_to_image(grab_test_int[i]))

for i in range(len(grab_train_int)):
    grab_train_img.append( int_to_image(grab_train_int[i]))

for i in range(len(ped_test_int)):
    ped_test_img.append( int_to_image( ped_test_int[i]))
    
for i in range(len(ped_train_int)):
    ped_train_img.append( int_to_image(ped_train_int[i]))
    
        
for i in range(5):
    image_grab = grab_train_img[i]
    image_ped = ped_train_img[i]
        
        
    imgplot_grab = plt.imshow(image_grab)
    plt.show()
        
    imgplot_ped = plt.imshow(image_ped)
    plt.show()
        
"""