from int_to_image import int_to_image
from skimage.feature import hog


def calculate_hog(X_int_arr):
	# #YOUR_CODE_HERE
    
    X_hog_arr = []
    
    for i in range(len(X_int_arr)):
        
        X_img = int_to_image(X_int_arr[i])
        fd, X_hog = hog(X_img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        X_hog_arr.append(X_hog)
    
    return X_hog_arr
    


"""
===============================================================
                      EXCERSISE 1.4
===============================================================
"""
"""
mat_file = loadmat("lab1_data/new_data_int.mat")

grab_test_int = np.array(mat_file['garb_test_int'])
grab_train_int = np.array(mat_file['garb_train_int'])
ped_test_int = np.array(mat_file['ped_test_int'])
ped_train_int = np.array(mat_file['ped_train_int'])
   
grab_test_hog = int_to_hog(grab_test_int)
grab_train_hog = int_to_hog(grab_train_int)
ped_test_hog = int_to_hog(ped_test_int)
ped_train_hog = int_to_hog(ped_train_int)

plt.imshow(ped_train_hog[1], cmap="gray")
plt.show()
"""
