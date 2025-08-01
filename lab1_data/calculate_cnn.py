import skimage
from load_data import load_data
from int_to_image import int_to_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet import MobileNet
import numpy as np

def image_to_convfeat(model, image):
    color = skimage.color.gray2rgb(image)
    img_data = np.expand_dims(color, axis=0)
    processed = preprocess_input(img_data)
    feature = model.predict(processed)
    return feature.flatten()

def calculate_cnn(model, X):
    X_img_arr = []
    conv_feat_arr = []
    for i in range(len(X)):
        X_img = int_to_image(X[i])
        X_img_arr.append(X_img)
        conv_feat = image_to_convfeat(model, X_img)
        
        conv_feat_arr.append(conv_feat)
        
    
    features = conv_feat_arr
    
    return features

"""
===============================================================
                      EXCERSISE 1.4
===============================================================
"""
"""
X_train_int, X_test_int, y_train, y_test = load_data()
model = MobileNet(weights='imagenet', include_top=False)
X_train_cnn = calculate_cnn(model, X_train_int)
X_test_cnn = calculate_cnn(model, X_test_int)
"""