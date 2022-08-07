import tensorflow as tf
import numpy as np
import os
import cv2
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

image_size=32
num_channels=3

# Let us restore the saved model 
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('models/trained_model.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./models/'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()
# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, 12)) 
predicted_class = None
dirs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "n", "m"]

def predict(img):
    
    image = cv2.resize(img, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    image = np.stack((image,)*3, axis=-1)
    images = []
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0) 
    x_batch = images.reshape(1, image_size,image_size,num_channels)
    
    # Creating the feed_dict that is required to be fed to calculate y_pred 
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    # print("--feed_dict_testing---", feed_dict_testing)
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    # Result is of this format [[probabiliy_of_classA probability_of_classB ....]]

    a = result[0].tolist()
    r=0
    max1 = max(a)
    index1 = a.index(max1)
    count = 0
    
    for name in dirs:
        if count==index1:
            predicted_class = name
        count+=1

    for i in a:
        if i!=max1:
            if max1-i<i:
                r=1                           
    if r ==0:
        # print(predicted_class)
        return predicted_class













