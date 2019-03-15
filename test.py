#predication file
import sys
import tensorflow as tf
import numpy as np
import os
import cv2

input_value = tf.placeholder(tf.float32, [None, 224, 224, 1])
mode = tf.estimator.ModeKeys.PREDICT
# vgg16 structure same as train.py file 
conv1 = tf.layers.conv2d(
    inputs=input_value,
    filters=64,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv1_nb = tf.layers.batch_normalization(conv1,training=mode==tf.estimator.ModeKeys.PREDICT,renorm=True)
conv1_final=tf.nn.relu(conv1_nb)
conv2 = tf.layers.conv2d(
    inputs=conv1_final,
    filters=64,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv2_nb = tf.layers.batch_normalization(conv2,training=mode==tf.estimator.ModeKeys.PREDICT,renorm=True)
conv2_final=tf.nn.relu(conv2_nb)
pool1 = tf.layers.max_pooling2d(inputs=conv2_final, pool_size=[2, 2], strides=2)
conv3 = tf.layers.conv2d(
    inputs=pool1,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv3_nb = tf.layers.batch_normalization(conv3,training=mode==tf.estimator.ModeKeys.PREDICT,renorm=True)
conv3_final=tf.nn.relu(conv3_nb)
conv4 = tf.layers.conv2d(
    inputs=conv3_final,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv4_nb = tf.layers.batch_normalization(conv4,training=mode==tf.estimator.ModeKeys.PREDICT,renorm=True)
conv4_final=tf.nn.relu(conv4_nb)
pool2 = tf.layers.max_pooling2d(inputs=conv4_final, pool_size=[2, 2], strides=2)
conv5 = tf.layers.conv2d(
    inputs=pool2,
    filters=256,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv5_nb = tf.layers.batch_normalization(conv5,training=mode==tf.estimator.ModeKeys.PREDICT,renorm=True)
conv5_final=tf.nn.relu(conv5_nb)
conv6 = tf.layers.conv2d(
    inputs=conv5_final,
    filters=256,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv6_nb = tf.layers.batch_normalization(conv6,training=mode==tf.estimator.ModeKeys.PREDICT,renorm=True)
conv6_final=tf.nn.relu(conv6_nb)
conv7 = tf.layers.conv2d(
    inputs=conv6_final,
    filters=256,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv7_nb = tf.layers.batch_normalization(conv7,training=mode==tf.estimator.ModeKeys.PREDICT,renorm=True)
conv7_final=tf.nn.relu(conv7_nb)
pool3 = tf.layers.max_pooling2d(inputs=conv7_final, pool_size=[2, 2], strides=2)
conv8 = tf.layers.conv2d(
    inputs=pool3,
    filters=512,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv8_nb = tf.layers.batch_normalization(conv8,training=mode==tf.estimator.ModeKeys.PREDICT,renorm=True)
conv8_final=tf.nn.relu(conv8_nb)
conv9 = tf.layers.conv2d(
    inputs=conv8_final,
    filters=512,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv9_nb = tf.layers.batch_normalization(conv9,training=mode==tf.estimator.ModeKeys.PREDICT,renorm=True)
conv9_final=tf.nn.relu(conv9_nb)
conv10 = tf.layers.conv2d(
    inputs=conv9_final,
    filters=512,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv10_nb = tf.layers.batch_normalization(conv10,training=mode==tf.estimator.ModeKeys.PREDICT,renorm=True)
conv10_final=tf.nn.relu(conv10_nb)
pool4 = tf.layers.max_pooling2d(inputs=conv10_final, pool_size=[2, 2], strides=2)
conv11 = tf.layers.conv2d(
    inputs=pool4,
    filters=512,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv11_nb = tf.layers.batch_normalization(conv11,training=mode==tf.estimator.ModeKeys.PREDICT,renorm=True)
conv11_final=tf.nn.relu(conv11_nb)
conv12 = tf.layers.conv2d(
    inputs=conv11_final,
    filters=512,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv12_nb = tf.layers.batch_normalization(conv12,training=mode==tf.estimator.ModeKeys.PREDICT,renorm=True)
conv12_final=tf.nn.relu(conv12_nb)
conv13 = tf.layers.conv2d(
    inputs=conv12_final,
    filters=512,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv13_nb = tf.layers.batch_normalization(conv13,training=mode==tf.estimator.ModeKeys.PREDICT,renorm=True)
conv13_final=tf.nn.relu(conv13_nb)

pool5 = tf.layers.max_pooling2d(inputs=conv13_final, pool_size=[2, 2], strides=2)

fc1 = tf.layers.conv2d(
    inputs=pool5,
    filters=4096,
    kernel_size=[7, 7],
    padding="valid",
    activation=tf.nn.relu
) 
fc2 = tf.layers.conv2d(
    inputs=fc1,
    filters=4096,
    kernel_size=[1, 1],
    padding="valid",
    activation=tf.nn.relu
)
fc3 = tf.layers.conv2d(
    inputs=fc2,
    filters=10,
    kernel_size=[1, 1],
    padding="valid",
    activation=None
) 
logits = tf.squeeze(fc3, axis=[1,2])
prediction = tf.argmax(input=logits, axis=1)
saver = tf.train.Saver()
#taking file name as a command line argument and predicting results using trained model 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "C:/Users/Dell/Desktop/DL/trained_model_final/DL/model.cktp-640")
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    test_x=np.array(img)
    test_data = (test_x-127.0)/np.float32(127)
    test_data.shape = [-1, 224, 224, 1]
    x = sess.run(prediction, feed_dict={input_value: test_data})
    if x==1:print("gossiping")  
    elif x == 2:print("isolation")
    elif x ==3:print("laughing")
    elif x ==4:print("pullinghair") 
    elif x ==5:print("punching") 
    elif x ==6:print("quarrel") 
    elif x==7:print("slapping")  
    elif x ==8:print("stabbing") 
    elif x ==9:print("strangle") 
    elif x ==0:print("nonbullying")




