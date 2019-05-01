#training model file
import sys
import tensorflow as tf
import numpy as np

input_value = tf.placeholder(tf.float32, [None, 224, 224, 1])
labels = tf.placeholder(tf.int32, [None])

mode = tf.estimator.ModeKeys.TRAIN

''' vgg16 convulation model with 13 convulation layer with batch normalization with 3 fully connected layers in end with gradenit decent optimizer and softmax at the end. 
'''

conv1 = tf.layers.conv2d(
    inputs=input_value,
    filters=64,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv1_nb = tf.layers.batch_normalization(conv1,training=mode==tf.estimator.ModeKeys.TRAIN,renorm=True)
conv1_final=tf.nn.relu(conv1_nb)
conv2 = tf.layers.conv2d(
    inputs=conv1_final,
    filters=64,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv2_nb = tf.layers.batch_normalization(conv2,training=mode==tf.estimator.ModeKeys.TRAIN,renorm=True)
conv2_final=tf.nn.relu(conv2_nb)
pool1 = tf.layers.max_pooling2d(inputs=conv2_final, pool_size=[2, 2], strides=2)
conv3 = tf.layers.conv2d(
    inputs=pool1,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv3_nb = tf.layers.batch_normalization(conv3,training=mode==tf.estimator.ModeKeys.TRAIN,renorm=True)
conv3_final=tf.nn.relu(conv3_nb)
conv4 = tf.layers.conv2d(
    inputs=conv3_final,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv4_nb = tf.layers.batch_normalization(conv4,training=mode==tf.estimator.ModeKeys.TRAIN,renorm=True)
conv4_final=tf.nn.relu(conv4_nb)
pool2 = tf.layers.max_pooling2d(inputs=conv4_final, pool_size=[2, 2], strides=2)
conv5 = tf.layers.conv2d(
    inputs=pool2,
    filters=256,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv5_nb = tf.layers.batch_normalization(conv5,training=mode==tf.estimator.ModeKeys.TRAIN,renorm=True)
conv5_final=tf.nn.relu(conv5_nb)
conv6 = tf.layers.conv2d(
    inputs=conv5_final,
    filters=256,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv6_nb = tf.layers.batch_normalization(conv6,training=mode==tf.estimator.ModeKeys.TRAIN,renorm=True)
conv6_final=tf.nn.relu(conv6_nb)
conv7 = tf.layers.conv2d(
    inputs=conv6_final,
    filters=256,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv7_nb = tf.layers.batch_normalization(conv7,training=mode==tf.estimator.ModeKeys.TRAIN,renorm=True)
conv7_final=tf.nn.relu(conv7_nb)
pool3 = tf.layers.max_pooling2d(inputs=conv7_final, pool_size=[2, 2], strides=2)
conv8 = tf.layers.conv2d(
    inputs=pool3,
    filters=512,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv8_nb = tf.layers.batch_normalization(conv8,training=mode==tf.estimator.ModeKeys.TRAIN,renorm=True)
conv8_final=tf.nn.relu(conv8_nb)
conv9 = tf.layers.conv2d(
    inputs=conv8_final,
    filters=512,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv9_nb = tf.layers.batch_normalization(conv9,training=mode==tf.estimator.ModeKeys.TRAIN,renorm=True)
conv9_final=tf.nn.relu(conv9_nb)
conv10 = tf.layers.conv2d(
    inputs=conv9_final,
    filters=512,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv10_nb = tf.layers.batch_normalization(conv10,training=mode==tf.estimator.ModeKeys.TRAIN,renorm=True)
conv10_final=tf.nn.relu(conv10_nb)
pool4 = tf.layers.max_pooling2d(inputs=conv10_final, pool_size=[2, 2], strides=2)
conv11 = tf.layers.conv2d(
    inputs=pool4,
    filters=512,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv11_nb = tf.layers.batch_normalization(conv11,training=mode==tf.estimator.ModeKeys.TRAIN,renorm=True)
conv11_final=tf.nn.relu(conv11_nb)
conv12 = tf.layers.conv2d(
    inputs=conv11_final,
    filters=512,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv12_nb = tf.layers.batch_normalization(conv12,training=mode==tf.estimator.ModeKeys.TRAIN,renorm=True)
conv12_final=tf.nn.relu(conv12_nb)
conv13 = tf.layers.conv2d(
    inputs=conv12_final,
    filters=512,
    kernel_size=[3, 3],
    padding="same",
    activation=None)
conv13_nb = tf.layers.batch_normalization(conv13,training=mode==tf.estimator.ModeKeys.TRAIN,renorm=True)
conv13_final=tf.nn.relu(conv13_nb)

pool5 = tf.layers.max_pooling2d(inputs=conv13_final, pool_size=[2, 2], strides=2)

fc1 = tf.layers.conv2d(
    inputs=pool5,
    filters=4096,
    kernel_size=[7, 7],
    padding="valid",
    activation=tf.nn.relu
) # 1x1x4096
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
) # 1x1x10
logits = tf.squeeze(fc3, axis=[1,2])
#calulating accuracy, loss and summary
acc = tf.reduce_mean(tf.cast(
        tf.equal(labels, tf.argmax(input=logits, axis=1, output_type=tf.int32)),
        tf.float32
    ))

loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

# ######################################
loss = tf.reduce_mean(loss)
global_step = tf.train.get_or_create_global_step()
#######################################

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train_op = optimizer.minimize(
    loss=loss,
    global_step=global_step)
update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train_op=tf.group([train_op,update_ops])

summary_op = tf.summary.merge([
    tf.summary.scalar("train/loss", loss),
    tf.summary.scalar("train/accuracy", acc),
])

# importing dataset from create_data file
from create_data import train_data
from create_data import train_labels 
#running model with batch size=32 and epoch=35
order = np.arange(len(train_data))
bs = 32
print("load data:", len(train_data))
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, "C:/Users/Dell/Desktop/DL/trained_model/DL/model.cktp-61")

    logger = tf.summary.FileWriter("log", sess.graph)
    for epo in range(20):
        np.random.shuffle(order)
        ind = 0
        while ind < len(order):
            _, summary, l, a, g = sess.run([train_op, summary_op, loss, acc, global_step], feed_dict={
                input_value: train_data[order[ind:ind+bs]],
                labels: train_labels[order[ind:ind+bs]]
            })
            
            if g % 10 == 0:
                print(g, l, a)
                
            logger.add_summary(summary, g)
            ind += bs
        saver.save(sess, "C:/Users/Dell/Desktop/DL/trained_model_final/DL/model.cktp", g)
        print("Epoch", epo, "done")
