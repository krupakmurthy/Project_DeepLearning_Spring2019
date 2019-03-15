# file to create train and test data in input format with labels 
import cv2 
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm 

TRAIN= 'C:/Users/Dell/Desktop/DL/output/train'
TEST= 'C:/Users/Dell/Desktop/DL/output/val'
size=224

#encoding labels with the number index 0 to 9 for 10 category
def label_img(img): 
	label = img.split('.')[-2]
	label=label.split('_')[-2]
	if label == 'gossiping' : return 1
	elif label == 'isolation': return 2
	elif label == 'laughing' : return 3
	elif label == 'pullinghair': return 4 
	elif label == 'punching': return 5
	elif label == 'quarrel': return 6
	elif label == 'slapping': return 7 
	elif label == 'stabbing': return 8
	elif label == 'strangle': return 9
	elif label == 'nonbullying': return 0
# creating train and test data reading images with fixed size and seperating labels from images and appending as an array
def create_train_data(): 
	training_data = [] 

	for img in tqdm(os.listdir(TRAIN)): 
		label = label_img(img) 
		path = os.path.join(TRAIN, img) 
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
		img = cv2.resize(img, (size,size))
		training_data.append([np.array(img),np.array(label)]) 

	return training_data 

def create_test_data(): 
	testing_data = [] 

	for img in tqdm(os.listdir(TEST)): 
		label = label_img(img) 
		path = os.path.join(TEST, img) 
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
		img = cv2.resize(img, (size, size))
		testing_data.append([np.array(img),np.array(label)])
        
	return testing_data
#Generating train and test data and labels in proper format
train_data = create_train_data() 
test_data = create_test_data() 

train_x = np.array([i[0] for i in train_data])
train_y = np.array([i[1] for i in train_data])
test_x = np.array([i[0] for i in test_data]) 
test_y = np.array([i[1] for i in test_data])
#normaliztion of images before giving it as input
test_data = (test_x-127.0)/np.float32(127)
test_data.shape = [test_data.shape[0], 224, 224, 1]
test_labels = test_y.astype(np.int32)
train_data = (train_x-127.0)/np.float32(127)
train_data.shape = [train_data.shape[0], 224, 224, 1]
train_labels = train_y.astype(np.int32) 