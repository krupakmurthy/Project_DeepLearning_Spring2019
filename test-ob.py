#file to view image and boxes for prediction of object detection
#import library 
import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

i=15

#load model
options = {
    'model': 'cfg/yolo_2.cfg',
    'load': 300,                             
    'threshold': 0.05,                       
    'gpu': 1.0                               
}

#load image
tfnet = TFNet(options)
img = cv2.imread('training/images/isolation_0.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# use YOLO to predict the image
result = tfnet.return_predict(img)
tl = (result[i]['topleft']['x'], result[i]['topleft']['y'])
br = (result[i]['bottomright']['x'], result[i]['bottomright']['y'])
label = result[i]['label']

# add the box and label and display it for any 1 prediction 
img = cv2.rectangle(img, tl, br, (0, 255, 0), 7) # draw a ractangle onto an image
img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2) # add laebl name
plt.imshow(img)
plt.show()