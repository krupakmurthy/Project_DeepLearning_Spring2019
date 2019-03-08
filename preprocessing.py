import os
os.getcwd()

from os.path import join
from os import listdir, rmdir
from shutil import move

#initial initilization
pullinghair = "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/pullinghair"
gossiping = "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/gossiping"
isolation = "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/isolation"
laughing = "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/laughing"
punching = "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/punching"
quarrel = "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/quarrel"
slapping = "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/slapping"
stabbing = "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/stabbing"
strangle = "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/strangle"
nonbullying = "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/nonbullying"


#Creating new files for naming convention.

path = "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/pullinghair_main"
os.makedirs(path)
path = "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/gossiping_main"
os.makedirs(path)
path = "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/isolation_main"
os.makedirs(path)
path = "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/laughing_main"
os.makedirs(path)
path = "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/punching_main"
os.makedirs(path)
path = "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/quarrel_main"
os.makedirs(path)
path = "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/slapping_main"
os.makedirs(path)
path = "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/stabbing_main"
os.makedirs(path)
path = "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/strangle_main"
os.makedirs(path)
path = "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/nonbullying_main"
os.makedirs(path)

#rename the files in the folder (images) here accordingly. I have used here a '_'.

for i, filename in enumerate(os.listdir(pullinghair)):
    os.rename("C:/Users/krupa krishnamurthy/split-folders/complete_dataset/pullinghair/" + filename, "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/pullinghair_main/pullinghair_" + str(i) + ".jpg")
rmdir(pullinghair)

for i, filename in enumerate(os.listdir(nonbullying)):
    os.rename("C:/Users/krupa krishnamurthy/split-folders/complete_dataset/nonbullying/" + filename, "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/nonbullying_main/nonbullying_" + str(i) + ".jpg")
rmdir(nonbullying)

for i, filename in enumerate(os.listdir(gossiping)):
    os.rename("C:/Users/krupa krishnamurthy/split-folders/complete_dataset/gossiping/" + filename, "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/gossiping_main/gossiping_" + str(i) + ".jpg")
rmdir(gossiping) 

for i, filename in enumerate(os.listdir(isolation)):
    os.rename("C:/Users/krupa krishnamurthy/split-folders/complete_dataset/isolation/" + filename, "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/isolation_main/isolation_" + str(i) + ".jpg")
rmdir(isolation) 

for i, filename in enumerate(os.listdir(laughing)):
    os.rename("C:/Users/krupa krishnamurthy/split-folders/complete_dataset/laughing/" + filename, "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/laughing_main/laughing_" + str(i) + ".jpg")
rmdir(laughing) 

for i, filename in enumerate(os.listdir(punching)):
    os.rename("C:/Users/krupa krishnamurthy/split-folders/complete_dataset/punching/" + filename, "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/punching_main/punching_" + str(i) + ".jpg")
rmdir(punching) 

for i, filename in enumerate(os.listdir(quarrel)):
    os.rename("C:/Users/krupa krishnamurthy/split-folders/complete_dataset/quarrel/" + filename, "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/quarrel_main/quarrel_" + str(i) + ".jpg")
rmdir(quarrel) 

for i, filename in enumerate(os.listdir(slapping)):
    os.rename("C:/Users/krupa krishnamurthy/split-folders/complete_dataset/slapping/" + filename, "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/slapping_main/slapping_" + str(i) + ".jpg")
rmdir(slapping) 

for i, filename in enumerate(os.listdir(stabbing)):
    os.rename("C:/Users/krupa krishnamurthy/split-folders/complete_dataset/stabbing/" + filename, "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/stabbing_main/stabbing_" + str(i) + ".jpg")
rmdir(stabbing) 

for i, filename in enumerate(os.listdir(strangle)):
    os.rename("C:/Users/krupa krishnamurthy/split-folders/complete_dataset/strangle/" + filename, "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/strangle_main/strangle_" + str(i) + ".jpg")
rmdir(strangle) 

#renaming the fodlers
os.rename("C:/Users/krupa krishnamurthy/split-folders/complete_dataset/isolation_main/", "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/isolation/")
os.rename("C:/Users/krupa krishnamurthy/split-folders/complete_dataset/pullinghair_main/", "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/pullinghair/")
os.rename("C:/Users/krupa krishnamurthy/split-folders/complete_dataset/gossiping_main/", "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/gossiping/")
os.rename("C:/Users/krupa krishnamurthy/split-folders/complete_dataset/laughing_main/", "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/laughing/")
os.rename("C:/Users/krupa krishnamurthy/split-folders/complete_dataset/punching_main/", "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/punching/")
os.rename("C:/Users/krupa krishnamurthy/split-folders/complete_dataset/quarrel_main/", "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/quarrel/")
os.rename("C:/Users/krupa krishnamurthy/split-folders/complete_dataset/slapping_main/", "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/slapping/")
os.rename("C:/Users/krupa krishnamurthy/split-folders/complete_dataset/stabbing_main/", "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/stabbing/")
os.rename("C:/Users/krupa krishnamurthy/split-folders/complete_dataset/strangle_main/", "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/strangle/")
os.rename("C:/Users/krupa krishnamurthy/split-folders/complete_dataset/nonbullying_main/", "C:/Users/krupa krishnamurthy/split-folders/complete_dataset/nonbullying/")

#split the dataset in test and train using split_folder package
#This will have subfolders with each category in test and train folders 
import split_folders

split_folders.ratio('C:/Users/krupa krishnamurthy/split-folders/complete_dataset', output="C:/Users/krupa krishnamurthy/split-folders/output", seed=1337, ratio=(.8, .2))




# to move the files from subfolders to "train", name the sources and the destination
import shutil

source1 = 'C:/Users/krupa krishnamurthy/split-folders/output/train/isolation/'
files1 = os.listdir(source1)
source2 = 'C:/Users/krupa krishnamurthy/split-folders/output/train/pullinghair/'
files2 = os.listdir(source2)
source3 = 'C:/Users/krupa krishnamurthy/split-folders/output/train/gossiping/'
files3 = os.listdir(source3)
source4 = 'C:/Users/krupa krishnamurthy/split-folders/output/train/laughing/'
files4 = os.listdir(source4)
source5 = 'C:/Users/krupa krishnamurthy/split-folders/output/train/punching/'
files5 = os.listdir(source5)
source6 = 'C:/Users/krupa krishnamurthy/split-folders/output/train/quarrel/'
files6 = os.listdir(source6)
source7 = 'C:/Users/krupa krishnamurthy/split-folders/output/train/slapping/'
files7 = os.listdir(source7)
source8 = 'C:/Users/krupa krishnamurthy/split-folders/output/train/stabbing/'
files8 = os.listdir(source8)
source9 = 'C:/Users/krupa krishnamurthy/split-folders/output/train/strangle/'
files9 = os.listdir(source9)
source10 = 'C:/Users/krupa krishnamurthy/split-folders/output/train/nonbullying/'
files10 = os.listdir(source10)

dest1 = 'C:/Users/krupa krishnamurthy/split-folders/output/train/'


#move files from subfodlers to the train folder (parent)
for f in files1:
        shutil.move(source1+f, dest1)      
        
for f in files2:
        shutil.move(source2+f, dest1)
        
for f in files3:
        shutil.move(source3+f, dest1)

for f in files4:
        shutil.move(source4+f, dest1)
        
for f in files5:
        shutil.move(source5+f, dest1)
        
for f in files6:
        shutil.move(source6+f, dest1)

for f in files7:
        shutil.move(source7+f, dest1)
        
for f in files8:
        shutil.move(source8+f, dest1)
        
for f in files9:
        shutil.move(source9+f, dest1)
        
for f in files10:
        shutil.move(source10+f, dest1)

#remove the subfolders in train folder
rmdir(source1) 
rmdir(source2) 
rmdir(source3)
rmdir(source4) 
rmdir(source5) 
rmdir(source6) 
rmdir(source7) 
rmdir(source8) 
rmdir(source9)
rmdir(source10)


# to move the files from subfolders to "test", name the sources and the destination

source_1 = 'C:/Users/krupa krishnamurthy/split-folders/output/val/isolation/'
files_1 = os.listdir(source_1)
source_2 = 'C:/Users/krupa krishnamurthy/split-folders/output/val/laughing/'
files_2 = os.listdir(source_2)
source_3 = 'C:/Users/krupa krishnamurthy/split-folders/output/val/gossiping/'
files_3 = os.listdir(source_3)
source_4 = 'C:/Users/krupa krishnamurthy/split-folders/output/val/slapping/'
files_4 = os.listdir(source_4)
source_5 = 'C:/Users/krupa krishnamurthy/split-folders/output/val/stabbing/'
files_5 = os.listdir(source_5)
source_6 = 'C:/Users/krupa krishnamurthy/split-folders/output/val/quarrel/'
files_6 = os.listdir(source_6)
source_7 = 'C:/Users/krupa krishnamurthy/split-folders/output/val/strangle/'
files_7 = os.listdir(source_7)
source_8 = 'C:/Users/krupa krishnamurthy/split-folders/output/val/pullinghair/'
files_8 = os.listdir(source_8)
source_9 = 'C:/Users/krupa krishnamurthy/split-folders/output/val/punching/'
files_9 = os.listdir(source_9)
source_10 = 'C:/Users/krupa krishnamurthy/split-folders/output/val/nonbullying/'
files_10 = os.listdir(source_10)


dest_1 = 'C:/Users/krupa krishnamurthy/split-folders/output/val/'


#move files from subfodlers to the test folder (parent)
for f in files_1:
        shutil.move(source_1+f, dest_1)
for f in files_2:
        shutil.move(source_2+f, dest_1)
for f in files_3:
        shutil.move(source_3+f, dest_1)
for f in files_4:
        shutil.move(source_4+f, dest_1)
for f in files_5:
        shutil.move(source_5+f, dest_1)
for f in files_6:
        shutil.move(source_6+f, dest_1)
for f in files_7:
        shutil.move(source_7+f, dest_1)
for f in files_8:
        shutil.move(source_8+f, dest_1)
for f in files_9:
        shutil.move(source_9+f, dest_1)
for f in files_10:
        shutil.move(source_10+f, dest_1)
        

        
#remove the subfolders in test folder
rmdir(source_1) 
rmdir(source_2) 
rmdir(source_3) 
rmdir(source_4) 
rmdir(source_5) 
rmdir(source_6) 
rmdir(source_7) 
rmdir(source_8) 
rmdir(source_9) 
rmdir(source_10)

