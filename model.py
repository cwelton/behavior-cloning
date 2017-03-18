#!/usr/bin/env python
import csv
import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout


### Load Data
datadir = os.path.abspath('../cloning_data')
#datadir = os.path.abspath('examples/data')
logfile = 'driving_log.csv'
modelname = 'model.h5'

with open(os.path.join(datadir, logfile)) as f:
    reader = csv.reader(f)
    next(reader, None)  # skip the headers
    lines = shuffle([line for line in reader])

#m = sorted([float(line[3]) for line in lines])
#test1 = os.path.join(datadir, 'IMG/center_2017_03_16_23_19_20_991.jpg')
#test2 = os.path.join(datadir, 'IMG/center_2017_03_16_23_10_35_237.jpg')
#test3 = os.path.join(datadir, 'IMG/center_2017_03_16_23_18_39_393.jpg')
# Select just 3 samples to start
#test = [line for line in lines if line[0] in (test1,test2,test3)]

#lines = [line for line in lines if abs(float(line[3])) > 0.02]

# Split input into histogram buckets
BUCKETS = [[],[],[],[]] 
for line in lines:
    BUCKETS[min(3, int(10*abs(float(line[3]))))].append(line)

print([len(b) for b in BUCKETS])

# Shuffle each bucket
for b in range(len(BUCKETS)):
    BUCKETS[b] = shuffle(BUCKETS[b])

#train, valid = train_test_split(lines, test_size=0.2)

def doplot():
    import matplotlib.pyplot as plt
    steer = [int(10*abs(float(line[3]))) for line in lines]
    plt.hist(steer, bins='auto')
    plt.show()




### Data generator
doflip = True
useside = True
sidesteer = 0.15
def image_angle(datum):
    flip = [False, True][np.random.randint(2)]
    camera = np.random.randint(3)
    
    name = datum[camera]
    image = cv2.imread(name)
    angle = float(datum[3])
    angle += [0,sidesteer,-sidesteer][camera]
    if flip:
        image = np.fliplr(image)
        angle = -angle
    return (image, angle)

def datagen(buckets, batch_size=32):
    i = 0
    assert(batch_size % len(buckets) == 0)
    while True:
        images = []
        angles = []
        while (len(images) < batch_size):
            for b in range(len(buckets)):
                datum = buckets[b][i % len(buckets[b])]
                image, angle = image_angle(datum)
                images.append(image)
                angles.append(angle)

            i += 1
            yield (np.array(images), np.array(angles))

train_generator = datagen(BUCKETS)


(row, col, ch) = (160, 320, 3)

if os.path.exists(modelname):
    model = load_model(modelname)
else:
    model = Sequential()

    # NVidia architecture
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.25))
    model.add(Dense(50))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')


model.fit_generator(train_generator,
                    samples_per_epoch=32*100,
                    nb_epoch=5)

model.save(modelname)
