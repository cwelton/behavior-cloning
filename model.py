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
DATADIR = os.path.abspath('../cloning_data')
SWERVEDIR = os.path.abspath('../cloning_data_swerve')
LOGFILE = 'driving_log.csv'
MODELNAME = 'model.h5'

# Configuration Parameters
doflip = True
useside = True
sidesteer = 0.20

class Record(object):
    """An input data record captured from the input files
    
    Each line of the csv file has a center, left, and right camera image as well
    as the steering wheel angle from the human data capture.
    """
    def __init__(self, line):
        self.center = line[0]
        self.left = line[1]
        self.right = line[2]
        self.angle = float(line[3])

def load_data(location):
    """Load a csv datafile from the given directory"""
    with open(os.path.join(location, LOGFILE)) as f:
        reader = csv.reader(f)
        next(reader, None)  # skip the headers
        return [Record(line) for line in reader]

def bucketize_data(records):
    buckets = [[] for i in range(3)]
    for record in records:
        if abs(record.angle) < 0.06:
            buckets[0].append(record)
        elif abs(record.angle) < 0.16:
            buckets[1].append(record)
        else:
            buckets[2].append(record)
    return [shuffle(b) for b in buckets]

def doplot(records):
    import matplotlib.pyplot as plt
    steer = [int(10*abs(record.angle)) for record in records]
    plt.hist(steer, bins='auto')
    plt.show()

def image_angle(record):
    flip = [False, True][np.random.randint(2)]
    camera = np.random.randint(3)
    
    name = [record.center, record.left, record.right][camera]
    if name is None:
        name = record.center
        
    image = cv2.imread(name)
    angle = record.angle
    angle += [0,sidesteer,-sidesteer][camera]
    if flip:
        image = np.fliplr(image)
        angle = -angle
    return (image, angle)

def datagen(buckets, batch_size=30):
    i = 0
    assert(batch_size % len(buckets) == 0)
    while True:
        images = []
        angles = []
        while (len(images) < batch_size):
            for b in range(len(buckets)):
                record = buckets[b][i % len(buckets[b])]
                image, angle = image_angle(record)
                images.append(image)
                angles.append(angle)

            i += 1
            yield (np.array(images), np.array(angles))

def validgen(records, batch_size=32):
    num_samples = len(records)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_samples = records[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                image, angle = image_angle(batch_sample)
                images.append(image)
                angles.append(angle)

            yield (np.array(images), np.array(angles))
                

def main():

    """
    There are two input data sets:
    1. DATADIR   - "good" human driving
    2. SWERVEDIR - human driving with almost no straight driving
 
    Mostly we want to use 1, however 2 has some interesting features in that
    it has viewing angles that do not show up in the "good" human driving set
    and these are important to get out of bad situations.  However the only
    data points we are actually interested in are where the steering wheel
    changes direction, and then we need to scale it down because the dataset
    oversteers constantly.
    """
    data = load_data(DATADIR)
    swerve = load_data(SWERVEDIR)
    prev = 0
    for record in swerve:
        if prev*record.angle < 0:
            record.left = None
            record.right = None
            record.angle *= 0.8     # dataset over compensates
            data.append(record)

        prev = record.angle
        
    """
    Allocate the training data into buckets based on steering angle and create
    a streaming data generator from it.  
    
    The generator will ensure an even distribution of data across each of the 
    angle buckets despite the buckets not being even depth.

    The generator will also randomly flip images/angles or inject images from
    the the side cameras.
    """
    train, test = train_test_split(data, test_size=0.15)
    train_buckets = bucketize_data(train)
    train_generator = datagen(train_buckets)
    test_generator = validgen(test)

    print("Training Size {}".format(len(train)))
    print("  buckets {}".format([len(b) for b in train_buckets]))
    print("Validation Size {}".format(len(test)))

    """
    Load or create the model.  In this case we are using a slightly modified
    version of the NVIDEA model, with added dropout layers to increase robustness
    of the data.
    """
    if os.path.exists(MODELNAME):
        model = load_model(MODELNAME)
    else:
        (row, col, ch) = (160, 320, 3)
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

    """
    Train and save the model
    """
    for i in range(10):
        model.fit_generator(train_generator,
                            validation_data=test_generator,
                            samples_per_epoch=30*100,
                            nb_val_samples=len(test),
                            initial_epoch=5*i,
                            nb_epoch=5*(i+1))
        model.save("{}.{}".format(MODELNAME, i))

if __name__ == '__main__':
    main()
