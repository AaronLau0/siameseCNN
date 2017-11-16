from keras import backend as K
from keras.layers import Input, merge
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda
from keras.models import Sequential, Model
from keras.utils import np_utils
from random import shuffle
from scipy.misc import imresize
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import time


def get_random_image(img_groups, group_names, gid):
    gname = group_names[gid]
    photos = img_groups[gname]
    pid = np.random.choice(np.arange(len(photos)), size=1)[0]
    pname = photos[pid]
    return gname + pname + ".jpg"


def create_triples(image_dir):
    img_groups = {}
    for img_file in os.listdir(image_dir):
        prefix, suffix = img_file.split(".")
        gid, pid = prefix[0:4], prefix[4:]
        if gid in img_groups:  # img_groups.has_key(gid):
            img_groups[gid].append(pid)
        else:
            img_groups[gid] = [pid]
    pos_triples, neg_triples = [], []
    # positive pairs are any combination of images in same group
    for key in img_groups.keys():
        triples = [(key + x[0] + ".jpg", key + x[1] + ".jpg", 1)
                   for x in itertools.combinations(img_groups[key], 2)]
        pos_triples.extend(triples)
    # need equal number of negative examples
    group_names = list(img_groups.keys())
    for i in range(len(pos_triples)):
        g1, g2 = np.random.choice(np.arange(len(group_names)), size=2, replace=False)
        left = get_random_image(img_groups, group_names, g1)
        right = get_random_image(img_groups, group_names, g2)
        neg_triples.append((left, right, 0))
    pos_triples.extend(neg_triples)
    shuffle(pos_triples)
    return pos_triples


def load_image(image_name):
    if not image_name in image_cache:  # image_cache.has_key(image_name):
        image = plt.imread(os.path.join(IMAGE_DIR, image_name)).astype(np.float32)
        image = imresize(image, (224, 224))
        # image = imresize(image, (320, 243))
        image = np.divide(image, 128)
        image_cache[image_name] = image
    return image_cache[image_name]


def generate_image_triples_batch(image_triples, batch_size, shuffle=False):
    while True:
        # loop once per epoch
        if shuffle:
            indices = np.random.permutation(np.arange(len(image_triples)))
        else:
            indices = np.arange(len(image_triples))
        shuffled_triples = [image_triples[ix] for ix in indices]
        num_batches = len(shuffled_triples) // batch_size
        for bid in range(num_batches):
            # loop once per batch
            images_left, images_right, labels = [], [], []
            batch = shuffled_triples[bid * batch_size: (bid + 1) * batch_size]
            for i in range(batch_size):
                lhs, rhs, label = batch[i]
                images_left.append(load_image(lhs))
                images_right.append(load_image(rhs))
                labels.append(label)
            Xlhs = np.array(images_left)
            Xrhs = np.array(images_right)
            Y = np_utils.to_categorical(np.array(labels), num_classes=2)
            yield ([Xlhs, Xrhs], Y)


def create_base_network(input_shape):
    seq = Sequential()
    # STARY NEFUNKCNI
    # CONV => RELU => POOL
    # seq.add(Conv2D(20, kernel_size=5, padding="same", input_shape=input_shape))
    # seq.add(Activation("relu"))
    # seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # # CONV => RELU => POOL
    # seq.add(Conv2D(50, kernel_size=5, padding="same"))
    # seq.add(Activation("relu"))myCNN.py
    # seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # # added
    # # CONV => RELU => POOL
    # seq.add(Conv2D(50, kernel_size=5, padding="same"))
    # seq.add(Activation("relu"))
    # seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #NEJLEPSI NA YF1
    seq.add(Conv2D(20, kernel_size=5, padding="same", input_shape=input_shape))
    seq.add(Activation("relu"))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=None))
    # CONV => RELU => POOL
    seq.add(Conv2D(50, kernel_size=5, padding="same"))
    seq.add(Activation("relu"))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=None))
    # added
    # CONV => RELU => POOL
    seq.add(Conv2D(50, kernel_size=5, padding="same"))
    seq.add(Activation("relu"))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=None))

    # # TEST NA YF2
    # seq.add(Conv2D(20, kernel_size=5, padding="same", input_shape=input_shape))
    # seq.add(Activation("relu"))
    # seq.add(MaxPooling2D(pool_size=(8, 8), strides=None))
    # # CONV => RELU => POOL
    # seq.add(Conv2D(50, kernel_size=5, padding="same"))
    # seq.add(Activation("relu"))
    # seq.add(MaxPooling2D(pool_size=(4, 4), strides=None))
    # # added
    # # CONV => RELU => POOL
    # seq.add(Conv2D(50, kernel_size=5, padding="same"))
    # seq.add(Activation("relu"))
    # seq.add(MaxPooling2D(pool_size=(2, 2), strides=None))
    #
    # Flatten => RELU
    seq.add(Flatten())
    seq.add(Dense(500))

    return seq


def cosine_distance(vecs, normalize=False):
    x, y = vecs
    if normalize:
        x = K.l2_normalize(x, axis=0)
        y = K.l2_normalize(x, axis=0)
    return K.prod(K.stack([x, y], axis=1), axis=1)


def cosine_distance_output_shape(shapes):
    return shapes[0]


def compute_accuracy(preds, labels):
    return labels[preds.ravel() < 0.5].mean()


DATA_DIR = "../"
IMAGE_DIR = "../yalefaces/"  # os.path.join(DATA_DIR, "jpg")
MODEL_NAME = 'SiamCNN_Model_yalefaces_{}'.format(time.strftime("%d_%m_%Y_%H-%M-%S"))

triples_data = create_triples(IMAGE_DIR)

print("# image triples:", len(triples_data))
[x for x in triples_data[0:5]]

BATCH_SIZE = 16  # 64

split_point = int(len(triples_data) * 0.7)
triples_train, triples_test = triples_data[0:split_point], triples_data[split_point:]

input_shape = (224, 224, 3)
# input_shape = (320, 243, 3)
base_network = create_base_network(input_shape)

image_left = Input(shape=input_shape)
image_right = Input(shape=input_shape)

vector_left = base_network(image_left)
vector_right = base_network(image_right)

distance = Lambda(cosine_distance,
                  output_shape=cosine_distance_output_shape)([vector_left, vector_right])

fc1 = Dense(128, kernel_initializer="glorot_uniform")(distance)
fc1 = Dropout(0.2)(fc1)
fc1 = Activation("relu")(fc1)

pred = Dense(2, kernel_initializer="glorot_uniform")(fc1)
pred = Activation("softmax")(pred)

model = Model(inputs=[image_left, image_right], outputs=pred)
# model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

NUM_EPOCHS = 100

image_cache = {}
train_gen = generate_image_triples_batch(triples_train, BATCH_SIZE, shuffle=True)
val_gen = generate_image_triples_batch(triples_test, BATCH_SIZE, shuffle=False)

num_train_steps = len(triples_train) // BATCH_SIZE
num_val_steps = len(triples_test) // BATCH_SIZE

history = model.fit_generator(train_gen,
                              steps_per_epoch=num_train_steps,
                              epochs=NUM_EPOCHS,
                              validation_data=val_gen,
                              validation_steps=num_val_steps)

model.save(filepath=DATA_DIR + "trained_models/" + MODEL_NAME + ".h5")

plt.subplot(211)
plt.title("Loss")
plt.plot(history.history["loss"], color="r", label="train")
plt.plot(history.history["val_loss"], color="b", label="validation")
plt.legend(loc="best")

plt.subplot(212)
plt.title("Accuracy")
plt.plot(history.history["acc"], color="r", label="train")
plt.plot(history.history["val_acc"], color="b", label="validation")
plt.legend(loc="best")

plt.tight_layout()
plt.show()
plt.savefig(filepath=DATA_DIR + "trained_models/", filename=MODEL_NAME + ".png")
