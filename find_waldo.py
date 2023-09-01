import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import optimizers
import keras_cv
from pathlib import Path
import numpy as np
from keras_cv import bounding_box
from keras_cv import utils
import os
import resource
from keras_cv import visualization
import tqdm
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
BATCH_SIZE = 1
"""
pretrained_model = keras_cv.models.RetinaNet.from_preset(
    "retinanet_resnet50_pascalvoc", bounding_box_format="xywh"
)

filepath = tf.keras.utils.get_file(origin="https://i.imgur.com/gCNcJJI.jpg")
image = keras.utils.load_img(filepath)
image = np.array(image)


inference_resizing = keras_cv.layers.Resizing(
    640, 640, pad_to_aspect_ratio=True, bounding_box_format="xywh"
)

image_batch = inference_resizing([image])

class_ids = [
    "Aeroplane",
    "Bicycle",
    "Bird",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cow",
    "Dining Table",
    "Dog",
    "Horse",
    "Motorbike",
    "Person",
    "Potted Plant",
    "Sheep",
    "Sofa",
    "Train",
    "Tvmonitor",
    "Total",
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))
print(image_batch.shape)
y_pred = pretrained_model.predict(image_batch)
print(y_pred)
"""

inference_resizing = keras_cv.layers.Resizing(
    1280, 1920, pad_to_aspect_ratio=True, bounding_box_format="xyxy"
)

class_ids = [ 
    "Waldo"
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))


def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):
    inputs = next(iter(inputs.take(1)))
    images = inputs["images"]
    print(utils.to_numpy(images).shape)
    bounding_boxes = inputs["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=10,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping,
    )


def unpackage_raw(inp, bounding_box_format):
    dataset = None
    for ind, row in inp.iterrows():
        image = keras.utils.load_img(row['filename'])
        image = np.array(image)
        sf = min((1280 / image.shape[-3]), (1920/image.shape[-2]))
        image = inference_resizing(image)
        data = {
            "bounding_boxes":{
                "classes": tf.cast(0, dtype = tf.float32),
                "boxes": tf.cast([row['xmin'] * sf, row['ymin'] * sf, row['xmax']*sf, row['ymax']*sf], dtype = tf.float32),
            },
            "images": tf.cast(image, dtype = tf.float32)
        }  
        data["bounding_boxes"]["boxes"] = tf.reshape(data["bounding_boxes"]["boxes"], [1, 1, 1, 4])
        data["bounding_boxes"]["classes"] = tf.reshape(data["bounding_boxes"]["classes"], [1, 1, 1])
        data["images"] = tf.reshape(data["images"], [1, 1, 1280, 1920, 3])
        nds =  tf.data.Dataset.from_tensor_slices(data)
        if dataset is None:
            dataset = nds
        else:
            dataset = dataset.concatenate(nds)
    return dataset

def load_data_from_csv(filename, batch_size = BATCH_SIZE, bounding_box_format = "xyxy"):
    csv = pd.read_csv(filename)
    csv = unpackage_raw(csv, bounding_box_format)
    return csv



train_ds = load_data_from_csv('annotations.csv')

"""
train_ds["bounding_boxes"]["boxes"] = tf.reshape(train_ds["bounding_boxes"]["boxes"], [1, 1, 1, 4])
train_ds["bounding_boxes"]["classes"] = tf.reshape(train_ds["bounding_boxes"]["classes"], [1, 1, 1])
train_ds["images"] = tf.reshape(train_ds["images"], [1, 1, 1280, 1920, 3])
print(train_ds["bounding_boxes"]["boxes"].shape) #needs to be (1, 1, 4)
print(train_ds["bounding_boxes"]["classes"].shape) # needs to be (1, 1)
print(train_ds["images"].shape) # should be (1, w, h, 3)


tensor = tf.data.Dataset.from_tensor_slices(train_ds)
"""

visualize_dataset(
    train_ds, bounding_box_format="xyxy", value_range=(0, 255), rows=1, cols=1
)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
def dict_to_tuple(inputs):
    return inputs["images"], bounding_box.to_dense(
        inputs["bounding_boxes"], max_boxes=32
    )

train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
one = next(iter(train_ds.take(1)))
print(train_ds.take(1))

class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xyxy",
            # passing 1e9 ensures we never evaluate until
            # `metrics.result(force=True)` is
            # called.
            evaluate_freq=1e9,
        )

    def on_epoch_end(self, epoch, logs):
        self.metrics.reset_state()
        for batch in tqdm.tqdm(self.data):
            images, y_true = batch[0], batch[1]
            y_pred = self.model.predict(images, verbose=0)
            self.metrics.update_state(y_true, y_pred)

        metrics = self.metrics.result(force=True)
        logs.update(metrics)
        return logs
       
base_lr = 0.005
# including a global_clipnorm is extremely important in object detection tasks
optimizer = tf.keras.optimizers.SGD(
    learning_rate=base_lr, momentum=0.9, global_clipnorm=10.0
)

model = keras_cv.models.RetinaNet.from_preset(
    "resnet50_imagenet",
    num_classes=len(class_mapping),
    # For more info on supported bounding box formats, visit
    # https://keras.io/api/keras_cv/bounding_box/
    bounding_box_format="xyxy",
)
model.compile(
    classification_loss="focal",
    box_loss="smoothl1",
    optimizer=optimizer,
    # We will use our custom callback to evaluate COCO metrics
    metrics=None,
)
model.fit(
    train_ds,
    validation_data = train_ds, # VERY BAD PRACTICE 
    epochs = 10,
)
#model.save('model.keras') # how does loading work?

def visualize_detections(model, dataset, bounding_box_format):
    images, y_true = next(iter(dataset.take(1)))
    y_pred = model.predict(images)
    y_pred = bounding_box.to_ragged(y_pred)
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=1,
        cols=1,
        show=True,
        font_scale=0.7,
        class_mapping=class_mapping,
    )


visualize_detections(model, train_ds, "xyxy")

"""
visualize_dataset(
    train_ds, bounding_box_format="xyxy", value_range=(0, 255), rows=1, cols = 1
)
#"""