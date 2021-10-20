import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from get_data import extract_donnees

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops

img_dataset = extract_donnees()

img_dataset_train = img_dataset[:750]
img_dataset_test = img_dataset[750 : -1]

print(img_dataset_train)