#unsere main datei
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

url = "https://myshare.leuphana.de/?t=dde5dfe5773fb088bd895a74b49933ab"
tf.keras.utils.get_file("For_model", url, cache_dir="dataset", extract=True)

