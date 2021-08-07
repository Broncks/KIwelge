#unsere main datei
import tensorflow as tf
#fmf
url = "https://myshare.leuphana.de/?t=dde5dfe5773fb088bd895a74b49933ab"
tf.keras.utils.get_file("For_model", url, cache_dir="dataset", extract=True)

