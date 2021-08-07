def plot_loss(history):
  plt.plot(np.arange(0, N), history.history['loss'], label='training loss')
  plt.plot(np.arange(0, N), history.history['val_loss'], label='validation loss')
  plt.title('Loss')
  plt.xlabel('No. of Epochs')
  plt.ylabel('loss value')
  plt.legend()
  plt.show()

def plot_accuracy(history):
  plt.plot(history.history['accuracy'], label='training accuracy')
  plt.plot(history.history['val_accuracy'], label='validation accuracy')
  plt.title('Accuracy')
  plt.xlabel('No. of Epochs')
  plt.ylabel('accuracy value')
  plt.legend()
  plt.show()

plot_loss(history)
plt.savefig("plotloss.png")

plot_accuracy(history)
plt.savefig("plotaccuracy.png")