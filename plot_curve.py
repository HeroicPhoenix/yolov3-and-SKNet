import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('logs/train.log')
epoch = data['epoch']
loss = data['loss']
val_loss = data['val_loss']

plt.figure()

plt.plot(epoch, loss, label='train_loss')
plt.plot(epoch, val_loss, label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.title('loss')

plt.show()