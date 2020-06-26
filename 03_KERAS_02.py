'''
This Program do classification task using neural network
NN from keras
'''
# =======================================================
# Import and preprocess data データを準備する
# =======================================================
import numpy as np
from keras.utils import to_categorical
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_class = 10

x_train = x_train.astype('float32')/ 255.0
x_test = x_test.astype('float32')/ 255.0

y_train = to_categorical(y_train, num_class)
y_test = to_categorical(y_test, num_class)

# =======================================================
# Neural Networks using keras's FUnctional API ニューラルネットワーク
# =======================================================

# Create Model  ニューラルネットワークを作成する

from keras.layers import Input, Flatten, Dense, Conv2D, BatchNormalization, LeakyReLU, Dropout, Activation
from keras.models import Model

input_layer = Input(shape= (32,32,3)) # image file is 32 x32 in 3 rgb channel

x = Conv2D(filters =32, kernel_size=3, strides =1, padding = 'same')(input_layer)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(filters =32, kernel_size=3, strides =2, padding = 'same')(input_layer)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(filters =64, kernel_size=3, strides =1, padding = 'same')(input_layer)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(filters =64, kernel_size=3, strides =2, padding = 'same')(input_layer)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Flatten()(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate = 0.5)(x)

x = Dense(num_class)(x)

output_layer = Activation('softmax')(x)

model = Model(input_layer,output_layer)

#model.summary() # for see summary of the model

# Compile the Model (Indicate 'Optimizer' and 'Loss function') オプティマイザーと学習率を特定する

from keras.optimizers import Adam

opt = Adam(lr = 0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train the Model ニューラルネットワークをトレーニングする

model.fit(x_train, y_train, batch_size=32, epochs=1, shuffle = True) #(input, realoutput)

# Evaluating the model ニューラルネットワークを評価する

model.evaluate(x_test, y_test)

# Visualizing the result using Matplotlib

classes = np.array(['airplane', 'automobile','bird','cat','deer','dog','frog','horse','ship','truck'])
preds = model.predict(x_test)
preds_single = classes[np.argmax(preds, axis = -1)]
actual_single = classes[np.argmax(y_test, axis = -1)]

import matplotlib.pyplot as plt

n_to_show = 10
indices = np.random.choice(range(len(x_test)), n_to_show)

fig = plt.figure(figsize=(15,3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis('off')
    ax.text(0.5, -0.35, 'predict=' + str(preds_single[idx]), fontsize = 10, ha='center', transform = ax.transAxes)
    ax.text(0.5, -0.7, 'actual=' + str(actual_single[idx]), fontsize = 10, ha='center', transform = ax.transAxes)
    ax.imshow(img)

plt.show()




