やさしいディープラーニング演習
----

(2019.03.13 谷岡)

# はじめに

この資料は、やさしいディープラーニング演習で環境設定とPythonの演習までが終了した後、
実際に手を動かしてディープラーニングを体験するカリキュラムについて説明したものです。
大きく分けて以下の4つの内容が用意されています。

1. 犬猫分類プログラム
2. Sequencialモデルの使い方
3. 手書き数字認識プログラム
4. 犬猫学習プログラム

# 犬猫分類プログラム

犬と猫の画像を分類するプログラムを実行してみます。

配布した資料の中の data.zip を解凍して、train というフォルダと validation という
フォルダを、Anaconda のフォルダの、今回作った keras フォルダ以下に展開します。
私の環境の場合は、以下のようになります。

C:/Users/taniokah/keras/
├─train
│  ├─cats
│  ├─dogs
└─validation
    ├─cats
    └─dogs

次に、以下の、```python と ``` で囲まれたプログラムをJupyter Notebookに書き込んで、
実行してみましょう。

```python
# coding: utf-8
# Dogs vs. Cats

# 必要なライブラリの読込
%matplotlib inline
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys

# 学習済みモデルVGG16の読込
model = VGG16(weights='imagenet')

# 画像判定のための関数
def predict(filename, featuresize):
    img = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(preprocess_input(x))
    results = decode_predictions(preds, top=featuresize)[0]
    return results

# 画像表示のための関数
def showimg(filename, title, i):
    im = Image.open(filename)
    im_list = np.asarray(im)
    plt.subplot(2, 5, i)
    plt.title(title)
    plt.axis("off")
    plt.imshow(im_list)

# 画像を判定
filename = "train/cats/cat.3591.jpg"
plt.figure(figsize=(20, 10))
for i in range(1):
    showimg(filename, "query", i+1)
plt.show()
results = predict(filename, 10)
for result in results:
    print(result)

# 画像を判定
filename = "train/dogs/dog.8035.jpg"
plt.figure(figsize=(20, 10))
for i in range(1):
    showimg(filename, "query", i+1)
plt.show()
results = predict(filename, 10)
for result in results:
    print(result)
```

このプログラムでは、すでに学習済みの VGG16 というモデルをダウンロードして、
犬と猫を識別します。

# Sequencialモデルの使い方

次に、新しいモデルを設計して、自分で学習してみるために、Sequencialモデルという
ライブラリを使ってみます。Keras では、Sequencialモデルを使って、ニューラル
ネットワークを構築していくことになります。

```python
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
import numpy as np

# ダミーデータ
data = np.random.random((1000, 784))
labels = np.random.randint(10, size=(1000, 1))
labels = np_utils.to_categorical(labels, 10)

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=784))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# モデルのコンパイル
model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# 学習を行う
model.fit(data, labels)
```

このサンプルプログラムの中で、注目するべきは、model = Sequencial() として
空のモデルを作っているところと、そこの model.add(...) としてニューラル
ネットワークの層を追加して行っているところです。

その後、model.compile(...) という関数でニューラルネットワークのコンパイル、
つまりネットワーク同士のつなぎこみなどを行い、最後に model.fit() でdataと
labelsに対して学習を行います。（このデータはランダムなので学習は収束しません）

# 手書き数字認識プログラム

今度は、Sequencialモデルで3層のニューラルネットワークを構築し、MNISTという
手書きの数字を分類するデータセットを学習してみます。

```python
# coding: utf-8
# MNISTサンプル

# Kerasをインポート
import keras

# MINISTのデータの他、必要なモジュールをインポート
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, CSVLogger
%matplotlib inline
import matplotlib.pyplot as plt

# バッチサイズ、クラス数、エポック数を定義
batch_size = 128
num_classes = 10
epochs = 20

# MNISTデータを読込
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# MNISTデータのうち10枚だけ表示
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.title("M_%d" % i)
    plt.axis("off")
    plt.imshow(x_train[i].reshape(28, 28), cmap=None)
plt.show()

# 画像サイズを正規化
x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 確認のために表示
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# モデルを作成
model = Sequential()
model.add(Dense(512, input_shape=(784, )))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

# サマリーを出力
model.summary()

# モデルのコンパイル
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=2)
csv_logger = CSVLogger('training.log')
hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1,
                 validation_split=0.1,
                 callbacks=[es, csv_logger])

# 学習を実行
score = model.evaluate(x_test, y_test, verbose=0)
print('test loss:', score[0])
print('test acc:', score[1])


# 学習結果を表示
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = len(loss)
plt.plot(range(epochs), loss, marker='.', label='loss(training data)')
plt.plot(range(epochs), val_loss, marker='.', label='val_loss(evaluationdata)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```

まず Keras が公開している MNIST データをダウンロードし、次に Sequencial で
ニューラルネットワークを構築します。その後、学習を実行し、学習結果の表示までを
実行します。

# 犬猫学習プログラム

さて、最初に犬と猫に分類するプログラムを紹介しましたが、最後に自分でニューラル
ネットワークを構築して学習と分類を実行してみます。

```python
# https://medium.com/@parthvadhadiya424/hello-world-program-in-keras-with-cnn-dog-vs-cat-classification-efc6f0da3cc5

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout

# dimensions of our images.
img_width, img_height = 150, 150

# Initialising the CNN
model = Sequential()

# Convolution
model.add(Conv2D(32, (3, 3), input_shape = (img_width, img_height, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2))) # Pooling

# Second convolutional layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Third convolutional layer
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
model.add(Flatten())

# Full connection
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

```

ここまでが、CNN(Convolutional Neural Network) と呼ばれるニューラルネットワーク
のモデルの構築の一例です。

さらに、学習と分類のプログラムを作ります。

```python
#https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
from keras.preprocessing.image import ImageDataGenerator
%matplotlib inline
import matplotlib.pyplot as plt

# dimensions of our images.
#img_width, img_height = 50, 50

train_data_dir = 'train'
validation_data_dir = 'validation'
nb_train_samples = 1000
nb_validation_samples = 1000
epochs = 50
batch_size = 16

train_data = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_data = ImageDataGenerator(rescale = 1./255)

train_generator = train_data.flow_from_directory(
    train_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'binary')

validation_generator = test_data.flow_from_directory(
    validation_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'binary')

print(nb_validation_samples // batch_size)

hist = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)

# 学習結果を表示
loss = hist.history['loss']
acc = hist.history['acc']
val_loss = hist.history['val_loss']
val_acc = hist.history['val_acc']
epochs = len(loss)
plt.plot(range(epochs), loss, marker='.', label='loss(training data)')
plt.plot(range(epochs), acc, marker='.', label='acc(training data)')
plt.plot(range(epochs), val_loss, marker='.', label='val_loss(evaluationdata)')
plt.plot(range(epochs), val_acc, marker='.', label='val_acc(evaluationdata)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss/acc')
plt.show()

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("saved model..! ready to go.")

```

さて、数字だけ見ても、どの程度のものかわかりにくいので、画像ごとに判定してみます。

```python
#https://github.com/parthvadhadiya/classify_dogs-vs-cats_using_keras/blob/master/use_model.py

from keras.models import model_from_json
from keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('model.h5')
print("Loaded model from disk")

'''loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
'''
loaded_model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# dimensions of our images.
#img_width, img_height = 50, 50

# 画像表示のための関数
def showimg(filename, title, i):
    im = Image.open(filename)
    im_list = np.asarray(im)
    plt.subplot(2, 5, i)
    plt.title(title)
    plt.axis("off")
    plt.imshow(im_list)

# 画像判定のための関数
def predictimg(filename, featuresize):
    img = image.load_img(filename, target_size = (img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    #preds = model.predict(preprocess_input(x))
    preds = model.predict(x)
    #results = decode_predictions(preds, top = featuresize)[0]
    #return results
    return preds

# 画像を判定
def predict(filename, label):
    results = predictimg(filename, 10)
    print(results)
    if results[0][0] == 1:
        label += ' = dog'
    else:
        label += ' = cat'
    plt.figure(figsize = (20, 10))
    showimg(filename, label, 1)
    plt.show()

predict('validation/cats/cat.1000.jpg', 'cat')
predict('validation/cats/cat.1001.jpg', 'cat')
predict('validation/cats/cat.1002.jpg', 'cat')
predict('validation/cats/cat.1003.jpg', 'cat')
predict('validation/cats/cat.1004.jpg', 'cat')
predict('validation/cats/cat.1005.jpg', 'cat')
predict('validation/cats/cat.1006.jpg', 'cat')
predict('validation/cats/cat.1007.jpg', 'cat')
predict('validation/cats/cat.1008.jpg', 'cat')
predict('validation/cats/cat.1009.jpg', 'cat')

predict('validation/dogs/dog.1000.jpg', 'dog')
predict('validation/dogs/dog.1001.jpg', 'dog')
predict('validation/dogs/dog.1002.jpg', 'dog')
predict('validation/dogs/dog.1003.jpg', 'dog')
predict('validation/dogs/dog.1004.jpg', 'dog')
predict('validation/dogs/dog.1005.jpg', 'dog')
predict('validation/dogs/dog.1006.jpg', 'dog')
predict('validation/dogs/dog.1007.jpg', 'dog')
predict('validation/dogs/dog.1008.jpg', 'dog')
predict('validation/dogs/dog.1009.jpg', 'dog')

```

ただし、このサンプルプログラムではなかなかうまく分類できないと思います。
さらにもっと精度を高められるように、工夫してみたり、ネットや文献を参考に
勉強してみてください。

＊できるだけ手元の環境でテスト実行しましたが、もしエラーがでたらご連絡ください。
