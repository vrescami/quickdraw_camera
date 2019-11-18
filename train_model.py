import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

def show_img(img):
    plt.imshow(img.reshape(28,28))
    plt.show()

def load_dataset(classes=[], ratio=0.8):
    train_data = []
    test_data = []

    for word in classes:
        one_hot = np.zeros((len(classes), 1))
        one_hot[classes.index(word)] = 1.0

        dataset = np.load("dataset/"+word+".npy")
        separator = int(len(dataset)*ratio)
        train_inputs, test_inputs = dataset[0:separator], dataset[separator:]

        [train_data.append((x.reshape((784,1)), one_hot)) for x in train_inputs]
        [test_data.append((x.reshape((784,1)), one_hot)) for x in test_inputs]

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    return train_data, test_data

def cnn_model(num_classes):
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu', data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


classes = [
    "banana","house","axe","cup","door","duck","ear","eye","guitar","pizza",
    "alarm_clock","angel","ant","apple","arm","bandage","bat","beach","bed","bicycle",
    "bird","book","cactus","camera","candle","car","carrot","cello","circle","cookie",
    "crab","crayon","dog","donut","face","feather","fish","flamingo","flower","fork",
    "grapes","grass","hand","hat","hexagon","hot_dog","ice_cream","key","knife","octopus"
]

train_original, test_original = load_dataset(classes)

train_X_original = np.array([t[0] for t in train_original])
train_y_original = np.array([t[1] for t in train_original])

test_X_original = np.array([t[0] for t in test_original])
test_y_original = np.array([t[1] for t in test_original])

train_X = train_X_original.reshape(train_X_original.shape[0], 1, 28, 28)
train_y = train_y_original.reshape(train_y_original.shape[0], train_y_original.shape[1])

test_X = test_X_original.reshape(test_X_original.shape[0], 1, 28, 28)
test_y = test_y_original.reshape(test_y_original.shape[0], test_y_original.shape[1])

np.random.seed(1234)
model_cnn = cnn_model(len(classes))
model_cnn.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=10, batch_size=200)
scores = model_cnn.evaluate(test_X, test_y, verbose=0)
print('Final CNN accuracy: ', scores[1])

model_cnn.save("working_model.h5")

for i in range(10):
    prediction = model_cnn.predict_classes(np.array([test_X[i]]))[0]
    print("Expected:", [classes[x] for x in range(len(classes)) if test_y[i][x] == 1.][0])
    print("Prediction:", classes[prediction])
    show_img(test_X_original[i])
