import ast
from tensorflow import keras
import math
import numpy
from pyclustering.cluster.gmeans import gmeans
from pyclustering.utils import read_sample
import operator
import os, ast
import tensorflow as tf
import time
def get_compiled_model(ANN, size, num_class):
    CNN = ['conv1d', 'conv2d', 'conv3d', 'maxpool1d', 'maxpool2d', 'maxpool3d', 'avgpool1d', 'avgpool2d',
           'avgpool3d', 'prelu', 'relu', 'relu6', 'batchnorm1d', 'batchnorm2d', 'batchnorm3d', 'dropout',
           'dropout2d', 'dropout3d', 'softmax', 'softmax2d', 'relu', 'conv1d', 'conv2d', 'conv3d', 'avg_pool1d',
           'avg_pool2d', 'avg_pool3d', 'max_pool1d', 'max_pool2d', 'max_pool3d', 'batch_norm', 'normalize',
           'linear', 'dropout', 'dropout2d', 'dropout3d', 'Linear', 'elu', 'tanh', 'softmax', 'sigmoid', 'Module',
           'relu6', 'flatten']
    ann = open(ANN, 'r')
    layers = ann.readlines()
    model = keras.Sequential()
    conv_counter = 0
    num_linear = 0
    linear_counter = 0
    for layer in layers:
        layer = ast.literal_eval(layer.strip())
        if 'func' in layer:
            layer['func'] = layer['func'].lower()
            if layer['func'] in CNN:
                if 'linear' in layer['func']:
                    num_linear += 1

    for layer in layers:
        layer = ast.literal_eval(layer.strip())
        if 'func' in layer:
            layer['func'] = layer['func'].lower()
            if layer['func'] in CNN:
                if 'conv' in layer['func']:
                    conv_counter += 1
                    if conv_counter == 1:
                        model.add(keras.layers.Conv2D(filters=layer['arg2'],
                                                      kernel_size=(layer['kernel_size'][0], layer['kernel_size'][0]),
                                                      input_shape=(size)))
                    else:
                        model.add(keras.layers.Conv2D(filters=layer['arg2'],
                                                      kernel_size=(layer['kernel_size'][0], layer['kernel_size'][0])))
                elif 'relu' in layer['func']:
                    if linear_counter == 0 and conv_counter > 1:
                        model.add(keras.layers.BatchNormalization())
                    model.add(keras.layers.Activation(keras.activations.relu))
                elif 'tanh' in layer['func']:
                    if linear_counter == 0 and conv_counter > 1:
                        model.add(keras.layers.BatchNormalization())
                    model.add(keras.layers.Activation(keras.activations.relu))
                elif "maxpool2d" in layer['func']:
                    model.add(keras.layers.MaxPool2D())
                elif 'flatten' in layer['func']:
                    model.add(keras.layers.GlobalAveragePooling2D())
                elif 'linear' in layer['func']:
                    linear_counter += 1
                    if linear_counter == num_linear:
                        model.add(keras.layers.Dense(num_class))
                    else:
                        model.add(keras.layers.Dense(layer['arg2']))
                elif 'dropout' in layer['func']:
                    model.add(keras.layers.Dropout(layer['arg1']))
                elif 'softmax' in layer['func']:
                    model.add(keras.layers.Activation(keras.activations.softmax))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def distance(x1, y1, x2, y2):
    return numpy.math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))


def gmeans_clustering(path_sample):
    # Read sample from file.
    sample = read_sample(path_sample)

    # Create instance of G-Means algorithm.
    # By default algorithm starts search from a single cluster.
    gmeans_instance = gmeans(sample, repeat=5).process()
    clusters = gmeans_instance.get_clusters()

    return clusters

# Cluster the models in database into small clusters
def model_clustering(meta_info, input_shape, n_classes, dimension_list={}):
    path_list = []
    temp_array = []
    # calculate the distance
    for r, d, f in os.walk(meta_info):
        for file in f:
            model = open(os.path.join(r, file), "r", encoding="ISO-8859-1")
            candidate_coor = ast.literal_eval(model.readlines()[-1])
            dist = distance(input_shape[0], input_shape[1], candidate_coor[0], candidate_coor[1])
            temp_array.append([dist, abs(candidate_coor[2] - n_classes)])
            dimension_list.update({os.path.join(r, file): [dist, abs(candidate_coor[2] - n_classes)]})

    sorted_a = sorted(temp_array)
    string = ''
    for i in sorted_a:
        string += str(i[0]) + " " + str(i[1]) + "\n"
    f = open("dist.data", "w")
    f.write(string)
    f.close()
    clusters = gmeans_clustering("dist.data")
    sorted_d = sorted(dimension_list.items(), key=operator.itemgetter(1))
    count_link = 0
    for i in clusters:
        if 1 in i:
            for link, key in sorted_d:
                path_list.append(link)
                count_link += 1
                if count_link == len(i):
                    break
            break
    return path_list


from tensorflow import keras
from keras.utils import np_utils
import numpy as np # linear algebra

# load data
x_train = np.load('xtrain_mala.npy')
y_train = np.load('ytrain_mala.npy')
x_test = np.load('xtest_mala.npy')
y_test = np.load('ytest_mala.npy')
y_train=np_utils.to_categorical(y_train,2)
y_test=np_utils.to_categorical(y_test,2)

#matching the models based on user's intent
model_list = []
if x_train.shape[3] == 3:
    model_list = model_clustering("models\\3_10", input_shape = x_train.shape[1:3], n_classes = y_train.shape[1])
elif x_train.shape[3] == 1:
    model_list = model_clustering("models\\1_10", input_shape=x_train.shape[1:3], n_classes=y_train.shape[1])

# Open a strategy scope.
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    model = get_compiled_model(model_list[0], x_train.shape[1:4], y_train.shape[1])


print(f'SHAPE OF TRAINING IMAGE DATA : {x_train.shape}')
print(f'SHAPE OF TESTING IMAGE DATA : {x_test.shape}')
print(f'SHAPE OF TRAINING LABELS : {y_train.shape}')
print(f'SHAPE OF TESTING LABELS : {y_test.shape}')

start = time.time()
model.fit(x_train,y_train,batch_size=50,epochs=20,verbose=1)
end = time.time()
print(end - start)
predictions = model.evaluate(x_test, y_test)
print(f'LOSS : {predictions[0]}')
print(f'ACCURACY : {predictions[1]}')