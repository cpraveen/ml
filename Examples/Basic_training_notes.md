# Basic parts of the Tensorflow training script


## Important libraries
The main Tensorflow, Keras and Numpy libraries need to be loaded as

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
```

Additional helper libraries are also required to visualize data

```python
%matplotlib inline  # Needed for ipython notebooks
import matplotlib
import matplotlib.pyplot as plt  
```

## Reproducibility
To ensure reproducibility of results, we need to seed various pseudorandom number generators (see [here](https://stackoverflow.com/a/52897216) for a discussion on this)

```python
import random
import os
def seed_random_number(seed):
  np.random.seed(seed)
  tf.set_random_seed(seed)
  os.environ['PYTHONHASHSEED']=str(seed)
  random.seed(seed)

random_seed = 1
seed_random_number(random_seed)
```

## Loading data
Load the training and testing data sets

```python
# Each row corresponds to data for a single sample
Train_Input = np.load('Data/Train_Inputs.npy')
Train_Label = np.load('Data/Train_Labels.npy')
Test_Input  = np.load('Data/Train_Inputs.npy')
Test_Label  = np.load('Data/Train_Labels.npy')
Ntrain      = len(Train_Input[:,0])
Ntest       = len(Test_Input[:,0])
```

At times, it is necessary to scale the input data to ensure proper generalizability of the network

```python
# Scaling input using user defined function
Train_Input = MyScale(Train_Input)
Test_Input  = MyScale(Test_Input)
```

## Creating and compiling an MLP model
To create a feed-network, we need to define the number of layers, the width of each layer and the activation functions. We use `keras.Sequential` to configure the model. For example, 

```python
model = keras.Sequential([
			keras.layers.Dense(128, input_shape=(10,), activation=tf.nn.relu),
			keras.layers.Dense(10, activation=tf.nn.softmax)
       ])
```
declares an **classification** MLP with a single hidden layer which takes an input of size `(*,10)`, has 128 hidden units and 10 output units. Furthemore, the ReLU activation function is used in the hidden layer. The layers in the network are added using 

```python
tf.layers.Dense(
		units,
		input_shape,   
		activation=None,
		use_bias=True,
		kernel_initializer=None,
		bias_initializer=tf.zeros_initializer(),
		kernel_regularizer=None,
		bias_regularizer=None,
	 )
```
where

* `units`: Positive integer, dimensionality of the output space.
* `input_shape`: Shape of input array
* `activation`: Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
* `use_bias`: Boolean, whether the layer uses a bias vector.
* `kernel_initializer`: Initializer for the kernel weights matrix.
* `bias_initializer`: Initializer for the bias vector.
* `kernel_regularizer`: Regularizer function applied to the kernel weights matrix.
* `bias_regularizer`: Regularizer function applied to the bias vector.

Note that the argument `input_shape` is needed only in the first layer. There are a few more arguments/options that can be specified for `tf.layers.Dense`, which can be found in the Tensorflow [documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)

To compile the model, we use `tf.keras.Model.compile`

```python
compile(
	optimizer,
	loss=None,
	metrics=None,
)
```
where

* `optimizer`: String (name of optimizer) or optimizer instance. See [optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers).
* `loss`: String (name of objective function) or objective function. See [losses](https://www.tensorflow.org/api_docs/python/tf/losses). 
* `metrics`: List of metrics to be evaluated by the model during training and testing.

## Training the network
To train the network, we use `tf.keras.Model.fit`

```python
fit(
	x=None,
	y=None,
	batch_size=None,
	epochs=1,
	verbose=1,
	callbacks=None,
	validation_split=0.0,
	validation_data=None,
	shuffle=True,
	initial_epoch=0,
	)
```    
where

* `x`: Input data. It could be:
* `y`: Target data or labels. 
* `batch_size`: Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32. 
* `verbose`: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
* `callbacks`: List of keras.callbacks.Callback instances. List of callbacks to apply during training.
* `validation_split`: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling. 
* `validation_data`: Data on which to evaluate the loss and any model metrics at the end of each epoch. `validation_data` will override validation_split. validation_data could be: - tuple (x_val, y_val) of Numpy arrays or tensors - tuple (x_val, y_val, val_sample_weights) of Numpy arrays - dataset or a dataset iterator
* `shuffle`: Boolean (whether to shuffle the training data before each epoch)
* `initial_epoch`: Integer. Epoch at which to start training (useful for resuming a previous training run).

## Evaluation of metric and predictions
To evaluate the loss and metric on the test data, we use `tf.keras.Model.evaluate`

```python
evaluate(
	x=None,
	y=None,
	)
```  

To evaluate the network predcitions for a given input set, we use `tf.kera.Model.predict`

```python
predict(x)
```

  


