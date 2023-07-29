# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Neural net models for tabular datasets."""

from typing import Union, List
import numpy as np
import tensorflow as tf

TfInput = Union[np.ndarray, tf.Tensor]


def exu(x, weight, bias):
  """ExU hidden unit modification."""
  return tf.exp(weight) * (x - bias)


# Activation Functions
def relu(x, weight, bias):
  """ReLU activation."""
  return tf.nn.relu(weight * (x - bias))


def relu_n(x, n = 1):
  """ReLU activation clipped at n."""
  return tf.clip_by_value(x, 0, n)


class ActivationLayer(tf.keras.layers.Layer):
  """Custom activation Layer to support ExU hidden units."""

  def __init__(self,
               num_units,
               name = None,
               activation = 'exu',
               trainable = True):
    """Initializes ActivationLayer hyperparameters.

    Args:
      num_units: Number of hidden units in the layer.
      name: The name of the layer.
      activation: Activation to use. The default value of `None` corresponds to
        using the ReLU-1 activation with ExU units while `relu` would use
        standard hidden units with ReLU activation.
      trainable: Whether the layer parameters are trainable or not.
    """
    super(ActivationLayer, self).__init__(trainable=trainable, name=name)
    self.num_units = num_units
    self._trainable = trainable
    if activation == 'relu':
      self._activation = relu
      self._beta_initializer = 'glorot_uniform'
    elif activation == 'exu':
      self._activation = lambda x, weight, bias: relu_n(exu(x, weight, bias))
      self._beta_initializer = tf.initializers.truncated_normal(
          mean=4.0, stddev=0.5)
    else:
      raise ValueError('{} is not a valid activation'.format(activation))

  def build(self, input_shape):
    """Builds the layer weight and bias parameters."""
    self._beta = self.add_weight(
        name='beta',
        shape=[input_shape[-1], self.num_units],
        initializer=self._beta_initializer,
        trainable=self._trainable)
    self._c = self.add_weight(
        name='c',
        shape=[1, self.num_units],
        initializer=tf.initializers.truncated_normal(stddev=0.5),
        trainable=self._trainable)
    super(ActivationLayer, self).build(input_shape)

  @tf.function
  def call(self, x):
    """Computes the output activations."""
    center = tf.tile(self._c, [tf.shape(x)[0], 1])
    out = self._activation(x, self._beta, center)
    return out


class FeatureNN(tf.keras.layers.Layer):
  """Neural Network model for each individual feature.

  Attributes:
    hidden_layers: A list containing hidden layers. The first layer is an
      `ActivationLayer` containing `num_units` neurons with specified
      `activation`. If `shallow` is False, then it additionally contains 2
      tf.keras.layers.Dense ReLU layers with 64, 32 hidden units respectively.
    linear: Fully connected layer.
  """

  def __init__(self,
               num_units,
               dropout = 0.5,
               trainable = True,
               shallow = True,
               feature_num = 0,
               name_scope = 'model',
               activation = 'exu'):
    """Initializes FeatureNN hyperparameters.

    Args:
      num_units: Number of hidden units in first hidden layer.
      dropout: Coefficient for dropout regularization.
      trainable: Whether the FeatureNN parameters are trainable or not.
      shallow: If True, then a shallow network with a single hidden layer is
        created, otherwise, a network with 3 hidden layers is created.
      feature_num: Feature Index used for naming the hidden layers.
      name_scope: TF name scope str for the model.
      activation: Activation and type of hidden unit(ExUs/Standard) used in the
        first hidden layer.
    """
    super(FeatureNN, self).__init__()
    self._num_units = num_units
    self._dropout = dropout
    self._trainable = trainable
    self._tf_name_scope = name_scope
    self._feature_num = feature_num
    self._shallow = shallow
    self._activation = activation

  def build(self, input_shape):
    """Builds the feature net layers."""
    self.hidden_layers = [
    ]
    if not self._shallow:
      self._h1 = tf.keras.layers.Dense(
          4,
          activation='sigmoid',
          use_bias=True,
          trainable=self._trainable,
          name='h1_{}'.format(self._feature_num),
          kernel_initializer='glorot_uniform')
    
      self.hidden_layers += [self._h1]
    self.linear = tf.keras.layers.Dense(
        1,
        use_bias=True,
        trainable=self._trainable,
        name='dense_{}'.format(self._feature_num),
        kernel_initializer='glorot_uniform')
    super(FeatureNN, self).build(input_shape)

  @tf.function
  def call(self, x, training):
    """Computes FeatureNN output with either evaluation or training mode."""
    with tf.name_scope(self._tf_name_scope):
      for l in self.hidden_layers:
        x = tf.nn.dropout(
            l(x), rate=tf.cond(training, lambda: self._dropout, lambda: 0.0))
      x = tf.squeeze(self.linear(x), axis=1)
    return x


class NAM(tf.keras.Model):
  """Neural additive model.

  Attributes:
    feature_nns: List of FeatureNN, one per input feature.
  """

  def __init__(self,
               num_inputs,
               num_units,
               trainable = True,
               shallow = True,
               feature_dropout = 0.0,
               dropout = 0.0,
               **kwargs):
    """Initializes NAM hyperparameters.

    Args:
      num_inputs: Number of feature inputs in input data.
      num_units: Number of hidden units in first layer of each feature net.
      trainable: Whether the NAM parameters are trainable or not.
      shallow: If True, then shallow feature nets with a single hidden layer are
        created, otherwise, feature nets with 3 hidden layers are created.
      feature_dropout: Coefficient for dropping out entire Feature NNs.
      dropout: Coefficient for dropout within each Feature NNs.
      **kwargs: Arbitrary keyword arguments. Used for passing the `activation`
        function as well as the `name_scope`.
    """
    super(NAM, self).__init__()
    self._num_inputs = num_inputs
    if isinstance(num_units, list):
      self._num_units = num_units
    elif isinstance(num_units, int):
      self._num_units = [num_units for _ in range(self._num_inputs)]
    self._trainable = trainable
    self._shallow = shallow
    self._feature_dropout = feature_dropout
    self._dropout = dropout
    self._kwargs = kwargs

  def build(self, input_shape):
    """Builds the FeatureNNs on the first call."""
    self.feature_nns = [None] * self._num_inputs
    for i in range(self._num_inputs):
      self.feature_nns[i] = FeatureNN(
          num_units=self._num_units[i],
          dropout=self._dropout,
          trainable=self._trainable,
          shallow=self._shallow,
          feature_num=i)
    self._bias = self.add_weight(
        name='bias',
        initializer=tf.keras.initializers.Zeros(),
        shape=(1,),
        trainable=self._trainable)
    self._true = tf.constant(True, dtype=tf.bool)
    self._false = tf.constant(False, dtype=tf.bool)

  def call(self, x, training = True):
    """Computes NAM output by adding the outputs of individual feature nets."""
    individual_outputs = self.calc_outputs(x, training=training)
    stacked_out = tf.stack(individual_outputs, axis=-1)
    training = self._true if training else self._false
    dropout_out = tf.nn.dropout(
        stacked_out,
        rate=tf.cond(training, lambda: self._feature_dropout, lambda: 0.0))
    out = tf.reduce_sum(dropout_out, axis=-1)
    return tf.nn.sigmoid(out + self._bias)

  def get_loss(self, x,true_value,monotonic_feature,individual_output,alpha_1,pair,pair1,pair2,pair3,alpha_2,pair_s, pair_s1,alpha_3):
    output=self.call(x,training=True)
    output=tf.reshape(output, len(x))
    true_value=tf.cast(true_value,tf.float32)
    
    #Binary cross entropy
    BCE=-tf.reduce_sum(tf.multiply(tf.math.log(output+0.00001),true_value)+tf.multiply((1-true_value),tf.math.log(1-output+0.00001)))/len(x)
    
    #Punishment
    matrdefault=np.zeros(len(x[0]))
    puni_1=0
    ind=0
    for i in range(len(x[0])):
      if(i in monotonic_feature):
        for j in range(len(individual_output[ind])):
          temp=np.zeros(len(x[0]))
          temp[i]=individual_output[ind][j]
          individual_output_now = self.calc_outputs([temp], training=True)    
          if(j>=1):
            puni_1+=max(individual_output_pre[i]-individual_output_now[i],0)
          individual_output_pre=individual_output_now
        ind+=1
    
    punish_1=alpha_1*puni_1
    print("loss of single monotonicity",punish_1)
    
    """
    puni_2=0
    for i in range(len(pair)):
      temp=np.zeros(len(x[0]))
      temp1=np.zeros(len(x[0]))
      temp2=np.zeros(len(x[0]))
      temp3=np.zeros(len(x[0]))
     
      temp[0:3]=pair[i]
      temp1[0:3]=pair1[i]
      temp2[0:3]=pair2[i]
      temp3[0:3]=pair3[i]

      out=self.calc_outputs([temp], training=True)
      out1=self.calc_outputs([temp1], training=True)
      out2=self.calc_outputs([temp2], training=True)
      out3=self.calc_outputs([temp3], training=True)

      puni_2+=max(out1[0]-out[0],0)
      puni_2+=max(out2[0]-out[0],0)
      puni_2+=max(out3[0]-out[0],0)
      
    punish_2=alpha_2*puni_2
    print("loss of strong pairwise monotonicity",punish_2)

    puni_3=0
    for i in range(len(pair_s)):
      temp=np.zeros(len(x[0]))
      temp1=np.zeros(len(x[0]))

      temp[0:3]=pair_s[i]
      temp1[0:3]=pair_s1[i]

      out=self.calc_outputs([temp], training=True)
      out1=self.calc_outputs([temp1], training=True)

      puni_3+=max(out1[0]-out[0],0)
    
    punish_3=alpha_3*puni_3
    print("loss of single monotonicity in strong pairwise monotonicity",punish_3)
      
   """

    ans = tf.constant(BCE+punish_1)
    print("overall loss",ans)
    return ans
    
  def get_grad(self, x,true_value,monotonic_feature,individual_output,alpha_1,pair,pair1,pair2,pair3,alpha_2,pair_s,pair_s1,alpha_3):
    with tf.GradientTape() as tape:
      tape.watch(self.variables)
      L = self.get_loss(x,true_value,monotonic_feature,individual_output,alpha_1,pair,pair1,pair2,pair3,alpha_2,pair_s,pair_s1,alpha_3)
      g = tape.gradient(L, self.variables)
    return g
    
  
  def network_learn(self, x,true_value,monotonic_feature,individual_output,alpha_1,pair,pair1,pair2,pair3,alpha_2,pair_s, pair_s1,alpha_3,learning_r):
    g = self.get_grad(x,true_value,monotonic_feature,individual_output,alpha_1,pair,pair1,pair2,pair3,alpha_2,pair_s, pair_s1,alpha_3)
    tf.keras.optimizers.Adam(learning_rate=learning_r).apply_gradients(zip(g, self.variables))


  def calc_outputs(self, x, training = True):
    """Returns the output computed by each feature net."""
    training = self._true if training else self._false
    list_x = tf.split(x, list(self._kwargs['kwargs']), axis=-1)
    return [
        self.feature_nns[i](x_i, training=training)
        for i, x_i in enumerate(list_x)
    ]


class DNN(tf.keras.Model):
  """Deep Neural Network with 10 hidden layers.

  Attributes:
    hidden_layers: A list of 10 tf.keras.layers.Dense layers with ReLU.
    linear: Fully-connected layer.
  """

  def __init__(self, trainable = True, dropout = 0.15):
    """Creates the DNN layers.

    Args:
      trainable: Whether the DNN parameters are trainable or not.
      dropout: Coefficient for dropout regularization.
    """
    super(DNN, self).__init__()
    self._dropout = dropout
    self.hidden_layers = [None for _ in range(10)]
    for i in range(10):
      self.hidden_layers[i] = tf.keras.layers.Dense(
          100,
          activation='relu',
          use_bias=True,
          trainable=trainable,
          name='dense_{}'.format(i),
          kernel_initializer='he_normal')
    self.linear = tf.keras.layers.Dense(
        1,
        use_bias=True,
        trainable=trainable,
        name='linear',
        kernel_initializer='he_normal')
    self._true = tf.constant(True, dtype=tf.bool)
    self._false = tf.constant(False, dtype=tf.bool)

  def call(self, x, training = True):
    """Creates the output tensor given an input."""
    training = self._true if training else self._false
    for l in self.hidden_layers:
      x = tf.nn.dropout(
          l(x), rate=tf.cond(training, lambda: self._dropout, lambda: 0.0))
    x = tf.squeeze(self.linear(x), axis=-1)
    return x
