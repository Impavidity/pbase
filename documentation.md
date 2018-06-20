# Overview

The text processing library is adapted from [torchtext](https://github.com/pytorch/text).

# Quick Start

## Concept

- Each Json object as an **Example**. 
- A **Dataset** Object comprises a **List[Example]**. 
- A **Field** Object is used for **preprocessing**, **padding**. Recommend to support **numericalize** in the future version.
- A **Iterator** Object comsumes **Dataset** Object and generate **Batch**.

## How to use

In the new framework, we recommend use this library in the following way. Here we use NER model as example.

First we define a class **TrainNERPipeline** inherent from class **Pipeline** to control the training loop. 

```
class TrainNERPipeline(Pipeline):
  """Training pipeline 
  """
  def __init__(self, config):
    super(TrainNERPipeline, self).__init__(config)
```

Here we pass all the configuration into the training pipeline object, including the hyper-parameters, input file, output file, etc.

In this training pipeline, you basically will go through the following procedures: 
- Data Processing
- Trainer Initialization 
- Training
- Save Model

This is the default training loop defined in the **Pipeline** clcass and **apply** is called to run all these function. However, you can define your own procedures and overried all these function.

```

class Pipeline(object):
  """Pipeline Class is the controller for model training,
   which is the replacement of ``train_loop.train_and_save``
  
  The basic procedure including:
    - data processing to generate data iterator
    - trainer initialization and apply to dataset for training
    - save models
    
  This class provide basic interface for the training pipeline, 
  however, the user could override the procedures for their needs.
  
  """
  def __init__(self, config):
    self.config = config

  def data_processing(self):
    raise NotImplementedError

  def trainer_initialization(self):
    raise NotImplementedError

  def training(self):
    raise NotImplementedError

  def save_model(self):
    raise NotImplementedError

  def apply(self):
    self.data_processing()
    self.trainer_initialization()
    self.training()
    self.save_model()
```

In the **data_processing** function, you will first retrieve data from local/hdfs and return a **List[Example]**.
The following exsiting code is an example.

```
ner_training_examples = subsample_on_debug(
    load_json_dataset(ner_hdfs_path(language, "training"), "ner-training.json"), debug)
ner_training_dataset = Dataset(ner_training_examples, filter_pred=None)
```
Here you can use filter_pred to control the example you want to use, if you do not use the whole dataset.

Then you could create a **Iterator** for this **Dataset** Object. 

```
train_iterator = Iterator(ner_training_dataset,
                            batch_size=batch_size,
                            train=True,
                            shuffle=True)
```
Here, you can control the **batch_size**, shuffle the dataset before each epoch or not, sort the dataset before each epoch or not, with the sort_key.
For more information, you could look into the source code for more control.

Now let's move to the modeling part.

Similar, here, we will follow a specific way to define the model. However, you could override regarding your needs.

```
class Model(object):
  """Model Class is the base class for model definition, training and evaluation. 
  
  def __init__(): preparing embeddings, generate operations for training and evaluation
  def build(): generate operations 
  def train(): train the dataset given a batch of examples
  def eval(): evaluate the batch given a batch of examples
  
  """
  def __init__(self, config):
    raise NotImplementedError

  def build(self):
    raise NotImplementedError

  def train(self, batch):
    raise NotImplementedError

  def eval(self, batch):
    raise NotImplementedError
```

Basically you will store all the configuration in the model. The **self.config** will used among the trianing process.

Then you will build the operation in the **build()** function.
Use BiLSTM as an example here.

```
def build(self):
  # NER specific functions
  self.add_placeholders()
  self.add_word_embeddings_op()
  self.add_logits_op()
  self.add_pred_op()
  self.add_loss_op()
  self.add_train_op()
  self.initialize_session()
```

You firstly add placeholders.
```
def add_placeholders(self):
  """Define the input of the computation graph 
  """
  # shape = ( Batch size, Max_length of sentence in batch)
  self.word_input_tokens = tf.placeholder(dtype=tf.string, shape=[None, None],
                           name=self.WORD_TOKENS_NODE_NAME)
  # shape = ( Batch size )
  self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[None],
                           name=self.SEQUENCE_LENGTH_NODE_NAME)
  # shape = ( Batch size, Max_length of sentence in batch, Max_length of word)
  self.char_input_tokens = tf.placeholder(dtype=tf.string, shape=[None, None, None],
                           name=self.CHAR_TOKENS_NODE_NAME)
  # shape = ( Batch size, Max_length of sentence )
  self.word_length = tf.placeholder(dtype=tf.int32, shape=[None, None],
                           name=self.WORD_LENGTH_NODE_NAME)
  # shape = ( Batch size, Max_length of sentence )
  self.labels = tf.placeholder(dtype=tf.int32, shape=[None, None],
                           name=self.LABEL_NODE_NAME)
  # hyper parameters
  self.learning_rate_placeholder = tf.placeholder(dtype=tf.float32, shape=[],
                           name=self.LEARNING_RATE_NODE_NAME)
  self.rnn_cell_dropout_placeholder = tf.placeholder_with_default(1.0, shape=[],
                           name=self.RNN_CELL_DROPOUT_NODE_NAME)
```

and then add embedding.

```
def add_word_embeddings_op(self):
  """Define the word wordembeddings 
  """
  self.word_embeddings = get_word_embeddings(self.config, debug=True)
```

Add your main operation here.

```
def add_logits_op(self):
  """Define the computation graph 
  """

  token_data = self.word_embeddings.embedding(self.input_tokens)
  self.input_sequence_length = tf.reduce_sum(
          tf.cast(
              tf.not_equal(self.input_tokens, self.TEXT_PAD_TOKEN),
          tf.int32),
        axis=1)

  with tf.variable_scope("sequence_labeling") as ner_scope:
    fw_cell = rnn.LSTMCell(self.config.number_of_lstm_cells, state_is_tuple=True)
    d_fw_cell = rnn.DropoutWrapper(fw_cell, output_keep_prob=self.rnn_cell_dropout_placeholder)

    bw_cell = rnn.LSTMCell(self.config.number_of_lstm_cells, state_is_tuple=True)
    d_bw_cell = rnn.DropoutWrapper(bw_cell, output_keep_prob=self.rnn_cell_dropout_placeholder)

    lstm_output, _ = tf.nn.bidirectional_dynamic_rnn(
      cell_fw=d_fw_cell,
      cell_bw=d_bw_cell,
      inputs=token_data,
      dtype=tf.float32,
      sequence_length=self.input_sequence_length)

    self.lstm_output_concat = tf.concat(lstm_output, 2, name=self.NER_LSTM_OUTPUT_NODE_NAME)
    self.scores = tf.layers.dense(
      inputs=self.lstm_output_concat,
      units=self.config.number_of_classes,
      use_bias=True,
      kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01,dtype=tf.float32),
      bias_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32),
      name=self.ENTITIES_SCORES_NODE_NAME)

    self.variables = tf.global_variables(scope=("^" + ner_scope.name))
```

and same for others.

```
def add_pred_op(self):
  """Compute the softmax for crf layer 
  """
  self.label_pred = tf.nn.softmax(self.scores, axis=2)

def add_loss_op(self):
  """Define the loss function 
  """
  losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=self.scores, labels=self.labels)
  mask = tf.sequence_mask(self.input_sequence_length)
  losses = tf.boolean_mask(losses, mask)
  self.loss = tf.reduce_mean(losses)

def add_train_op(self):
  """Define Optimizer 
  """
  gd_update = self.config["ner_gd_update"]
  if  gd_update == "adam":
    lr = self.config["ner_adam_lr"]
    optimizer = tf.train.AdamOptimizer(lr)
  elif gd_update == "sgd":
    lr = self.config["ner_sgd_lr"]
    optimizer = tf.train.GradientDescentOptimizer(lr)
  else:
    raise NotImplementedError("Unknown optimizer {}".format(gd_update))

  self.train_op = optimizer.minimize(self.loss)
  
def initialize_session(self):
  self.sess = tf.Session()
  self.sess.run(tf.global_variables_initializer())

```

For **train** function, you basically process the **Batch** and build a feed dict for training.

```
def train(self, batch):
  feed_dict = process_batch(batch)
  self.sess.run(self.train_op, feed_dict=feed_dict)
```

After you prepare the data iterator and model, you could use them for training now.

Now let's move to the **Pipeline.training** function.
Bascially in this function, you will iterate on each batch which would be fed into model for training. 
You will also specify how many epoch you want to run, or break when meet some condition (such as metrics does not increase for several epoch).

For example,
```
with tf.device(self.device):
  tf_config=tf.ConfigProto(allow_soft_placement=True)
  tf_config.gpu_options.allow_growth = True
  with tf.Session(config=tf_config) as sess:
    ner_model = NERModel(config=self.config)
    ner_model.build()
    for batch_idx, batch in enumerate(self.train_iterator):
      ner_model.train(batch)
```


