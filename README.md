# pbase

Just a deep learning library for my convenience to develop the code quickly.

## Documents

On the progress.

[Tutorial and Documents](https://impavidity.github.io/pbase/index.html)


## TODO

- k-max pooling: https://discuss.pytorch.org/t/resolved-how-to-implement-k-max-pooling-for-cnn-text-classification/931/3 , http://www.cnblogs.com/Joyce-song94/p/7277871.html
- Connect to Stanford core NLP and support automatically annotate corpus for training
- Vis idea, use your own layer to support this feature
- training APP (support results distribution)
- support pair wise training (hinge loss, max margin loss)


# Install

## Conda 
We use conda as your python environment manager, which we highly recommend. To install conda, please visit [here](https://www.anaconda.com/download/) to install the Python3.6 version in your machine.

For example, we use linux 64bit OS so we download regarding version, and isntall. 
```
wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh
bash Anaconda3-5.1.0-Linux-x86_64.sh
```

## Environment
We recommend you to use *environment* to manage your library dependencies. 
To create an environment and activate it:
```
conda create --name deeplearning
source activate deeplearning
```
Now, you are in your created environment calld *deeplearning*.
To leave this environment, just simply use
```
source deactivate
```

## PyTorch, torchtext and TensorFlow
Here we use the PyTorch 0.4.0 version and torchtext 0.2.3 
```
conda install torchvision -c pytorch
conda install -c anaconda cython
conda install -c anaconda scipy 
pip install torchtext
```
We utilize the *TensorFlow* for visualization. Here in our machine, we use cuda8.0, so we need to install the package with specific version.
```
pip install tensorflow-gpu==1.4.1
```
If you are using higher version of cuda, you could use 
```
pip install --ignore-installed --upgrade tfBinaryURL
```
for the URL, please visit [here](https://www.tensorflow.org/install/install_linux?hl=en#the_url_of_the_tensorflow_python_package) to get the one regarding your machine.




