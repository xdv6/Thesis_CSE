# Thesis_CSE



## Installation



First create conda environment with python 3.7

### TF and baselines

https://github.com/openai/baselines

Clone the repo and cd into it:

```
git clone https://github.com/openai/baselines.git
cd baselines
```

​    

  

If you don't have TensorFlow installed already, install your favourite flavor of TensorFlow. In most cases, you may use

```
pip install tensorflow-gpu==1.14 # if you have a CUDA-compatible gpu and proper drivers
```

​    

or

```
pip install tensorflow==1.14
```

​    

to install Tensorflow 1.14, which is the latest version of Tensorflow supported by the master branch. Refer to [TensorFlow installation guide](https://www.tensorflow.org/install/) for more details.

Install baselines package

```
pip install -e .
```





### MUJOCO

https://gist.github.com/saratrajput/60b1310fe9d9df664f9983b38b50d5da





### Robosuite

https://robosuite.ai/docs/installation.html

if there is a problem with module not found, use the ```pip install robosuite``` instead of that from the requirements.txt file



probleem met ELG oplossen : 

https://robosuite.ai/docs/installation.html



