# A simple baseline that questions the use of pretrained-models in continual learning 

## Dataset preparation

Most datasets will be downloaded automatically 

1. Download Imagenet-R from [Here](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)

2. Download CoRe50 from [Here](http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip)


## Prepare Environment

Create environment using environment.yml

`conda env create -f environment.yml -p ./env`

`conda activate ./env`

## Run the experiments 

The experiments by following code

`python main_{dataset-name}.py`