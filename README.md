# RTNet: A neural network that exhibits the signatures of human perceptual decision making
The preprint of the paper is avilable [here](https://www.biorxiv.org/content/10.1101/2022.08.23.505015v2.abstract).

## Note from Tom 
You can download my trained models (using pretrained AlexNet) from: https://1drv.ms/f/s!AvS3REhgQw0Rt_RXFLhokt5UclKAaQ?e=QxYH5t
Store it under results/pretrained_models/

For the simulation data (using RTNet.ipynb) I used model_01.pt and model_01_params.pt from the OSF page (see below). 
Store it under data/Bayesian_models/

Most of the code is from Rafiei et al. (2024). I made changes here and there to accommodate a different file hierarchy and the corresponding file handling. The file train_pretrained_AlexNet.ipynb is similar to train.ipynb with the important difference that I am using a pre-trained AlexNet instead of training the network from scratch. This is the file that I am having difficulty with. I think perhaps the parameters are not stored correctly after training, which would explain why it looks good during training but not when evaluating. However, the method Rafiei are using (with the random_module function) is a bit unfamiliar to me (it is an outdated function that the current Pyro tutorials are not addressing) and I am trying to figure out what is going on there. 

There are also R-related files that I used to do analysis on simulated data created with RTNet.ipynb. 

## Rest of README is from Rafiei et al.

## Requirments
The files in this repository should be run on Google Colab. All dependencies will be automatically taken care of when running the code on Colab. 

## Run the code
The instructions to run the simulations are given at the begining of each notebook. In summary, all you need to do is to provide the path for pretrained models and output path to save the resulting simulations. Once that is done, you are all set. Press Run button. 

In case you want to train a model from scratch, please use the *train* notebook. Don't forget to save the trained model. You can later use the saved model to run simulations. 

For two levels of noise and two threshold levels, simulation results for each model will be ready in less than 5 minutes for the whole MNIST dataset. Expected example output is provided [here](https://github.com/frafiei3/RTNet/blob/main/RTNet_simulation_example.csv).

## Pretrained models
To download the pretrained models, you can go to our [OSF page](https://osf.io/akwty/). 


