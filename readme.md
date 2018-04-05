# Sample solution for Kaggle Jigsaw competition

Pipeline author: [Andrey Vykhodtsev](https://www.linkedin.com/in/vykhand/)

[GitHub Pages link](https://vykhand.github.io/kaggle-jigsaw)

## Attribution

My work here is focused on tooling, monitoring, execution, clean pipeline and integration with Azure services. Other code was adopted from many public kernels and github repositories that can be [found here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/kernels) and [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion). I tried to keep the references to original kernels or githubs in the code comments, when appropritate. I apologize if I missed some references or quotes, and if so, please let me know.

## About this solution

This is the pipeline I developed during the [Kaggle Jigsaw Toxic comment classification challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). In this competition, I focused more on tools rather than on winning or developing novel models.

The goal of developing and publishing this is to share reusable code that will help other people run Kaggle competitions on Azure ML Services.

This is the list of features:

 * Everything is a hyperparameter approach. Config files control everything from model parameters to feature generation and cv.
 * Run experiments locally or remotely using Azure ML Workbench
 * Run experiments on GPU-based [Azure DSVM](https://aka.ms/dsvm) machines
 * Easily start and stop VMs
 * Add and remove VMs to and from the VM fleet
 * Zero manual data download for new VMs - files are downloaded from Azure Storage on demand by code.
 * Caching feature transformations in Azure Storage
 * Shutting down the VMs at the end of experiment
 * Time and memory usage logging
 * Keras/Tensorboard integration
 * Usage of [Sacred](https://pypi.python.org/pypi/sacred) library and logging experiment reporducibility data to CosmosDB
 * Integration with Telegram Bot and Telegram notifications
 * Examples of running LightGBM, sklearn models, Keras/Tensorflow


## Presentations and blog posts

This solution is accompanying a few presentations that I gave on [PyData meetup in Ljubljana](https://www.meetup.com/PyData-Slovenia-Meetup/) and [AI meetup in Zagreb](https://www.meetup.com/Artificial-Intelligence-Future-Meetup/). Links to the slides:

 * [PyData Ljubljana #5](https://www.slideshare.net/andreyvykhodtsev/20180328-av-kagglejigsawwithamlwb-92229518)
 * AI Future Zagreb

I am also writing a blog post which is going to published [here](https://vykhand.github.io).

## Technical overview of the solution

This sample solution allows you to run text classification experiments via command line or via [Azure ML Workbench](http://aka.ms/AzureMLGettingStarted).
To see, how to run, configure and monitor experiments, refer to the [Standard Operating Procedures]() section.
If you wish to extend or modify the code, please refer to [Modifying the solution]

### Pipeline

Pipeline intends to be extendable and configurable to support "everything is a hyperparameter" approach.

![Pipeline: code structure](img/pipeline.png)

### Architecture
Below is the diagram of compute architecture. I used cheap Windows DSVM (burst instances) to run series of experiments and stop the machines after all experiments are finished.
I used 2-3 CPU VMs and 2 GPU vms to run experiments. I used minimial sized collections for CosmosDB to store "python/sacred" experiment information.
Information from Azure ML runs is stored in a separate storage account, which has 1 folder per experiment, automatically populated by AML.


![Training pipeline architecture on azure](img/arch.png)

### Experiment flow



TODO: draw better diagram

![Experiment flow doodle drawing](img/experiment_flow_doodle.png)

## Deploying solution

 1. Check out the [git repository]()
 1. [Install AML Workbench (AMLWB)]()
 1. [Add github folder to AMLWB]()
 1. Refer to [Azure CLI]() section to set up az cli and find your tenant, and also login into azure from CLI
 1. Use the [Example script]() to set up new VM
 1. Create Azure Storage and load [Data]() to it
 1. [Create Cosmos DB for Sacred experiment monitoring]()
 1. [Create telegram bot]()
 1. Copy environment variables from [Example VM configuration]() to your amlconfig \<vm\>.runconfig and populate them with your own values
 1. If you are setting up a GPU VM, be sure to follow the [instruction]() or math the [Example GPU config]()

### Setting up Azure CLI and getting your tenant

### Setting up Cosmos DB

### Creating storage account and loading data

### Setting up Azure ML Services

In this competition, I used experiment monitoring facility and notebook facility. I have not used many other useful capabilities described in the [documentation]()

More information is availabie at <http://aka.ms/AzureMLGettingStarted>

## Data downloads

Data for this competition is not distributed with this repository. You need to download it from the competition page located [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data). There are also multiple other external files that I used for this competition:

 * [Google's Full list of banned words](https://www.freewebheaders.com/full-list-of-bad-words-banned-by-google/)
 * [Glove embedding vectors](https://nlp.stanford.edu/projects/glove/)
 * [FastText embedding vectors](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)




## Standard operating procedures

### Running experiments with AMLWB
### Accessing experiment files

#### Submitting to Kaggle

### Adding new compute
### Deallocating machines
### Running series of experiments
### Monitoring GPU usage
### Monitoring Tensorflow using Tensorboard
### Retrieving experiment results from CosmosDB
### Accessing experiment storage directly

## Modifying solution
### Code structure

## Future work and TODOs
