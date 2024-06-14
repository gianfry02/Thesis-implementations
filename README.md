# Thesis-implementations
This repository contains all the implementations done for the thesis "Quantization Algorithms for Neural Networks: A comparative analysis with other reduction techniques".  

## Quantization
The original quantization method can be found in this repository: 
* [quantized_neural_networks](https://github.com/elybrand/quantized_neural_networks).

Some modifications were done to it, so you can just download this one if you are interested in trying this implementation on your local machine.

## Setting Up Docker (Optional)
To run the code in a docker container, you'll first need to download [Docker](https://www.docker.com/get-started). Once you've got Docker installed, navigate to the github repo on your machine and run the following CLI command: 
```
docker build --tag quant_nnets .
```
This will build a docker image with the python 3.7.10 base image and download other required python packages from ```requirements.txt```. This is a somewhat large docker image since we have to download Tensorflow, but it's significantly smaller than the bloated AWS DLAMI I tried messing with. It may take a minute or two to build. To see that the docker image has built successefully, run the command `docker images` and look for `quant_nnets` under the repository column.

Another thing you'll have to do is to modify how much memory Docker Desktop is allotted. In Docker Destop, navigate to "Resources", then to "Advanced". I was able to run the scripts with 16GB memory allottment. This may be overkill, but I didn't bother to fine tune.

## Running Experiments
Once the image is built, you can start running the experiments. These experiments are set up so that model training and model compression occur in two separate scripts. Once we have a trained network, that network is saved in the directory `serialized_models`. To persist that trained model on your local machine, we'll use Docker volumes.

### Network Training (MNIST & CIFAR10)

To train a network on a docker container, run the following command
```
docker run -dit --name train_container \
                -v [absolute/path/to/repo]/serialized_models:/serialized_models \
                -v [absolute/path/to/repo]/train_logs:/train_logs \
           quant_nnets python [train_mnist_mlp.py, train_cifar10_cnn.py]
```
**NOTE**: To run the scripts without instantiating a Docker contianer, just run the command `python [train_mnist_mlp.py, train_cifar10_cnn.py]` in the `scripts` subdirectory.

**NOTE**: If you're running the scripts in a container, the `-d` flag will run the container in the background, so you won't see any output onto the command line during model training. If you want to track the model training progress in real-time, you can replace the `-dit` flag with `-it`. This runs the docker container in attached mode. You will see the verbose output from the Keras training in your terminal. 
### Network Compression (MNIST & CIFAR10)
Once you have a pretrained network saved in `serialized_models` you're ready to run the network quantization script. Both model quantization scripts `quantize_pretrained_mlp.py` and `quantize_pretrained_cnn.py` define parameter grids at the top of their respective python files which they cross-validate over. Those parameter grids include:
1. choices for the number of training data to be used to learn the quantization,
2. the number of bits to use in the quantization alphabet, defined as `bits`,
3. the radius of the quantization alphabet, defined as `alphabet_scalars`.

In the paper, we only validate over the last two parameters. You're welcome to play around with the number of samples used to train the quantization. Just note that if you do make any modifications to the scripts, you'll need to rebuild the Docker container by again calling `docker build --tag quant_nnets .` Fortunately, Docker builds containers incrementally so you won't have to wait as long as you did the first time since all of the requirements will have already been downloaded.

**BEWARE** that training the quantization cannot be done with mini-batches of data! You will find that using the entire CIFAR10 dataset to train the quantization requires an enormous amount of disk space (think: for a convolutional layer with 256 filters, every image gets transformed from having 3 channels to 256 channels; further, instead of acting on the entire image the network acts on patches of data. That means the number of training images is not equal to the number of "samples" to learn the quantization; the number of samples is actually magnified by the number of patches and the number of channels). You'll also happen to find, like I did when I tried using the entire training data set, that you don't get that much better performance than using a much smaller subset. (As a matter of fact, you can actually get competitive performance only using ~1 training image to learn a 4 bit quantization for CIFAR10. Don't ask me to explain why. I wanted to throw this in the paper but it seems like a quirk and not a trend. Generally speaking, I found that the alphabet scalar and the bit budget were more influential on quantization test accuracy than the number of training samples.) I've set it up so that the training data for the quantization at each layer is saved to disk and is read one at a time to prevent running out of RAM if your local machine is constrained with RAM or if you really want to go overboard with the number of images.

To run the network quantization scripts on a particular serialzed model, run the following command
```
docker run -dit --name quant_container \
                -v [absolute/path/to/repo]/serialized_models:/serialized_models \
                -v [absolute/path/to/repo]/quantized_models:/quantized_models \
                -v [absolute/path/to/repo]/train_logs:/train_logs \
                -v [absolute/path/to/repo]/model_metrics:/model_metrics \
           quant_nnets python quantize_pretrained_[mlp, cnn].py [name of serialized model in serialized_models/]
```

While the quantization script is running, it will log the progress of the quantization to the file `train_logs/model_quantizing.log`. Once the network is fully quantized with a particular parameter configuration the script will do two things:
1. the parameters for that quantization and the test accuracies of the analog and quantized networks will be written to a .csv file in `model_metrics/`. This .csv file will store the results of all of the quantizations from one call to the script.
2. the quantized model will be serialized into the directory `quantized_models/`.

**NOTE**: To run the scripts without instantiating a Docker contianer, just run the command `python quantize_pretrained_[mlp, cnn].py [name of serialized model in serialized_models/]` in the `scripts` subdirectory.

**NOTE**: Running the quantization script for both MNIST and CIFAR10 networks takes a while because of the number of parameters to cross-validate over. For the MNIST data set, we also use half the training set (25000 images) which is far more than is needed to get a competitive quantization, and the layers are fairly wide. The median time it took to quantize a network on my machine for the MNIST network was around 5 minutes. The CIFAR10 network takes some time because the network is deep-ish, the code requires reformatting the data to "vectorize" the image patches, and we also need some OS calls to save the output of previous layers to disk so we don't run out of RAM. The median time it took to quantize a CIFAR10 network was around 30 minutes.

**NOTE**: As before, if you're running the scripts in a container then remove the `-d` flag if you want to track the model quantization progress in real-time. I don't recommend that you do since all of the activity is logged and you can also look at the .csv file to see the performance of the freshly quantized networks.

## Lumping
The lumping algorithm was implemented from this paper:
*  [Compressing Neural Networks via Formal Methods](https://www.sciencedirect.com/science/article/pii/S0893608024003356).

A "toy" example can be found below:

Input:
```
N = {
    'k': 4,
    'S': [{0, 1}, {2, 3}, {4, 5}, {6}],
    'W': {
        1: {(0, 2): 0.1, (1, 3): 0.2, (0, 3): 0.3, (1, 2): 0.4},
        2: {(2, 4): 0.5, (3, 4): 0.6, (2, 5): 1, (3, 5): 1.2},
        3: {(4, 6): 0.7, (5, 6): 0.8}
    },
    'b': {
        1: {2: 0.1, 3: 0.2},
        2: {4: 0.3, 5: 0.6},
        3: {6: 0.4}
    },
    'A': {
        1: {2: 'ReLU', 3: 'ReLU'},
        2: {4: 'ReLU', 5: 'ReLU'},
        3: {6: 'ReLU'}
    }
}

k, S_prime, W_prime, b_prime, A_prime = lump_neural_network(N)
print(k)
print(S_prime)
print(W_prime)
print(b_prime)
print(A_prime)
```
```
Output: 
4
{0: {0: [0], 1: [1]}, 1: {(2,): [2], (3,): [3]}, 2: {(4, 5): [4, 5]}, 5: 0.5, 3: {(6,): [6]}}
{1: {(0, (2,)): 0.1, (1, (2,)): 0.4, (0, (3,)): 0.3, (1, (3,)): 0.2}, 2: {((2,), (4,)): 0.5, ((3,), (4,)): 0.6}, 3: {((4, 5), (6,)): 1.1}}
{1: {(2,): 0.1, (3,): 0.2}, 2: {(4,): 0.3}, 3: {(6,): 0.4}}
{1: {(2,): 'ReLU', (3,): 'ReLU'}, 2: {(4,): 'ReLU'}, 3: {(6,): 'ReLU'}}
```
