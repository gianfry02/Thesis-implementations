# Thesis-implementations
This repository contains all the implementations done for my thesis "Quantization Algorithms for Neural Networks: A comparative analysis with other reduction techniques".  

## Quantization
The original quantization method can be found in this repository: 
* [quantized_neural_networks](https://github.com/elybrand/quantized_neural_networks).
Some modifications were done to it, so you can just download this one if you are interested in trying my implementation on your machine. 

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
