A very simple example of **probabilistic** neural network/**uncertainty-aware** neural network.

* Intro

Given an input data point, neural networks usually produce a deterministic output value. However, due to the noise in labels - or the stochastic nature in the real mapping - the output given an input might be a distribution. 

Therefore, it may be benificial if the network can produce the distribution of the output conditioned on the input, cuz we'll be aware of how uncertain the network is about its prediction. 

This is a very simple example of how we may enable the network to estimate that distribution. The network has two heads, one predicts the expectation, the other predicts the variance. 

* Result

The training data is sampled from a gaussian distribution whose mean and variance varies according to the input x. The predicted distribution is compared against the true one. The network not only fits the expected output value correctly, but is also aware of the varying level of uncertainty at different locations.

![Results on a toy example](/result.png)

* Reference

Checkout this paper for details:

Nix, David A., and Andreas S. Weigend. "Estimating the mean and variance of the target probability distribution." Proceedings of 1994 IEEE International Conference on Neural Networks (ICNN'94). Vol. 1. IEEE, 1994.

