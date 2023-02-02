# LearningNeuralNetwork

## Attempt 1

```
Target:
My target is to get the base network with less than 10000 parameters and achieve accuracy around 99%
in 15 epochs

Result:
1. Total params: 7,760
2. Training accuracy: 99.15 %
3. Test accuracy: 99.05 %

Analysis:
1. With 8 covolutional layer and batch normalization network is showing quite stable
training and testing result
2. I have not observed any overfitting/underfitting sympton in network as there is not
a big diversion between training and testing result
3. With this the target is achieved
4. Now need to think on increasing the network capacity

```

## Attempt 2

```
Target:
Increasing the neural network capacity by adding more features. Also by seeing the dataset adding
data augmentation by adding rotation. Target is to achieve 99.4 % test accuracy.

Result:
1. Total params: 9,956
2. Training accuracy: 99.23 %
3. Test accuracy: 99.27 %

Analysis:
1. By increasing the capability of network network is learning better but not able to reach the desired
accuracy of 99.40 in 15 epochs.
2. The network learning is consistance as I don't see much difference in training and testing results.
3. This indicates that if learning rate is increased and step LR is introduced with 6 steps and gamma=0.1.
My network can hit the desired accuracy in 15epochs.

```

## Attempt 3

```
Target:
Doubling the learning rate and using stepLR with 5 steps and gamma=0.1 to improve learning rate upto 99.4
and to achieve consistance accuracy.

Result:
1. Total params: 9,956
2. Training accuracy: 99.41 %
3. Test accuracy: 99.45 %

Analysis:
1. Consistant desired testing accuracy is achieved.
2. Training accuracy is little low which I feel is due to the augmentation dataset in training.
```
