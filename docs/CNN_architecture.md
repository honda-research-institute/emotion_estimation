** Self Supervised Representation Learning architecture
 - from the paper: "[Self-supervised ECG Representation Learning for Emotion Recognition](https://ieeexplore.ieee.org/document/9161416)"

| Module | layer details, filter size, no. of filters | Feature shape|
| ------------------ |
|Input | - | N samples x 1 x 2000 |
| ------ | ------ | ------ |
|CNN block | [conv block, 1 x 32, 32] x 2 with Leaky RELU activations | |
| | [maxpool, 1 x 8, stride = 2] | |
| | [conv block, 1 x 16, 64] x 2 with Leaky RELU activations | |
| | [maxpool, 1 x 8, stride = 2] | |
| | [conv block, 1 x 8, 128] x 2 with Leaky RELU activations | |
| | global max pooling | N samples x 128 features|
| ------ | ------ | ------ |
| Fully connected block | [dense, 128 nodes] x 2 with Leaky RELU activations | |
| | [dense, 7 nodes] x 1 with Leaky RELU activations | N samples x 7| 
| ------ | ------ | ------ |

*** CNN resource:
- basic information about CNN padding, pooling and stride. (also clarifies the "VALID" vs "SAME" options for padding)
https://towardsdatascience.com/the-most-intuitive-and-easiest-guide-for-convolutional-neural-network-3607be47480

*** Parameter tuning and convergence:
ADAM: https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c 


*** Loading a partial torch model for downstream tasks:
https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/56


