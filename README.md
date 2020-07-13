## Neural Style Transfer using Cycle-Consistent GAN
### Outline
1. [Introduction](https://github.com/bethanystate/CS7641_project/blob/master/README.md#introduction)
2. [Dataset](https://github.com/bethanystate/CS7641_project/blob/master/README.md#dataset)
3. [Unsupervised Learning](https://github.com/bethanystate/CS7641_project/blob/master/README.md#unsupervised-learning)
4. [Supervised Learning](https://github.com/bethanystate/CS7641_project/blob/master/README.md#supervised-learning)
5. [Evaluation Metrics](https://github.com/bethanystate/CS7641_project/blob/master/README.md#evaluation-metrics)
6. [Conclusion and Future Work](https://github.com/bethanystate/CS7641_project/blob/master/README.md#conclusion-and-future-work)
7. [References](https://github.com/bethanystate/CS7641_project/blob/master/README.md#references)
8. [Contributions](https://github.com/bethanystate/CS7641_project/blob/master/README.md#contributions)

## Introduction
Image-to-image translation is a class of vision and graphics problems where the goal is to learn the
mapping between an input image and an output image using a training set of aligned image pairs. [1]
Neural Style Transfer is one way to perform image-to-image translation, which synthesizes a novel
image by combining the content of one image with the style of another image-based on matching
the Gram matrix statistics of pre-trained deep features. [2] The primary focus of this project is to
directly learn the mapping between two image collections (in an unsupervised way), rather than
between two specific images, by trying to capture correspondences between higher-level appearance
structures. [3]
## Dataset
The dataset used for this image-to-image translation was sourced from a [UC Berkley CycleGAN Directory](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/) by way of [TensorFlow](https://www.tensorflow.org/datasets/catalog/cycle_gan#cycle_ganmonet2photo). It consists of 8,000+ images from 2 classes: French Impressionist paintings and modern photography both of landscapes and other natural scenes. The completeness of this dataset is attributed to the labeled training set of the two aligned classes that we then use to synthesize new images of nature in the style of Monet.
## Unsupervised Learning
#### Preprocessing
#### Cycle-GAN
#### VAE (Variational Auto-Encoder)
#### Results
## Supervised Learning??
#### Method
#### Results
## Loss Functions
In CycleGAN, there is no paired data to train on, so there is no guarantee that the input <img src="https://render.githubusercontent.com/render/math?math=X"> and the target pair <img src="https://render.githubusercontent.com/render/math?math=Y"> are meaningful during training. Thus, in order to enforcee that the network learns the correct mapping, the cycle-consistency loss is used.
#### Adversarial loss
The objective of adversarial losses for the mapping function <img src="https://render.githubusercontent.com/render/math?math=G : X \rightarrow Y"> and its discriminator <img src="https://render.githubusercontent.com/render/math?math=D_{Y}"> is expressed as:
 
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{GAN}(G, D_{Y}, X, Y)=\mathbb{E}_{y~p}_{data}(y)[\logD_{y}(y)]+\mathbb{E}_{x~p}_{data}(x)[\log(1-D_{y}(G(x))]"> [3]  

The goal is to generate images that are similar in style to the target domain while distinguising between the test data and the training data. 

#### Cycle-Consistent loss 
Adversarial losses alone do not guarantee that the content will preserved as it is mapped from the input to the target domain; therefore, cycle-consistent functions are implemented in order to prevent the learned mappings from contradicting each other. This cycle consistency loss objective is: 

<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{cyc}(G, F)=\mathbb{E}_{x~p}_{data}(x)[\|F(G(x))-x\|_{1}]+\mathbb{E}_{y~p}_{data}(y)[\|G(F(y))-y\|_{1}]"> [3]  

#### Perceptual loss 
## Evaluation Metrics
#### Analysis
## Conclusion and Future Work
## References
[1] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1125–1134, 2017.
[2] L. A. Gatys, A. S. Ecker, and M. Bethge, “A neural algorithm of artistic style,” arXiv pre printarXiv:1508.06576, 2015.
[3] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” in Proceedings of the IEEE international conference on computer vision, pp. 2223–2232, 2017.
[4] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” in Advances in neural information processing systems, pp. 2672–2680, 2014.
#### Contributions
