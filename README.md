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

## Formulation
In CycleGAN, there is no paired data to train on, so there is no guarantee that the input <img src="https://render.githubusercontent.com/render/math?math=X"> and the target pair <img src="https://render.githubusercontent.com/render/math?math=Y"> are meaningful during training. Thus, in order to enforcee that the network learns the correct mapping, the cycle-consistency loss is used.
#### Adversarial loss
The objective of adversarial losses for the mapping function <img src="https://render.githubusercontent.com/render/math?math=G : X \rightarrow Y"> and its discriminator <img src="https://render.githubusercontent.com/render/math?math=D_{Y}"> is expressed as:
 
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{GAN}(G, D_{Y}, X, Y)=\mathbb{E}_{y~p}_{data}(y)[\logD_{y}(y)]+\mathbb{E}_{x~p}_{data}(x)[\log(1-D_{y}(G(x))]"> [3]  

The goal is to generate images that are similar in style to the target domain while distinguising between the test data and the training data. 
#### Cycle-Consistent loss 
Adversarial losses alone do not guarantee that the content will preserved as it is mapped from the input to the target domain; therefore, cycle-consistent functions are implemented in order to prevent the learned mappings from contradicting each other. This cycle consistency loss objective is: 

<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{cyc}(G, F)=\mathbb{E}_{x~p}_{data}(x)[\|F(G(x))-x\|_{1}]+\mathbb{E}_{y~p}_{data}(y)[\|G(F(y))-y\|_{1}]"> [3]  

![Cycle-Consistency Loss](https://miro.medium.com/max/1258/1*XhdrXh3UfCM4CecRrTwMCQ.png)
<img src="https://render.githubusercontent.com/render/math?math=\text{forward cycle consistency loss: } X->G(X)->F(G(X))~ \hat X">
<img src="https://render.githubusercontent.com/render/math?math=\text{backward cycle consistency loss: } Y->F(Y)->G(F(Y))~ \hat Y">

#### Total Generator Loss


## Implementation
#### Generator Architecture
##### Resnet
The architecture for the style transfer networks follows a residual network (resnet) block design having convolutional layers with skip connections that can be used when the input data and the output data have the same dimensions. When those dimensions are not the same, mapping is still possible by increasing the spatial reflection padding with zeroes in order to match the additional dimensions needed. [5] This is used to reduce artifacts around the borders of the generated image.
##### U-Net
The network architecture consists of two 3x3 convolutions (unpadded convolutions), each followed by instance normalization and a rectified linear unit (ReLU) and a pooling operation with stride 2 for downsampling an input. During upsampling, a 3 × 3 convolution with no padding reduces the size of a feature map by 1 pixel on each side, so in this case the identity connection performs a center crop on the input feature map. [6]

#### Discriminator Architecture
##### PatchGAN
##### PixelGAN
##### ImageGAN
#### Training Details
In order to stabilize our training procedures, we contructed a loop that consists of four basic steps:
 - Get the predictions
 - Calculate the loss
 - Calculate the gradient using backpropogation
 - Apply the gradient to the optimizer
 
Within these training procedures, there is a random square cropping of the original images for training and a defined learning rate of <img src="https://render.githubusercontent.com/render/math?math=2e^{-4}">. However, in practice, the objective is divided by 2 while optimizing D, which slows down the rate at which D learns, relative to the rate of G. This learning rate is constant for the first 100 epochs and linearly decays to zero over the next 100 epochs.    

## Experiments and Results
#### ResNet on different datasets  with default Generator and PatchGAN
 - Monet
 - Cezanne
 - Ukiyou
 - Van Gogh
#### Different generator architecture with PatchGAN
 - Resnet with norm_type = Batch Norm 
 - Resnet with norm_type = Instance Norm
 - Resnet with type_net = ”non-residual”
 - U-Net and PatchGAN with norm_type = Batch Norm
 - U-Net with norm_type = Batch Norm and PatchGAN with InstanceNorm 
 - U-Net with norm_type = Instance Norm
#### ResNet with default generator configuration and different discriminator
 - PixelGAN
 - PatchGAN (n_layers=1) 
 - PatchGAN (n_layers=3) 
 - PatchGAN (n_layers=5) = ImageGAN
#### ResNet with default config but with different padding type
 - Reflect
 - Zero
 - Symmetric
 #### Different loss function
 - Binary Cross Entropy for adversarial loss
 - MSE for adversarial loss 


### Evaluation Metrics
### Analysis

## Conclusion and Future Work

## References
 - [1] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1125–1134, 2017.
 - [2] L. A. Gatys, A. S. Ecker, and M. Bethge, “A neural algorithm of artistic style,” arXiv pre printarXiv:1508.06576, 2015.
 - [3] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” in Proceedings of the IEEE international conference on computer vision, pp. 2223–2232, 2017.
 - [4] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” in Advances in neural information processing systems, pp. 2672–2680, 2014.
 - [5] K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, 2016, pp. 770-778
 - [6] U-Net: O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in Medical Image Computing and Computer-Assisted Intervention (MICCAI), pp. 234-241, 2015.

#### Contributions
