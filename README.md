## Introduction
Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. [1] Neural Style Transfer is one way to perform image-to-image translation, which synthesizes a novel image by combining the content of one image with the style of another image-based on matching the Gram matrix statistics of pre-trained deep features [2]. Unlike recent work on "neural style transfer", we used CycleGAN [3] method which learns to mimic the style of an entire collection of artworks, rather than transferring the style of a single selecterd piece of art. Therefore, we can learn to generate photos in the style of, e.g., Van Gogh, rather than just in the style of Starry Night.

## Dataset
The dataset used for this project is sourced from a [UC Berkley CycleGAN Directory](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/) and is downloaded from by[TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/cycle_gan#cycle_ganmonet2photo). It consists of 8,000+ images from 2 classes: French Impressionist paintings and modern photography both of landscapes and other natural scenes. The size of the dataset for each artist/style was 526, 1073, 400, and 463 for Cezanne, Monet, Van Gigh, and Ukiyo-e. 

## Formulation
In CycleGAN, there is no paired data to train on, so there is no guarantee that the input <img src="https://render.githubusercontent.com/render/math?math=X"> and the target pair <img src="https://render.githubusercontent.com/render/math?math=Y"> are meaningful during training. Thus, in order to enforcee that the network learns the correct mapping, the cycle-consistency loss is used. In addition, adversarial loss is used to train generator and discriminator networks. Moreover, the identity loss is used to make sure generators generate the same image if the input image belongs to their target domian. 
#### Adversarial loss
The objective of adversarial losses for the mapping function <img src="https://render.githubusercontent.com/render/math?math=G : X \rightarrow Y"> and its discriminator <img src="https://render.githubusercontent.com/render/math?math=D_{Y}"> is expressed as:
 
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{GAN}(G, D_{Y}, X, Y)=\mathbb{E}_{y~p}_{data}(y)[\logD_{y}(y)]%2B\mathbb{E}_{x~p}_{data}(x)[\log(1-D_{y}(G(x))]"> [3]

In the above formula, generator <img src="https://render.githubusercontent.com/render/math?math=G"> tries to minimize the:

<img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}_{x~p}_{data}(x)[\log(1-D_{y}(G(x))]">

and in fact is trained to maximize the:

<img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}_{x~p}_{data}(x)[D_{y}(G(x)]">

while the discriminator 
<img src="https://render.githubusercontent.com/render/math?math=D_{Y}"> is trained to maximize the entire:

<img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}_{y~p}_{data}(y)[\logD_{y}(y)]%2B\mathbb{E}_{x~p}_{data}(x)[\log(1-D_{y}(G(x))]">.

On other hand, the same loss is applied for mapping from <img src="https://render.githubusercontent.com/render/math?math=F : Y \rightarrow X"> and its discriminator <img src="https://render.githubusercontent.com/render/math?math=D_{X}">:
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{GAN}(F, D_{X}, Y, X)=\mathbb{E}_{x~p}_{data}(x)[\logD_{x}(x)]%2B\mathbb{E}_{y~p}_{data}(y)[\log(1-D_{x}(F(y))]">

Therefore, the total Adverserial loss is expressed as:

<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{GAN}(F, D_{X}, Y, X)">+<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{GAN}(G, D_{Y}, X, Y)">

The goal is to generate images that are similar in style to the target domain while distinguising between the test data and the training data. 
#### Cycle-Consistent loss 
Adversarial losses alone do not guarantee that the content will preserved as it is mapped from the input to the target domain; therefore, cycle-consistent functions are implemented in order to prevent the learned mappings from contradicting each other. This cycle consistency loss objective is: 

<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{cyc}(G, F)=\mathbb{E}_{x~p}_{data}(x)[\|F(G(x))-x\|_{1}]%2B\mathbb{E}_{y~p}_{data}(y)[\|G(F(y))-y\|_{1}]"> [3]  

![Cycle-Consistency Loss](https://miro.medium.com/max/1258/1*XhdrXh3UfCM4CecRrTwMCQ.png)

<img src="https://render.githubusercontent.com/render/math?math=\text{forward cycle consistency loss: } X \rightarrow G(X) \rightarrow F(G(X))~ \hat X">
<img src="https://render.githubusercontent.com/render/math?math=\text{backward cycle consistency loss: } Y \rightarrow F(Y) \rightarrow G(F(Y))~ \hat Y">

#### Identity loss 
For painting to photo, it is helpful to introduce an additional loss to encourage the mapping to preserve color composition between the input and output. In particular, Identity loss regularizes the generator to be near an identity mapping when real samples of the target domain are provided as the input to the generator:

<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{identity}(G, F)=\mathbb{E}_{y~p}_{data}(y)[\|G(y)-y\|_{1}]%2B\mathbb{E}_{x~p}_{data}(x)[\|F(x)-x\|_{1}]">

#### Total Generator Loss
Summing the total previously explained loss functions lead to the following total losss function:

<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{GAN}(F, D_{X}, Y, X)%2B\mathcal{L}_{GAN}(G, D_{Y}, X, Y)%2B\mathcal{L}_{cyc}(G, F)%2B\mathcal{L}_{identity}(G, F)">

## Implementation
### Network Architecture
The architecture for our generative networks is adopted from Johnson et al. who have shown impressive results for neural style trasnfer. Similar to Johnson et al. [], we use instance normalization [] instead of batch normalization []. Both generator and discriminator use modeules of the form convolution-InstanceNormalizatio-ReLu []. The keys features of the network are detailed below: 
#### Generator Architecture 
A defining feature of image-to-image translation problems is that they map a high resolution input grid to a high resolution output grid. In addition, for the problems we consider, the input and output differ in surface appearance, but both are renderings of the same underlying structure. Therefore, structure in the input is roughly aligned with structure in the output. The generator architecture is designed around these considerations.

#### ResNet
We use 9 residual blocks for 256 × 256 training images. The residual block design we used is from Gross and Wilber [], which differs from that of He et al [] in that the ReLU nonlinearity following the addition is removed. The naming convention we followed is same as in the Johnson et al.'s [Github repository](https://github.com/jcjohnson/fast-neural-style). Let c7s1-k denote a 7 × 7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1. dk denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2. Reflection padding was used to reduce artifacts. Rk denotes a residual block that contains two 3 × 3 convolutional layers with the same number of filters on both layer. uk denotes a 3 × 3 fractional-strided-Convolution-InstanceNorm-ReLU layer with k filters and stride 1/2. Note that is is our default generator network. The network with 9 residual blocks consists of:  
<span style="font-family:Papyrus; font-size:4em;">c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3</span>


#### U-Net
The network architecture consists of two 3x3 convolutions (unpadded convolutions), each followed by instance normalization and a rectified linear unit (ReLU) and a pooling operation with stride 2 for downsampling an input. During upsampling, a 3 × 3 convolution with no padding reduces the size of a feature map by 1 pixel on each side, so in this case the identity connection performs a center crop on the input feature map.[6] In other words, the U-net architecture provides low-level information with a sortof shortcut across the network.

#### Discriminator Architecture
##### PatchGAN
The aim with PatchGAN  is to use the generative model in order classify whether overlapping image patches are real or fake. Because this discriminator requires fewer parameters, it works well with arbitrarily large images by running the discriminator convolutionally across an image and averaging the responses. This discrimiator acts as a for of texture or style loss, and unless noted otherwise, our experiments use 70 x 70 PatchGANs. Figure below shows the architecture of PatchGAN.
![alt text](https://github.com/bethanystate/CS7641_project/blob/master/images/disc1.png?raw=true)

##### PixelGAN
A 1x1 PixelGAN has small receptive fields because it is the most shawllow model. It is expected to encourage colorfulness while having no effect on spatial details, or sharpness. In image processing, this would be useful in color balancing and histogram matching of RGB images. Figure below shows the architecture of this PixelGAN.
![alt text](https://github.com/bethanystate/CS7641_project/blob/master/images/disc2.png?raw=true)

##### ImageGAN
Full 286 x 286 ImageGAN requires more parameters and greater depth than the PatchGAN because it has a much larger receptive field. It is expected that the full-sized ImaeGAN will provide use with sharper image translations, but may be a lot harder to train our model. Figure below shows the architecture of ImageGAN.
![alt text](https://github.com/bethanystate/CS7641_project/blob/master/images/disc3.png?raw=true)

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
![alt text](https://github.com/bethanystate/CS7641_project/blob/master/Results/Figure%203/Discriminator_types1.png?raw=true)
![alt text](https://github.com/bethanystate/CS7641_project/blob/master/Results/Figure%203/Discriminator_types2.png?raw=true) 
![alt text](https://github.com/bethanystate/CS7641_project/blob/master/Results/Figure%203/Discriminator_types3.png?raw=true)

#### ResNet with default config but with different padding type
 - Reflect
 - Zero
 - Symmetric
![alt text](https://github.com/bethanystate/CS7641_project/blob/master/Results/Figure%206/padding_type1.jpg?raw=true)
![alt text](https://github.com/bethanystate/CS7641_project/blob/master/Results/Figure%206/padding_type2.jpg?raw=true)
![alt text](https://github.com/bethanystate/CS7641_project/blob/master/Results/Figure%206/padding_type3.jpg?raw=true)
![alt text](https://github.com/bethanystate/CS7641_project/blob/master/Results/Figure%206/padding_type4_.jpg?raw=true) 
 
#### Different loss function
 - Binary Cross Entropy for Adversarial Loss
 - MSE for Adversarial Loss 
![alt text](https://github.com/bethanystate/CS7641_project/blob/master/Results/Figure%205/training_loss.jpg?raw=true) 

## Evaluation Metrics
## Analysis

## Conclusion and Future Work

### References
 - [1] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1125–1134, 2017.
 - [2] L. A. Gatys, A. S. Ecker, and M. Bethge, “A neural algorithm of artistic style,” arXiv pre printarXiv:1508.06576, 2015.
 - [3] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” in Proceedings of the IEEE international conference on computer vision, pp. 2223–2232, 2017.
 - [4] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” in Advances in neural information processing systems, pp. 2672–2680, 2014.
 - [5] K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, 2016, pp. 770-778
 - [6] U-Net: O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in Medical Image Computing and Computer-Assisted Intervention (MICCAI), pp. 234-241, 2015.

### Contributions
