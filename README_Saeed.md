## Formulation
In CycleGAN, there is no paired data to train on, so there is no guarantee that the input <img src="https://render.githubusercontent.com/render/math?math=X"> and the target pair <img src="https://render.githubusercontent.com/render/math?math=Y"> are meaningful during training. Thus, in order to enforcee that the network learns the correct mapping, the cycle-consistency loss is used.
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
#### Generator Architecture
##### Resnet
The architecture for the style transfer networks follows a residual network (resnet) block design having convolutional layers with skip connections that can be used when the input data and the output data have the same dimensions. When those dimensions are not the same, mapping is still possible by increasing the spatial reflection padding with zeroes in order to match the additional dimensions needed. [5] This is used to reduce artifacts around the borders of the generated image.
##### U-Net
The network architecture consists of two 3x3 convolutions (unpadded convolutions), each followed by instance normalization and a rectified linear unit (ReLU) and a pooling operation with stride 2 for downsampling an input. During upsampling, a 3 × 3 convolution with no padding reduces the size of a feature map by 1 pixel on each side, so in this case the identity connection performs a center crop on the input feature map.[6] In other words, the U-net architecture provides low-level information with a sortof shortcut across the network.

#### Discriminator Architecture
##### PatchGAN
The aim with PatchGAN  is to use the generative model in order classify whether overlapping image patches are real or fake. Because this discriminator requires fewer parameters, it works well with arbitrarily large images by running the discriminator convolutionally across an image and averaging the responses. This discrimiator acts as a for of texture or style loss, and unless noted otherwise, our experiments use 70 x 70 PatchGANs

##### PixelGAN
A 1x1 PixelGAN has small receptive fields because it is the most shawllow model. It is expected to encourage colorfulness while having no effect on spatial details, or sharpness. In image processing, this would be useful in color balancing and histogram matching of RGB images.

##### ImageGAN
Full 286 x 286 ImageGAN requires more parameters and greater depth than the PatchGAN because it has a much larger receptive field. It is expected that the full-sized ImaeGAN will provide use with sharper image translations, but may be a lot harder to train our model.

#### Training Details
In order to stabilize our training procedures, we contructed a loop that consists of four basic steps:
 - Get the predictions
 - Calculate the loss
 - Calculate the gradient using backpropogation
 - Apply the gradient to the optimizer
 
Within these training procedures, there is a random square cropping of the original images for training and a defined learning rate of <img src="https://render.githubusercontent.com/render/math?math=2e^{-4}">. However, in practice, the objective is divided by 2 while optimizing D, which slows down the rate at which D learns, relative to the rate of G. This learning rate is constant for the first 100 epochs and linearly decays to zero over the next 100 epochs.
