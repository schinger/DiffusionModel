[TOC]

## 1. 背景介绍

生成模型是机器学习领域中一个重要的研究方向，其目标是学习数据的概率分布，并能够生成新的、与训练数据同分布的数据样本。近年来，生成模型在图像生成、视频生成、文本创作、蛋白质发现等领域取得了显著的成果，并展现出巨大的应用潜力。

传统的生成模型（自回归模型（Autoregressive ）在另外一篇文章介绍），如变分自编码器（VAE）和生成对抗网络（GAN），各有其优缺点。朴素的VAE 通常会生成较为模糊的图像，而 GAN 则面临着训练不稳定和模式坍塌等问题。

在这样的背景下，Diffusion Model 应运而生。它基于一个优雅的思想：**通过逐渐添加高斯噪声将数据转换为近似纯噪声，然后学习逆转这个过程，将噪声逐步恢复成新的数据样本。**

Diffusion Model 的优势在于：

* **生成质量高：** Diffusion Model 在图像生成方面表现出色，能够生成高质量、高分辨率的图像。
* **训练稳定性好：** 相比 GAN，Diffusion Model 的训练过程更加稳定，不容易出现模式坍塌等问题。
* **理论基础扎实：** Diffusion Model 的理论基础来源于非平衡热力学，具有良好的理论特性。


**Diffusion Model 的发展历程可以追溯到 2015 年的论文 “Deep Unsupervised Learning using Nonequilibrium Thermodynamics” 。** 该论文提出了基于扩散过程的生成模型框架，但真正引起广泛关注的是 “Denoising Diffusion Probabilistic Models” 论文的发表，该论文提出了一种非常简洁的 Diffusion 算法，并通过实验证明了其在图像生成方面的优越性。

**近年来，Diffusion Model 发展迅速，涌现出许多改进算法和应用场景，例如 Stable Diffusion系列、DALL-E 系列、SoRA系列等。** 这些模型的成功进一步证明了 Diffusion Model 的强大能力。


接下来，我们将简单描述 Diffusion Model 的原理，并结合代码进行深入分析。
## 2. Diffusion Model 简要原理

Diffusion Model 的核心思想可以概括为两个阶段 （*公式推导* 部分会详细论述所有公式的来龙去脉）：

### 2.1 前向扩散 (Forward Diffusion Process)
在这个阶段，我们逐步向原始数据中添加高斯噪声，使其最终变成近似纯噪声。这个过程可以看作是一个马尔可夫链，每一步都只依赖于前一步的状态。

假设  $x_0$  表示原始数据，  $x_t$  表示在时间步  $t$  添加噪声后的数据，  $T$  表示总的时间步数。那么前向扩散过程可以用如下公式表示：


$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
$$

其中，$\beta_t$  是时间步  $t$  的噪声方差，通常是一个单调递增的序列。 $\mathcal{N}(x; \mu, \Sigma)$ 表示均值为 $\mu$，方差为 $\Sigma$ 的高斯分布。

上面的公式可以等价的写成如下形式 (记 $\alpha_t = 1 - \beta_t$)：
$$
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_t
$$

**这个公式直接表达了向样本$x_{t-1}$添加噪声的$\epsilon_t$过程，通过递推运算我们可以得到任意时间步 $t$ 的  $x_t$：**


$$
x_t = \sqrt{\bar\alpha_t} x_{0}  + \sqrt{1 - \bar\alpha_t} \epsilon
$$

其中，$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$， $\epsilon$以及$\epsilon_t$服从标准高斯分布。

### 2.2 反向扩散 (Reverse Diffusion Process)
在这个阶段，我们学习一个模型来逆转前向扩散过程，将纯噪声逐步恢复成新的数据样本：


$$
p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

其中，$\mu_\theta(x_t, t)$ 和 $\Sigma_\theta(x_t, t)$  是模型的预测，分别表示在时间步  $t$  的均值和方差，DDPM这篇文章简单起见另$\Sigma_\theta(x_t,t) = \sigma^2_t I$为常量，其中$\sigma_t^2 = \beta_t$或者$\sigma_t^2 = \tilde\beta_t =  \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\beta_t$ (本文的实现)。

**训练目标是使得反向扩散过程生成的样本分布尽可能接近原始数据的分布。** 根据后续的推导，另:
$$
\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t))
$$
训练目标简化为优化模型参数  $\theta$，使模型预测添加至样本的噪声：


$$
L_\text{simple}(\theta) = \mathbb{E}_{t,x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_{0}  + \sqrt{1 - \bar\alpha_t} \epsilon, t) \|^2 \right]
$$

其中，$\epsilon$ 是前向扩散过程中添加的噪声，$\epsilon_\theta(\sqrt{\bar\alpha_t} x_{0}  + \sqrt{1 - \bar\alpha_t} \epsilon, t) =\epsilon_\theta(x_t, t)$ 是模型预测的噪声。

训练完毕后，我们可以采样$ x_{t-1} \sim p_\theta(x_{t-1}|x_t)$, 即：
$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t))+\sigma_tz
$$

其中，$z$是标准高斯分布。根据上面的公式，我们即可得到DDPM的训练及采样算法：
<img src="images/Algorithm.png" alt="Algorithm" width="1500"/>

在下一节中，我们将结合代码来详细讲解 Diffusion Model 的实现细节。

## 3. 代码实现

本节将结合代码详细讲解 Diffusion Model 的实现细节，并展示如何使用该模型生成爱心和 MNIST 手写数字图像。**不同于传统的 Diffusion Model 使用 U-Net 作为模型结构，我们使用最简单的 MLP来实现，非常简洁，总共不到 300 行代码：https://github.com/schinger/DiffusionModel**

### 3.1 数据集准备

代码中提供了两个数据集：`perfect_heart_dataset` 和 `mnist_dataset`，分别用于生成 2D 心形数据（每个点都是一个样本）和 MNIST 手写数字图像。

* **2D 爱心数据：**  通过[数学公式](https://mathworld.wolfram.com/HeartCurve.html)生成一个完美的爱心（够浪漫吧！）。

* **MNIST 手写数字图像：**  使用 PyTorch 自带的 torchvision.datasets.MNIST 加载 MNIST 数据集，并转化为一维向量。



### 3.2 模型结构

我们使用一个简单的 `MLP` 。模型的输入是带有噪声的数据 $x_t$ 和时间步 $t$，输出是预测的噪声 $\epsilon_\theta(x_t, t)$。

模型结构如下：

* **时间嵌入 (Time Embedding):**  将时间步 $t$ 转换为一个向量，使用 Sinusoidal Embedding 方法。

* **数据嵌入 (Data Embedding):**  对于 2D 数据，将每个坐标 (x, y) 分别转换为一个固定维度的向量，[同样使用 Sinusoidal Embedding 方法](https://arxiv.org/abs/2006.10739)。对于 MNIST 数据，直接将图像展平为一个向量。

* **`MLP` (Joint MLP):**  将时间嵌入和数据嵌入拼接在一起，输入到一个 MLP 中，最后输出预测的噪声。


### 3.3 扩散过程

代码中定义了一个 `Diffusion` 类来实现前向扩散和反向扩散过程。

* **前向扩散 (`add_noise`):**  根据公式  $x_t = \sqrt{\bar\alpha_t} x_{0} + \sqrt{1 - \bar\alpha_t} \epsilon$  向数据中添加噪声。

* **反向扩散 (`sample_step`):**  根据公式  $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t))+\sigma_tz$ 从噪声中逐步恢复数据。



### 3.4 训练过程

训练过程主要包括以下步骤：

1. 从数据集中随机采样一个 batch 的数据。
2. 随机选择时间步 $t$。
3. 根据前向扩散公式，向数据中添加噪声。
4. 将带有噪声的数据和时间步输入到模型中，预测噪声。
5. 计算预测噪声和真实噪声之间的均方误差 (MSE) 损失。$\| \epsilon - \epsilon_\theta(x_t, t) \|^2$
6. 反向传播损失，更新模型参数。



### 3.5 生成样本

训练完成后，我们可以使用模型来生成新的样本。

1. 从标准高斯分布中采样随机噪声。
2. 从 $T$ 到 $1$ 迭代，每一步都使用模型预测噪声，并根据反向扩散公式从噪声中逐步恢复数据。
3. 最后得到的数据即为生成的样本。



### 3.6 结果动画展示

* **2D 心形数据正反向扩散：**

![2D Data Generation](images/animation_2d.gif)

* **MNIST 手写数字图像生成：**

<img src="images/animation_mnist.gif" alt="drawing" width="500"/>


**通过以上代码和结果展示，我们可以看到，Diffusion Model 的实现并不复杂，使用简单的 MLP 模型也可以取得不错的效果。**


接下来，我们深入探讨 Diffusion Model 的公式推导，帮助读者更深入地理解其背后的数学原理。


## 4. 公式推导

为了深入理解 Diffusion Model 的工作原理，本节将从头开始推导其背后的数学公式，只要求读者具备基本的概率论知识。

### 4.1 变分下界 (Variational Lower Bound) 的推导

我们的目标是学习数据的概率分布，并能够生成新的样本。由于直接优化 $p_\theta(x_0)$ 比较困难，我们对$-\log p_\theta(x_0)$进行一些推导，考虑到Diffusion Model为隐变量模型（其中$x_{1:T}$为隐变量），且KL散度非负的性质:

$$\begin{aligned}
- \log p_\theta({x}_0) 
&\leq - \log p_\theta({x}_0) + D_\text{KL}(q({x}_{1:T}\vert{x}_0) \| p_\theta({x}_{1:T}\vert{x}_0) ) \\
&= -\log p_\theta({x}_0) + \mathbb{E}_{{x}_{1:T}\sim q({x}_{1:T} \vert {x}_0)} \Big[ \log\frac{q({x}_{1:T}\vert{x}_0)}{p_\theta({x}_{0:T}) / p_\theta({x}_0)} \Big] \\
&= -\log p_\theta({x}_0) + \mathbb{E}_q \Big[ \log\frac{q({x}_{1:T}\vert{x}_0)}{p_\theta({x}_{0:T})} + \log p_\theta({x}_0) \Big] \\
&= \mathbb{E}_q \Big[ \log \frac{q({x}_{1:T}\vert{x}_0)}{p_\theta({x}_{0:T})} \Big] \\
&= L
\end{aligned}
$$

$-L$即为传说中的Variational Lower Bound(VLB)，又称为Evidence Lower Bound(ELBO)(应用Jensen不等式也可以推出同样的结果)：

$$
\log p_\theta({x}_0) \geq -L
$$

我们的优化目标即为最小化$L$，对其进行一系列恒等变换：

$$
\begin{aligned}
L
&= \mathbb{E}_{q} \Big[ \log\frac{q({x}_{1:T}\vert{x}_0)}{p_\theta({x}_{0:T})} \Big] \\
&= \mathbb{E}_q \Big[ \log\frac{\prod_{t=1}^T q({x}_t\vert{x}_{t-1})}{ p_\theta({x}_T) \prod_{t=1}^T p_\theta({x}_{t-1} \vert{x}_t) } \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta({x}_T) + \sum_{t=1}^T \log \frac{q({x}_t\vert{x}_{t-1})}{p_\theta({x}_{t-1} \vert{x}_t)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta({x}_T) + \sum_{t=2}^T \log \frac{q({x}_t\vert{x}_{t-1})}{p_\theta({x}_{t-1} \vert{x}_t)} + \log\frac{q({x}_1 \vert {x}_0)}{p_\theta({x}_0 \vert {x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta({x}_T) + \sum_{t=2}^T \log \Big( \frac{q({x}_{t-1} \vert {x}_t, {x}_0)}{p_\theta({x}_{t-1} \vert{x}_t)}\cdot \frac{q({x}_t \vert {x}_0)}{q({x}_{t-1}\vert{x}_0)} \Big) + \log \frac{q({x}_1 \vert {x}_0)}{p_\theta({x}_0 \vert {x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta({x}_T) + \sum_{t=2}^T \log \frac{q({x}_{t-1} \vert {x}_t, {x}_0)}{p_\theta({x}_{t-1} \vert{x}_t)} + \sum_{t=2}^T \log \frac{q({x}_t \vert {x}_0)}{q({x}_{t-1} \vert {x}_0)} + \log\frac{q({x}_1 \vert {x}_0)}{p_\theta({x}_0 \vert {x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta({x}_T) + \sum_{t=2}^T \log \frac{q({x}_{t-1} \vert {x}_t, {x}_0)}{p_\theta({x}_{t-1} \vert{x}_t)} + \log\frac{q({x}_T \vert {x}_0)}{q({x}_1 \vert {x}_0)} + \log \frac{q({x}_1 \vert {x}_0)}{p_\theta({x}_0 \vert {x}_1)} \Big]\\
&= \mathbb{E}_q \Big[ \log\frac{q({x}_T \vert {x}_0)}{p_\theta({x}_T)} + \sum_{t=2}^T \log \frac{q({x}_{t-1} \vert {x}_t, {x}_0)}{p_\theta({x}_{t-1} \vert{x}_t)} - \log p_\theta({x}_0 \vert {x}_1) \Big] \\
&= \mathbb{E}_q [\underbrace{D_\text{KL}(q({x}_T \vert {x}_0) \parallel p_\theta({x}_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_\text{KL}(q({x}_{t-1} \vert {x}_t, {x}_0) \parallel p_\theta({x}_{t-1} \vert{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta({x}_0 \vert {x}_1)}_{L_0} ]
\end{aligned}
$$
其中，第五行等式利用了马尔科夫性和贝叶斯公式：
$$
q(x_t\vert{x}_{t-1}) = q(x_t\vert{x}_{t-1},x_0) = \frac{q(x_{t-1},x_t\vert{x}_0)}{q(x_{t-1}\vert{x}_0)} = \frac{q(x_{t-1}\vert{x}_t,x_0)q(x_t\vert{x}_0)}{q(x_{t-1}\vert{x}_0)}
$$
接下来，我们分析 $L_T$、$L_{t-1}$ 和 $L_0$ 的具体形式。
首先是 $L_T$，由于反向扩散过程从标准高斯分布中开始采样， $p_\theta(x_T)$ 是一个高斯分布，其实与$\theta$无关，因此 $L_T$ 为常数，可以忽略掉。
### 4.2 $L_{t-1}$ 的推导

我们先来计算$q({x}_{t-1} \vert {x}_t, {x}_0)$：
$$
\begin{aligned}
q({x}_{t-1} \vert {x}_t, {x}_0) 
&= q({x}_t \vert {x}_{t-1}, {x}_0) \frac{ q({x}_{t-1} \vert {x}_0) }{ q({x}_t \vert {x}_0) } \\
&\propto \exp \Big(-\frac{1}{2} \big(\frac{({x}_t - \sqrt{\alpha_t} {x}_{t-1})^2}{\beta_t} + \frac{({x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} {x}_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{({x}_t - \sqrt{\bar{\alpha}_t} {x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp \Big(-\frac{1}{2} \big(\frac{{x}_t^2 - 2\sqrt{\alpha_t} {x}_t {{x}_{t-1}} {+ \alpha_t} {{x}_{t-1}^2} }{\beta_t} + \frac{ {{x}_{t-1}^2} {- 2 \sqrt{\bar{\alpha}_{t-1}} {x}_0} {{x}_{t-1}} {+ \bar{\alpha}_{t-1} {x}_0^2}  }{1-\bar{\alpha}_{t-1}} - \frac{({x}_t - \sqrt{\bar{\alpha}_t} {x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp\Big( -\frac{1}{2} \big( {(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} {x}_{t-1}^2 - {(\frac{2\sqrt{\alpha_t}}{\beta_t} {x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} {x}_0)} {x}_{t-1} { + C({x}_t, {x}_0) \big) \Big)}
\end{aligned}
$$
其中，$C({x}_t, {x}_0)$ 与 ${x}_{t-1}$ 无关，可以忽略掉。可以看到，$q({x}_{t-1} \vert {x}_t, {x}_0)$ 也是一个高斯分布，记其为：
$$
q({x}_{t-1} \vert {x}_t, {x}_0) = \mathcal{N}({x}_{t-1}; {\tilde{{\mu}_t}}({x}_t, {x}_0), {\tilde{\beta}_t} {I})
$$
其中：
$$
\begin{aligned}
\tilde{\beta}_t 
&= 1/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) 
= 1/(\frac{\alpha_t - \bar{\alpha}_t + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})})
= {\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
\tilde{{\mu}}_t ({x}_t, {x}_0)
&= (\frac{\sqrt{\alpha_t}}{\beta_t} {x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} {x}_0)/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) \\
&= (\frac{\sqrt{\alpha_t}}{\beta_t} {x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} {x}_0) {\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} {x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} {x}_0\\
\end{aligned}
$$
回忆前向扩散过程的公式（利用高斯分布和仍然是高斯分布的性质进行递推）：
$$
\begin{aligned}
{x}_t 
&= \sqrt{\alpha_t}{x}_{t-1} + \sqrt{1 - \alpha_t}{\epsilon}_{t} \\
&= \sqrt{\alpha_t \alpha_{t-1}} {x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{{\epsilon}} \\
&= \dots \\
&= \sqrt{\bar{\alpha}_t}{x}_0 + \sqrt{1 - \bar{\alpha}_t}{\epsilon} 
\end{aligned}
$$
反解出${x}_{0}$，带入$\tilde{{\mu}_t}({x}_t, {x}_0)$：
$$
\begin{aligned}
\tilde{{\mu}}_t({x}_t, {x}_0)
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} {x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \frac{1}{\sqrt{\bar{\alpha}_t}}({x}_t - \sqrt{1 - \bar{\alpha}_t}{\epsilon}) \\
&= {\frac{1}{\sqrt{\alpha_t}} \Big( {x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} {\epsilon} \Big)}
\end{aligned}
$$
我们再来看$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma^2_t I)$，由于后面要计算两者的KL散度$D_\text{KL}(q({x}_{t-1} \vert {x}_t, {x}_0) \parallel p_\theta({x}_{t-1} \vert{x}_t))$，我们可以**简单的**将${\mu}_\theta({x}_t, t)$建模为类似的形式(同时，我们立即得到采样算法`Algorithm 2`)：
$$
\begin{aligned}
{\mu}_\theta({x}_t, t) &= {\frac{1}{\sqrt{\alpha_t}} \Big( {x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} {\epsilon}_\theta({x}_t, t) \Big)} 
\end{aligned}
$$
两个高斯分布的KL散度有解析解：
$$
\begin{aligned}
L_{t-1} &= \mathbb{E}_{q}\left[ \frac{1}{2\sigma_t^2} \|\tilde{\mu}_{t}(x_t,x_0) - \mu_{\theta}(x_t, t)\|^2 \right] + C \\
&=\mathbb{E}_{{x}_0, \epsilon}\left[ \frac{\beta_t^2}{2\sigma_t^2\alpha_t (1-\bar{\alpha}_t)} \left\| \epsilon - \epsilon_{\theta}(x_t, t) \right\|^2 \right] + C \\
&=\mathbb{E}_{{x}_0, \epsilon}\left[ \frac{\beta_t^2}{2\sigma_t^2\alpha_t (1-\bar{\alpha}_t)} \left\| \epsilon - \epsilon_{\theta}(\sqrt{\bar{\alpha}_t} {x}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t) \right\|^2 \right] + C \\
\end{aligned}
$$
其中，$C$是与$\theta$无关的常数项。
### 4.3 $L_0$ 的推导
我们将$p_\theta(x_0 \vert x_1)$依据$\mathcal{N}(x_{0}; \mu_\theta(x_1, 1), \sigma^2_1 I)$拆解成按数据维度独立连乘的形式：
$$
\begin{aligned}
p_\theta({x}_0 | {x}_1) &= \prod_{i=1}^{D} \int_{\delta_-(x_0^i)}^{\delta_+(x_0^i)} \mathcal{N}(x; \mu_\theta^i({x}_1, 1), \sigma_1^2) dx  \\
\delta_+(x) &= \begin{cases}
\infty & \text{if } x=1\\
x+\frac{1}{255} & \text{if } x<1
\end{cases} 
\qquad
\delta_-(x) = \begin{cases}
-\infty & \text{if } x=-1\\
x-\frac{1}{255} & \text{if } x>-1
\end{cases}
\end{aligned}
$$
对于图像来讲，$D$为像素点个数，我们假设像素点的取值在0到255之间，将其线性映射到$[-1,1]$区间。
我们可以得到：
$$
\begin{aligned}
\log p_\theta({x}_0 | {x}_1) &= \sum_{i=1}^{D} \log \int_{\delta_-(x_0^i)}^{\delta_+(x_0^i)} \mathcal{N}(x; \mu_\theta^i({x}_1, 1), \sigma_1^2) dx \\
&\approx \sum_{i=1}^{D} \log \frac{1}{\sqrt{2\pi}\sigma_1}  \exp\left(-\frac{(x_0-\mu_\theta^i({x}_1, 1))^2}{2\sigma_1^2}\right)\frac{2}{255} \\
&= \sum_{i=1}^{D} \frac{-1}{2\sigma_1^2}(x_0-\mu_\theta^i({x}_1, 1))^2 + C \\
&= \frac{-1}{2\sigma_1^2}\|x_0-\mu_\theta(x_1, 1)\|^2 + C
\end{aligned}
$$
带入：
$$
x_0 = \frac{1}{\sqrt{\alpha_1}}(x_1-\sqrt{1-{\alpha}_1}\epsilon)
$$
$$
\mu_\theta(x_1, 1) = \frac{1}{\sqrt{\alpha_1}}(x_1-\frac{1-\alpha_1}{\sqrt{1-{\alpha}_1}}\epsilon_\theta(x_1,1))
$$
得到：
$$
L_0 = \mathbb{E}_{{x}_0, \epsilon}\left[ \frac{\beta_1}{2\sigma_1^2\alpha_1} \left\| \epsilon - \epsilon_{\theta}(x_1, 1) \right\|^2 \right] + C
$$
与$L_{t-1}$当$t=1$时形式相同。
### 4.4 最终形式
忽略前面的系数，我们得到简化后的最终优化目标 `Algorithm 1`：
$$
L_\text{simple}(\theta) = \mathbb{E}_{t,x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_{0}  + \sqrt{1 - \bar\alpha_t} \epsilon, t) \|^2 \right]
$$
在实践中，忽略掉系数的简化形式效果往往更好。有一个简单的解释是，简化版本相当于**down-weight**了$t$比较小时的项，这样模型更加关注于学习$t$比较大的项：加入的噪音比较大，更加值得学习。至此，我们完成了Diffusion Model的推导。

## 5. 延伸阅读

我们上面仅讨论了Diffusion Model的最朴素形式，实际上，Diffusion Model有很多改进和延伸。首先是更实用的条件生成，我们改而建模$p_\theta(x \vert y)$，其中$y$可以是任意的条件信息，例如文本描述、图像等等，目前比较简单的做法是[Classifier-Free Guidance](https://openreview.net/forum?id=qw8AKxfYbI)。其次，朴素的Diffusion Model生成样本时，要执行$T$次反向扩散过程，这在计算上是非常昂贵的，因此有很多工作提出了更高效的方法，例如，减少采样次数的[DDIM](https://arxiv.org/abs/2010.02502)； 在更小的潜在空间而不是原始的像素空间上执行Diffusion的[LDM](https://arxiv.org/abs/2112.10752)等。另外，从模型结构的角度，除了传统的U-Net，目前发现更通用更能Scale的[Diffusion Transformer](https://arxiv.org/abs/2212.09748)等。应用方面，Diffusion Model除了在图像/视频生成，超分辨率生成上效果出众外，还在蛋白质预测（AlphaFold 3），音乐创作，机器人控制等方面有成功的应用。最后是理论方面，形式上，它与[denoising score matching](https://arxiv.org/abs/1907.05600)，[Langevin dynamics](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf)，  [Markovian Hierarchical Variational Autoencoder](https://arxiv.org/pdf/2208.11970)， [stochastic differential equation](https://openreview.net/forum?id=PxTIG12RRHS)有诸多类似和联系，感兴趣的读者可以进一步阅读。
