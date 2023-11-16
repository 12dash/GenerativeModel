# Normalazing Flow
## Introduction
The basic equation which lead to normalizing flow called as **change of variable**.  
Consider two random variable $Z$ and $X$, where $X$ is the underlying distribution of your data. 
$f$ is a **invertible** transformation, $f : X \rightarrow Z$
$$
p_{x}(x) = p_{z}(z) | \det(\frac{\partial z}{\partial x}) |
$$

## References
Below are few references for Normalizing Flow models
* [NICE](https://arxiv.org/pdf/1410.8516.pdf)
* [R-NVP](https://arxiv.org/pdf/1605.08803.pdf)