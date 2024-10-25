<img src="/pics/logo.png" width="950px"> 


--------------------------------------------------------------------------------

### This repository is for paper "Gaussian-Based Swap Operator for Context-Aware Extraction of Building Boundary Vectors"


> We implement it by CUDA/C++, therefore, in here, pls following the steps to install G-Swap v1 and v2
 
```
To get started, clone the repository and install the required dependencies:

git clone git@github.com:moulelin/G-Swap.git
cd GSwap
pip install -r requirements.txt
```

## Why we need CUDA?

- Implement sophisticated or new <u>**operator** </u> that doesn't provide by Pytorch API
  ![](pics/1.png)
    
    The `resources` you can refer:

    - [1. Pytorch source](https://pytorch.org/cppdocs/api/namespace_at.html#namespace-at)
        
    
    - [2. Pytorch C++ API](https://pytorch.org/cppdocs/)
        
- Compile and install the extension manually.

## Install G-Swap v1
```bash
cd swapv1
```

```bash
python setup.py install
```

Note: If you are using Windows, please open the Command Prompt in administrator mode instead of the default mode.
> The G-Swap operator is now available in your Python libraries, and you can call it just like any other library.
> However, G-Swap depends on PyTorch and requires importing "torch" before importing "Swap"

## run the demo

```bash
cd ..
```

```bash
python demo_swapv1.py
```

In Swap v1, the diagonal positions of the second half of the channels are swapped.

```angular2html
 [[ 16.,  17.,  18.,  19.],
  [ 20.,  21.,  22.,  23.],
  [ 24.,  25.,  26.,  27.],
  [ 28.,  29.,  30.,  31.]],

 [[ 37.,  36.,  39.,  38.],
  [ 33.,  32.,  35.,  34.],
  [ 45.,  44.,  47.,  46.],
  [ 41.,  40.,  43.,  42.]]
```

The position of 32 and 37 are exchanged

### This is not perfect enough, therefore, we have developed Swap v2

## Install Swap v2 (exactly described in our paper)
```bash
cd swapv2
```

```bash
# install Swap v2
python setup.py install 
```

And then you can run the demo
```angular2html
cd ..
```

```bash
python demo_swapv2.py
```

Noticed that:
```python
swap_layer = SwapV2(p = 0.5)
```

In here, this Hyperparameter is the P that controls the proportion of Swap operation

You will get the following results:
```python

[[[[-0.6075, -0.6932, -0.5155,  ..., -0.5917, -0.5663, -0.6142],
          [-0.5535, -0.5088, -0.7040,  ..., -0.4965, -0.5851, -0.7346],
          [-0.5116, -0.5622, -0.6748,  ..., -0.5330, -0.5504, -0.5115],
          ...,
          [-0.7200, -0.6353, -0.5884,  ..., -0.7607, -0.5803, -0.7851],
          [-0.7521, -0.5087, -0.6758,  ..., -0.6489, -0.4367, -0.4643],
          [-0.6328, -0.5792, -0.5460,  ..., -0.6432, -0.5136, -0.6763]],

         [[ 0.5172,  0.5455,  0.4892,  ...,  0.5839,  0.5977,  0.5850],
          [ 0.5798,  0.5557,  0.6446,  ...,  0.5386,  0.4897,  0.5708],
          [ 0.5802,  0.5729,  0.5786,  ...,  0.5463,  0.4929,  0.5484],
          ...,
          [ 0.6124,  0.4958,  0.5831,  ...,  0.6083,  0.5401,  0.6652],
          [ 0.6336,  0.5592,  0.6479,  ...,  0.5064,  0.4413,  0.4674],
          [ 0.5484,  0.4718,  0.5957,  ...,  0.6358,  0.5919,  0.5199]],

         [[-0.6075, -0.6932, -0.5155,  ..., -0.5917, -0.5663, -0.6142],
          [-0.5535, -0.5088, -0.7040,  ..., -0.4965, -0.5851, -0.7346],
          [-0.5116, -0.5622, -0.6748,  ..., -0.5330, -0.5504, -0.5115],
          ...,
          [-0.7200, -0.6353, -0.5884,  ..., -0.7607, -0.5803, -0.7851],
          [-0.7521, -0.5087, -0.6758,  ..., -0.6489, -0.4367, -0.4643],
          [-0.6328, -0.5792, -0.5460,  ..., -0.6432, -0.5136, -0.6763]]],
            .
            .
            .
         [[-0.7052, -0.7899, -0.7223,  ..., -0.6803, -0.5802, -0.5135],
          [-0.5952, -0.6952, -0.4695,  ..., -0.6289, -0.5145, -0.7129],
          [-0.4810, -0.5645, -0.4635,  ..., -0.7883, -0.6338, -0.4545],
          ...,
          [-0.7487, -0.5313, -0.5946,  ..., -0.5234, -0.6946, -0.5478],
          [-0.7056, -0.6804, -0.4585,  ..., -0.7165, -0.4861, -0.5106],
          [-0.5284, -0.6190, -0.6047,  ..., -0.6925, -0.4353, -0.7202]]]], device='cuda:0', grad_fn=<CatBackward0>)

```