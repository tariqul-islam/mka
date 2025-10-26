# MKA: Manifold-approximated Kernel Alignment

## Install

```
git clone https://github.com/tariqul-islam/mka
cd mka
pip install .
```

Will try to add global pip install later.

## Simple Usage

```
from mka import MKA, kCKA, CKA_RBF

mka_score = MKA(X,Y) #X,Y: NxF matrix with N samples and F features
kcka_score = kCKA(X,Y) #only with RBF Kernel
cka_score = CKA(X,Y) #only with RBF kernel

```


## Citation

If you use the code, please consider citing our papers:

```
@article{islam2025mkarobust,
  title={Manifold Approximation leads to Robust Kernel Alignment},
  author={Islam, Mohammad Tariqul and Liu, Du and Sarkar, Deblina},
  journal={ArXiv}
}


@article{islam2025kernel,
  title={Kernel Alignment using Manifold Approximation},
  author={Islam, Mohammad Tariqul and Liu, Du and Sarkar, Deblina},
  journal={ICLR 2025 Re-Align Workshop}
}
```
