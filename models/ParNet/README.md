# ParNet - Non-Deep Networks
---
This is an unofficial PyTorch implementation for ParNet (https://arxiv.org/pdf/2110.07641v1.pdf)

##### Official Repo
Official Repo: https://github.com/imankgoyal/NonDeepNetworks

<p align="center">
  <img src="https://raw.githubusercontent.com/hexhowells/Neural-Network-Implementations/main/models/ParNet/img/ParNet-Architecture.jpg" width=80%>
</p>

Notes
-----
- Trained on CIFAR and Caltech256 for verification
- Small modifications were made in the downsampling block as dim matching errors were occuring for certain image sizes
- Learning rate was decreased from what was specified in the paper during experiments

TODO
-----
- Add structural reparameterization (only affects model size at inference so isn't crucial)
- Look into issue with dim matching as described above
- Add modified model for CIFAR datasets
-  Add code to train and validate the networks
