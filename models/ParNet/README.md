# ParNet - Non-Deep Networks
This is an unofficial PyTorch implementation for ParNet (https://arxiv.org/pdf/2110.07641v1.pdf)

<p align="center">
  <img src="https://raw.githubusercontent.com/hexhowells/Neural-Network-Implementations/main/models/ParNet/img/ParNet-Architecture.jpg" width=80%>
</p>

#### Official Repo
https://github.com/imankgoyal/NonDeepNetworks

#### Citation
```
@article{goyal2021nondeep,
  title={Non-deep Networks},
  author={Goyal, Ankit and Bochkovskiy, Alexey and Deng, Jia and Koltun, Vladlen},
  journal={arXiv:2110.07641},
  year={2021}
}
```

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
- Add pre-trained models
