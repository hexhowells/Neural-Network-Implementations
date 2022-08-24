from prettytable import PrettyTable
import numpy as np
import torchvision.transforms as transforms


def model_summary(model):
    print("##### {} #####".format(model.__class__.__name__))
    
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for layer, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([layer, param])
        total_params += param

    print(table)
    print(f"Total Trainable Params: {total_params:,}\n")
    return total_params


inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

def renderize(x):
    x = inv_normalize(x)
    x = x.detach().numpy()
    x = x[0, :, :, :]  # remove batch dimension
    x = np.moveaxis(x, 0, -1)  # move colour channel as last channel
    x = np.clip(x, 0, 1)

    return x