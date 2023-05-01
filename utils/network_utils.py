import torch


def load_checkpoint(path, model, optimizer=None):
    pth = torch.load(path, map_location=torch.device('cpu'))

    model.load_state_dict(pth["state_dict"])
    if optimizer:
        optimizer.load_state_dict(pth["optimizer"])

    print("Checkpoint {} successfully loaded".format(path))

    return pth["epoch"], pth["total_iter"]


def save_checkpoint(state, path):
    torch.save(state, path)
