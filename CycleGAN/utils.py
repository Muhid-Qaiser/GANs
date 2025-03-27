import random, torch, os, numpy as np
import torch.nn as nn
import config
import copy


from torchvision.utils import save_image
def save_some_examples(gen_H, gen_Z, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen_H.eval()
    gen_Z.eval()
    with torch.no_grad():
        y_fake = gen_H(x) # x is Zebra, Convert it into Horse
        x_fake = gen_Z(y_fake) # y_fake is generate Horse, Convert it into Zebra
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        x_fake = x_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/Horse_gen_{epoch}.png")
        save_image(x_fake, folder + f"/Zebra_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_Zebra_{epoch}.png")
    gen_H.train()
    gen_Z.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False