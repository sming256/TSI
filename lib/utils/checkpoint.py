import os
import torch


def save_checkpoint(model, epoch, cfg, best=False):
    exp_name = cfg.EXP_NAME

    state = {"epoch": epoch, "state_dict": model.module.state_dict()}
    checkpoint_dir = "./exps/%s/checkpoint/" % (exp_name)

    if not os.path.exists(checkpoint_dir):
        os.system("mkdir -p %s" % (checkpoint_dir))
    checkpoint_path = os.path.join(checkpoint_dir, "epoch_%d.pth.tar" % epoch)
    torch.save(state, checkpoint_path)

    if best:
        best_path = os.path.join(checkpoint_dir, "best.pth.tar")
        torch.save(state, best_path)
