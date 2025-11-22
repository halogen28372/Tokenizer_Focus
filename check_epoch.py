
import torch

def check_epoch():
    checkpoint = torch.load('checkpoints_ebt_s2/best.pt', map_location='cpu', weights_only=False)
    print(f"Best checkpoint is from epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Best val acc: {checkpoint.get('s2_pixel_acc', 'unknown')}")

if __name__ == "__main__":
    check_epoch()

