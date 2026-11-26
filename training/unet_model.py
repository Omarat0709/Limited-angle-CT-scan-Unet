"""
FBPConvNet-style U-Net for limited-angle CT artifact removal.
============================================================

WHAT THIS FILE IS
-----------------
This is the neural network itself -- the thing that learns to turn a streaky,
limited-angle FBP reconstruction into a clean image. It contains no training
logic and no data loading; it is purely the model definition.

THE IDEA IN ONE PARAGRAPH
-------------------------
A limited-angle FBP image and the true clean image are *mostly the same* -- the
phantom is in roughly the right place, it is just corrupted by streaks and
missing-wedge blur. So instead of asking the network to redraw the whole image
from scratch, we ask it to predict only the CORRUPTION (the "residual"), and we
add that correction back onto the input. This is the central trick of
FBPConvNet (Jin et al., 2017, "Deep Convolutional Neural Network for Inverse
Problems in Imaging") and it makes the network much easier to train.

THE SHAPE OF THE NETWORK (a "U-Net")
------------------------------------
A U-Net is shaped like the letter U:
  * The left side (the ENCODER) repeatedly shrinks the image while growing the
    number of feature channels -- it compresses "what is in the image" into a
    small, deep representation. Shrinking lets later layers "see" large regions
    of the image at once, which is what you need to recognise a streak that runs
    across the whole picture.
  * The right side (the DECODER) does the reverse: it grows the image back to
    full resolution while shrinking the channels, rebuilding a full-size output.
  * The horizontal bars of the U (the SKIP CONNECTIONS) hand each encoder level's
    detailed feature map straight across to the matching decoder level. Without
    them, fine detail lost during shrinking could never be recovered.

Our encoder has 4 downsampling steps, so the spatial size goes
256 -> 128 -> 64 -> 32 -> 16, and the channel width doubles each step
64 -> 128 -> 256 -> 512 -> 1024. (256 must be divisible by 16; it is.)

This module does NOT import TIGRE -- it works only on the .npy image pairs
produced by generate_dataset.py.
"""

import torch
import torch.nn as nn


def _conv_block(in_ch, out_ch):
    """One U-Net 'double convolution' block: (conv -> BN -> ReLU) done twice.

    * Conv2d 3x3, padding=1 keeps the spatial size unchanged; it is the layer
      that actually learns image features (edges, textures, streak patterns).
      bias=False because the BatchNorm right after it has its own shift term,
      so a conv bias would be redundant.
    * BatchNorm2d normalises each channel across the batch. It stabilises and
      speeds up training -- the network is far less sensitive to learning rate.
    * ReLU is the nonlinearity (max(0, x)); without a nonlinearity, stacking
      convs would collapse into a single linear operation and learn nothing
      interesting. inplace=True just saves a little memory.

    Two conv layers per block (rather than one) is the standard U-Net design:
    it gives each level more representational power for a modest cost.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class FBPConvNet(nn.Module):
    """U-Net with a global residual connection (FBPConvNet, Jin et al. 2017).

    Args:
        in_channels:  channels in the input image. 1 for grayscale CT.
        out_channels: channels in the output image. 1.
        base_channels: feature width of the FIRST encoder level (64 in the
            paper). Every deeper level doubles it. Halve this to 32 if you ever
            run out of GPU memory -- it roughly quarters the memory use.
        residual: if True (default), output = input + network(input), i.e. the
            network learns the correction. Set False to train a plain U-Net that
            predicts the clean image directly -- useful as an ablation to *prove*
            in the thesis that the residual connection helps.
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 residual=True):
        super().__init__()                 # required for any nn.Module
        self.residual = residual
        c = base_channels                  # shorthand; c = 64 by default

        # ---- Encoder (the left, downward side of the U) ----
        # Each enc* block doubles the channel count. The pooling that halves the
        # spatial size happens in forward(), not here.
        self.enc1 = _conv_block(in_channels, c)        # operates at 256x256
        self.enc2 = _conv_block(c, c * 2)              # operates at 128x128
        self.enc3 = _conv_block(c * 2, c * 4)          # operates at  64x64
        self.enc4 = _conv_block(c * 4, c * 8)          # operates at  32x32
        self.pool = nn.MaxPool2d(2)        # 2x2 max-pool: halves H and W

        # ---- Bottleneck (the bottom of the U) ----
        # Smallest spatial size, largest channel count -- the most "abstract"
        # representation, where the network has the widest view of the image.
        self.bottleneck = _conv_block(c * 8, c * 16)   # operates at 16x16

        # ---- Decoder (the right, upward side of the U) ----
        # Each level: ConvTranspose2d UPSAMPLES (doubles H and W and halves the
        # channels), we then concatenate the matching encoder feature map along
        # the channel axis (the skip connection), then a conv block fuses them.
        #
        # Note the channel bookkeeping for each dec block: after up-conv we have
        # Nc channels, concatenation with the encoder skip (also Nc) gives 2Nc,
        # which is why each dec block's input is twice its output.
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = _conv_block(c * 16, c * 8)   # 8c (up) + 8c (skip e4) = 16c in
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = _conv_block(c * 8, c * 4)    # 4c + 4c = 8c in
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = _conv_block(c * 4, c * 2)    # 2c + 2c = 4c in
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = _conv_block(c * 2, c)        # c + c = 2c in

        # ---- Output projection ----
        # 1x1 conv mixes the final c feature channels down to a single image
        # channel. No activation here -- the output is a real-valued image
        # (or, with residual=True, a real-valued correction), not a probability.
        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x):
        """Run one forward pass. x has shape (Batch, 1, 256, 256)."""
        # --- Encoder path ---
        # Each line: apply the conv block, then (for deeper levels) pool first.
        # We KEEP e1..e4 in variables because the decoder needs them as skips.
        e1 = self.enc1(x)                  # (B,  c, 256, 256)
        e2 = self.enc2(self.pool(e1))      # (B, 2c, 128, 128)
        e3 = self.enc3(self.pool(e2))      # (B, 4c,  64,  64)
        e4 = self.enc4(self.pool(e3))      # (B, 8c,  32,  32)

        # --- Bottleneck ---
        b = self.bottleneck(self.pool(e4)) # (B, 16c, 16, 16)

        # --- Decoder path ---
        # For each level: upsample, concat the saved encoder map (dim=1 is the
        # channel axis), then fuse with a conv block. torch.cat is what physically
        # implements the skip connection.
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))  # back to 32x32
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # back to 64x64
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # back to 128x128
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # back to 256x256

        out = self.out_conv(d1)            # (B, 1, 256, 256)

        # --- Global residual skip (the FBPConvNet trick) ---
        # 'out' is the predicted correction; adding the original input x turns it
        # into the predicted clean image. With residual=False we skip this and
        # treat 'out' as the clean image directly.
        if self.residual:
            out = out + x
        return out


def count_parameters(model):
    """Total number of trainable weights -- a quick measure of model size."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Run `python training/unet_model.py` to self-test the model in isolation:
    # it should print matching input/output shapes and a parameter count near
    # 31 million for base_channels=64. (Requires PyTorch, not TIGRE.)
    net = FBPConvNet()
    dummy = torch.randn(2, 1, 256, 256)    # a fake batch of 2 blank images
    y = net(dummy)
    print("input ", tuple(dummy.shape))
    print("output", tuple(y.shape))
    print(f"trainable parameters: {count_parameters(net):,}")
