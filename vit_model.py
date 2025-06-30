import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTClassifier(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Φορτώνουμε το προκαταρτισμένο (pretrained) ViT-B_16 από το torchvision
        if pretrained:
            # Από torchvision >=0.13, ορίζουμε weights=ViT_B_16_Weights.IMAGENET1K_V1
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.vit = vit_b_16(weights=None)

        # Το αρχικό head (fully-connected) είναι για 1000 κλάσεις (ImageNet).
        # Το αντικαθιστούμε με ένα Linear που επιστρέφει 2 logits (COVID / non-COVID).
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor σχήματος (B, 3, 224, 224) – επαναλαμβάνουμε το grayscale 3 φορές
           ή φορτώνουμε ήδη RGB εικόνα.
        Επιστρέφει logits (B, 2).
        """
        return self.vit(x)
    

