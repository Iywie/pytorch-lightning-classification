import torch
import numpy as np
from pytorch_lightning import LightningModule
from torch.nn import CrossEntropyLoss

from models.backbones.vgg import VGG
from models.heads.cls_head import VGGHead


class LitVGG(LightningModule):

    def __init__(self, cfgs):
        super().__init__()
        # ========== Grand Parameters
        self.cfg_dataset = cfgs['dataset']
        self.cfg_backbone = cfgs['backbone']
        self.cfg_train = cfgs['train']
        # ========== dataset
        self.num_classes = self.cfg_dataset['num_classes']
        # ========== backbone
        num_blocks = self.cfg_backbone['num_blocks']
        # ========== network
        self.backbone = VGG(num_blocks)
        self.head = VGGHead(512, num_classes=self.num_classes)
        # ========== train
        self.optimizer = self.cfg_train['optimizer']
        self.lr = self.cfg_train['learning_rate']
        self.momentum = self.cfg_train['momentum']
        self.loss_fn = CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        result = self.backbone(x)
        result = self.head(result)
        loss = self.loss_fn(result.view(-1, self.num_classes), y.view(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        result = self.backbone(x)
        result = self.head(result)
        preds = torch.argmax(result, dim=-1)
        return preds, y

    def validation_epoch_end(self, val):
        all_preds = np.array([])
        all_labels = np.array([])
        for batch in val:
            all_preds = np.append(all_preds, batch[0].detach().cpu().numpy())
            all_labels = np.append(all_labels, batch[1].detach().cpu().numpy())
        accuracy = (all_preds == all_labels).mean()
        print("Val Accuracy: %2.5f" % accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.lr,
                                    momentum=self.momentum)
        return optimizer
