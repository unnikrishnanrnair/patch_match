import os 

from torch import optim, nn, utils
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
import torchvision as tv
from pytorch_lightning.callbacks import ModelCheckpoint

from model import ClassifierArcface, Classifier
from dataset_utils.dataset_train_classifier import PatchMatchTrainClassifier
from dataset_utils.dataset_val_classifier import PatchMatchValClassifier

class PatchMatch(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ClassifierArcface()
        self.loss = nn.CrossEntropyLoss()

    def training_step(self, batch):
        img, labels = batch
        preds = self.model(img)
        loss = self.loss(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        preds = self.model(img)
        loss = self.loss(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log("val_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

def main():
    # init the pl object
    model_patch_match = PatchMatch()
    # setup data
    dataset_train = PatchMatchTrainClassifier('patches_matching_data/train/patches/')
    train_loader = utils.data.DataLoader(dataset_train, batch_size=64, num_workers=16, shuffle=True, persistent_workers=True)
    dataset_val = PatchMatchValClassifier('patches_matching_data/test/patches/')
    val_loader = utils.data.DataLoader(dataset_val, batch_size=64, num_workers=16, persistent_workers=True)
    # train the model
    save_folder = 'model_saves/classifier_arcface_version_0'
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_folder),
        mode="max",
        save_top_k=1,
        every_n_epochs=1,
        monitor="val_acc",
        save_weights_only=True,
    )
    trainer = pl.Trainer(
        max_epochs=25,
        accelerator="gpu",
        callbacks=checkpoint_callback,
        precision=16,
        logger=pl.loggers.TensorBoardLogger(
            save_dir=os.path.join(save_folder)
        ),
    )
    trainer.fit(model=model_patch_match, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()