import os 

from torch import optim, nn, utils
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
import torchvision as tv
from pytorch_lightning.callbacks import ModelCheckpoint

from model import TripletLossModel
from dataset_utils.dataset_train_triplet import PatchMatchTrainTriplet

class PatchMatch(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = TripletLossModel()
        self.loss = nn.TripletMarginLoss()

    def training_step(self, batch):
        img1, img2, img3 = batch
        a, p, n = self.model(img1, img2, img3)
        loss = self.loss(a, p, n)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

def main():
    # init the pl object
    model_patch_match = PatchMatch()
    # setup data
    dataset_train = PatchMatchTrainTriplet('patches_matching_data/train/patches/')
    train_loader = utils.data.DataLoader(dataset_train, batch_size=64, num_workers=16, shuffle=True, persistent_workers=True)
    # train the model
    save_folder = 'model_saves/triplet_version_0'
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_folder),
        mode="min",
        save_top_k=1,
        every_n_epochs=1,
        monitor="train_loss",
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
    trainer.fit(model=model_patch_match, train_dataloaders=train_loader)

if __name__ == "__main__":
    main()