from pathlib import Path
import pytorch_lightning as pl


def train_kfold_dataset(model, config, data, checkpoint_path: Path):

    trainer = pl.Trainer(**config["trainer"])
    # tuner = Tuner(trainer)
    trainer.fit(model, datamodule=data, ckpt_path=checkpoint_path)
