from os import path

import pandas as pd

from configs.data_config import DataConfig
from configs.experiment_config import ExperimentConfig
from data_loaders.data_loader import DataLoader
from dataset.emotions_dataset import EmotionsDataset
from executors.trainer import Trainer
from transforms.image_transforms import Sequential
from utils.enums import SetType


def _predict(trainer: Trainer) -> None:
    data_config = DataConfig()
    transforms = Sequential(data_config.eval_transforms)
    data_loader = DataLoader(
        dataset=EmotionsDataset(
            data_config, SetType.TEST, transforms=transforms,
        ),
        batch_size=ExperimentConfig.TRAIN_BATCH_SIZE,
        shuffle=False,
    )
    trainer.load('model_0.463.pickle')
    predictions, paths = trainer.predict(data_loader)
    converted_paths = [
        f'test/{path.basename(image_path)}' for image_path in paths
    ]
    df = pd.DataFrame({'ID': converted_paths, 'prediction': predictions})
    df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    trainer = Trainer(ExperimentConfig())

    # one batch overfitting
    # trainer.batch_overfit()

    # model training
    # trainer.fit()
    _predict(trainer)
