from configs.experiment_config import experiment_cfg
from executors.trainer import Trainer


if __name__ == '__main__':
    trainer = Trainer(experiment_cfg)

    # one batch overfitting
    trainer.batch_overfit()

    # model training
    # trainer.fit()
