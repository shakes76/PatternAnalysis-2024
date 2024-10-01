import logging

from torch.utils.data import DataLoader
from modules import TumorTrainer, HyperParams
from dataset import TumorPairDataset

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.DEBUG)
    hparams = HyperParams()
    trainer = TumorTrainer(hparams)
    dataset = TumorPairDataset(
        pathlib.Path("data/train"), pathlib.Path("data/pairs.csv")
    )
    train_loader = DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=hparams.batch_size,
        num_workers=2,
        drop_last=True,
    )


if __name__ == "__main__":
    main()
