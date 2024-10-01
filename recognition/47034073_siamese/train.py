import logging

from modules import TumorTrainer

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.DEBUG)

    trainer = TumorTrainer()


if __name__ == "__main__":
    main()
