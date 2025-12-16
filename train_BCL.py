from trainer.Trainer_BCL import Trainer_BCL
from datetime import datetime


def main():
    trainer_bcl = Trainer_BCL()
    trainer_bcl.train()


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    print('Time elapsed: {}'.format(datetime.now() - start_time))
    print('program finish')