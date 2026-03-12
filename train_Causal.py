from datetime import datetime

from trainer.Trainer_Causal import Trainer_Causal


def main():
    trainer_causal = Trainer_Causal()
    trainer_causal.train()


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    print("Time elapsed: {}".format(datetime.now() - start_time))
    print("program finish")
