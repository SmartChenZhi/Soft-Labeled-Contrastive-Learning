from datetime import datetime

import numpy as np
import torch

from trainer.Trainer_Causal import Trainer_Causal


def summarize_scores(tag, results):
    if "dc" not in results:
        print(f"{tag}: no dice scores returned")
        return
    dc = results["dc"]
    if len(dc) == 2:
        mean_dc = dc[0]
    else:
        mean_dc = np.round((dc[0] + dc[2] + dc[4]) / 3, 3)
    print(f"{tag} mean dice: {mean_dc}")
    # print(f"{tag} raw dc: {dc}")


def main():
    trainer = Trainer_Causal()

    if trainer.args.restore_from is None:
        raise ValueError("Please provide the trained checkpoint with -restore_from.")

    checkpoint = torch.load(trainer.args.restore_from, map_location=trainer.device)
    model_state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    trainer.segmentor.load_state_dict(model_state, strict=False)
    trainer.segmentor.eval()

    results_valid = trainer.eval(modality="target", phase="valid", toprint=True)
    summarize_scores("Target valid", results_valid)

    results_test = trainer.eval(modality="target", phase="test", toprint=True)
    summarize_scores("Target test", results_test)

    if trainer.args.train_with_s:
        results_source = trainer.eval(modality="source", phase="test", toprint=True)
        summarize_scores("Source test", results_source)

    trainer.writer.close()


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    print("Time elapsed: {}".format(datetime.now() - start_time))
    print("program finish")
