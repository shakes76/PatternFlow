import matplotlib.pyplot as plt

from constants import BATCH_SIZE, TEST_BATCHES
from dataset import test_data
from utils import DSC
from train import train


if __name__ == "__main__":
    model = train()

    for x_test_batch, y_test_batch in test_data.shuffle(32).take(TEST_BATCHES):
        y_pred_batch = model.predict(x_test_batch)

        plt.figure(figsize=(64, 16), dpi=100)
        _, axes = plt.subplots(2, BATCH_SIZE)
        for ax_row, batch in zip(axes, [y_pred_batch, y_test_batch]):
            for ax, img in zip(ax_row, batch):
                ax.imshow(img)
                ax.axis("off")

        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.tight_layout()
        plt.show()

        avg_dsc = (
            sum(
                DSC(y_true, y_pred)
                for y_true, y_pred in zip(y_test_batch, y_pred_batch)
            )
            / BATCH_SIZE
        )
        print(f"Average DSC for batch is {avg_dsc}")
