import matplotlib.pyplot as plt

from constants import BATCH_SIZE, TEST_BATCHES
from dataset import test_data
from utils import DSC
from train import train


if __name__ == "__main__":
    model = train()

    total_dsc = 0

    for x_test_batch, y_test_batch in test_data.shuffle(128).take(TEST_BATCHES):
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

        total_dsc_for_this_batch = sum(
            DSC(y_true, y_pred) for y_true, y_pred in zip(y_test_batch, y_pred_batch)
        )
        total_dsc += total_dsc_for_this_batch

        print(f"Average DSC for batch is {total_dsc_for_this_batch / BATCH_SIZE}")

    print(
        f"Average DSC across all test batches is {total_dsc / (TEST_BATCHES * BATCH_SIZE)}"
    )
