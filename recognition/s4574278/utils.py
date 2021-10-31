import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches


def draw_bbox(image, boxes):
    tableau_light = (
        np.array(
            [
                (158, 218, 229),
                (219, 219, 141),
                (199, 199, 199),
                (247, 182, 210),
                (196, 156, 148),
                (197, 176, 213),
                (255, 152, 150),
                (152, 223, 138),
                (255, 187, 120),
                (174, 199, 232),
            ]
        )
        / 255
    )

    plt.imshow(image)

    def plot_bbox(bbox, color):
        plt.gca().add_patch(
            patches.Rectangle(
                xy=(bbox[0], bbox[1]),
                width=bbox[2] - bbox[0],
                height=bbox[3] - bbox[1],
                fill=False,
                edgecolor=color,
                linewidth=2,
            )
        )

    for i, box in enumerate(boxes):
        plot_bbox(box, tableau_light[i % 10])
