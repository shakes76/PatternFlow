from matplotlib import pyplot as plt, patches
import numpy as np
import torch

##########################################################
# Visualization
##########################################################
tableau_light = np.array(
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
colormap = tableau_light / 255


def draw_bbox(image, boxes):
    """Drop the bounding box"""
    plot_image(image)
    for i, box in enumerate(boxes):
        plot_rectangle(box, colormap[i % 10])


def draw_bbox_label(image, boxes, labels):
    plot_image(image)
    ax = plt.gca()
    # label = '{} {:.2f}'.format(predicted_class, score)
    for i, (box, label) in enumerate(zip(boxes, labels)):
        color = colormap[i % 10]
        plot_rectangle(box, color)
        x = box[0]
        y = box[1]
        ax.text(
            x,
            (y - 10),
            label,
            verticalalignment="center",
            color="black",
            fontsize=10,
            weight="bold",
        ).set_bbox(dict(facecolor=color, edgecolor='transparent'))


def plot_image(image, ax=plt.gca()):
    if torch.is_tensor(image):
        image = image.permute(1, 2, 0)
    ax.axis("off")
    return ax.imshow(image, interpolation="nearest")


def plot_rectangle(bbox, color, ax=plt.gca()):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return ax.add_patch(
        patches.Rectangle(
            xy=(bbox[0], bbox[1]),
            width=width,
            height=height,
            fill=False,
            visible=True,
            edgecolor=color,
            linewidth=2,
        )
    )


if __name__ == "__main__":
    from dataset import IsicDataSet

    data = IsicDataSet("./dataset/input", "./dataset/annotation", ["lesion"])
    plt.show()

    for i in range(1, 2):
        # draw_bbox(data[i][0], data[i][1])
        draw_bbox_label(data[i][0], data[i][1], ["Test"])
