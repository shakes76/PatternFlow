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
    """Draw the bounding box"""
    ax = plt.subplot(111)
    plot_image(image, ax)
    for i, box in enumerate(boxes):
        plot_rectangle(box, colormap[i % 10], ax)
    plt.show()


def draw_bbox_label(image, boxes, labels):
    """Draw the image then the bounding box with label on top"""
    ax = plt.gca()
    plot_image(image, ax)
    # label = '{} {:.2f}'.format(predicted_class, score)
    for i, (box, label) in enumerate(zip(boxes, labels)):
        color = colormap[i % 10]
        plot_rectangle(box, color, ax)
        x = box[0]
        y = box[1]
        ax.text(
            (x + 5),
            (y - 12),
            label,
            verticalalignment="baseline",
            color="black",
            fontsize=10,
            weight="bold",
            clip_on=False,
            fontfamily='sans-serif',
            bbox=dict(facecolor=color, edgecolor=color)
        )
    plt.show()


def plot_image(image, ax=plt.gca()):
    """Show image on the pyplot figure"""
    if torch.is_tensor(image):
        image = image.permute(1, 2, 0)
    ax.axis("off")
    return ax.imshow(image, interpolation="nearest")


def plot_rectangle(bbox, color, ax=plt.gca()):
    """Plot the holo rectangle of bounding box"""
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
