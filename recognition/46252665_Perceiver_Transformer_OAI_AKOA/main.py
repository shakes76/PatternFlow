"""
Main Driver file

@author: Pritish Roy
@email: pritish.roy@uq.edu.au
"""

from display.display_plot import PlotPatches
from perceiver.train import TransformerTrainer

from pre_process.dataset_processor import ProcessDataset


def run():
    """Sequential set of actions, comment out lines as per requirement"""

    # Create Dataset
    dataset = ProcessDataset()
    dataset.do_action()

    # Train
    transformer = TransformerTrainer()
    transformer.do_action()

    # Plot Patches
    plot = PlotPatches()
    plot.do_action()


if __name__ == '__main__':
    run()
