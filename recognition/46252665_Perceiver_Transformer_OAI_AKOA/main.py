from pre_process.dataset_processor import ProcessDataset


def run():
    # Create Dataset
    dataset = ProcessDataset()
    dataset.do_action()


if __name__ == '__main__':
    run()
