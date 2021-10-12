from pre_process.process_dataset import ProcessDataset


def run():
    # Create Dataset
    dataset = ProcessDataset()
    dataset.do_action()


if __name__ == '__main__':
    run()
