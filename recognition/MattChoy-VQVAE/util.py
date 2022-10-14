import os
def check_adni_dataset(dataset_fp):
    #  check that ./data/adni/test exists
    assert os.path.exists(
        os.path.join(dataset_fp, "test")
    ), "Testing dataset subdirectory doesn't exist"
    #  check that ./data/adni/train exists
    assert os.path.exists(
        os.path.join(dataset_fp, "train")
    ), "Training dataset subdirectory doesn't exist"

    #  check that ./data/adni/test/AD
    assert os.path.exists(
        os.path.join(dataset_fp, "test", "AD")
    ), "Alzheimer's dataset subdirectory doesn't exist"

    #  check that ./data/adni/test/NC
    assert os.path.exists(
        os.path.join(dataset_fp, "test", "NC")
    ), "Normal dataset subdirectory doesn't exist"

    #  check that ./data/adni/train/AD
    assert os.path.exists(
        os.path.join(dataset_fp, "train", "AD")
    ), "Alzheimer's dataset subdirectory doesn't exist"

    #  check that ./data/adni/train/NC
    assert os.path.exists(
        os.path.join(dataset_fp, "train", "NC")
    ), "Normal dataset subdirectory doesn't exist"

def get_test_train_split(dataset_fp):
    TEST_AD = len(os.listdir(os.path.join(dataset_fp, "test", "AD")))
    TEST_NC = len(os.listdir(os.path.join(dataset_fp, "test", "NC")))
    TRAIN_AD = len(os.listdir(os.path.join(dataset_fp, "train", "AD")))
    TRAIN_NC = len(os.listdir(os.path.join(dataset_fp, "train", "NC")))

    TOTAL_SAMPLES = float(TEST_AD + TEST_NC + TRAIN_AD + TRAIN_NC)

    print(f"Training samples:               {TRAIN_AD + TRAIN_NC} ({round((TRAIN_AD + TRAIN_NC) / TOTAL_SAMPLES, 4)}%)")
    print(f"Testing samples:                {TEST_AD + TEST_NC} ({round((TEST_AD + TEST_NC) / TOTAL_SAMPLES, 4)}%)")

    print(f"Normal Cognitive (NC) samples:  {TRAIN_NC + TEST_NC} ({round((TRAIN_NC + TEST_NC) / TOTAL_SAMPLES, 4)}%)")
    print(f"Alzheimer's Samples (AD):       {TRAIN_AD + TEST_AD} ({round((TRAIN_AD + TEST_AD) / TOTAL_SAMPLES, 4)}%)")