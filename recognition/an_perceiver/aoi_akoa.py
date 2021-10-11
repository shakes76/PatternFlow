"""AOI AKOA Knee MRI Images Dataset.

The dataset has no pre-defined splits, these have instead been assigned by:
- 70/15/15 for train/validation/test
- balanced laterality for test
- remaining population laterality ratios for train and validation

Patients (identified by the OAI prefix in the filename) are assigned one split.

@author Anthony North
"""

import itertools
import re
import tensorflow as tf
import tensorflow_datasets as tfds

# TODO(aoi_akoa): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
"""

# TODO(aoi_akoa): BibTeX citation
_CITATION = """
"""

# test patient ids
_TEST = [
    # fmt: off
    9031930, 9251077, 9252629, 9099363, 9258864, 9282203, 9325209, 9298541,
    9369286, 9302260, 9453364, 9443223, 9519044, 9465298, 9560051, 9581241,
    9890414, 9014797, 9036287, 9055361, 9062645, 9081858, 9275309, 9210230,
    9357383, 9278158, 9423086, 9362264, 9574271
    # fmt: on
]

# validation patient ids
_VALIDATION = [
    # fmt: off
    9479978, 9375317, 9819744, 9844581, 9849372, 9502938, 9281187, 9806950,
    9171097, 9406033, 9658152
    # fmt: on
]


class AoiAkoa(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for aoi_akoa dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Download processed OAI AOKOA Knee MRI Images from UQ Blackboard. Place the
    `akoa_analysis.zip` in `manual_dir`
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(
                        shape=(228, 260, 1), encoding_format="png"
                    ),
                    "label": tfds.features.ClassLabel(names=["left", "right"]),
                    "patient_id": tf.int32,
                    "baseline": tf.int32,
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://nda.nih.gov/oai/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        # open archive at manual download dir
        archive_path = dl_manager.manual_dir / "akoa_analysis.zip"
        archive = dl_manager.iter_archive(archive_path)
        examples = list(self._generate_examples(archive))
        subset_archive = lambda predicate: filter(lambda x: predicate(x[1]), examples)

        return {
            "train": subset_archive(
                lambda x: x["patient_id"] not in itertools.chain(_VALIDATION, _TEST)
            ),
            "validation": subset_archive(lambda x: x["patient_id"] in _VALIDATION),
            "test": subset_archive(lambda x: x["patient_id"] in _TEST),
        }

    def _generate_examples(self, archive):
        """Yields examples."""

        for _, img in archive:
            name = img.name.split("/")[-1]
            yield name, {
                "image": img,
                "label": self._get_laterality(name),
                "patient_id": self._get_patient_id(name),
                "baseline": self._get_baseline(name),
            }

    def _get_laterality(self, name: str) -> str:
        """Get knee laterality from image name."""

        is_left = re.search(r"left\.nii|l_e_f_t\.nii", name, re.IGNORECASE)
        is_right = re.search(r"right\.nii|r_i_g_h_t\.nii", name, re.IGNORECASE)

        assert is_left or is_right
        return "left" if is_left else "right"

    def _get_patient_id(self, name: str) -> int:
        """Get patient identifier from image name."""

        match = re.search(r"(?<=OAI)\d+", name, re.IGNORECASE)
        assert match is not None

        return int(match.group(0))

    def _get_baseline(self, name: str) -> int:
        """Get baseline from image name."""

        match = re.search(r"(?<=Baseline_)\d+", name, re.IGNORECASE)
        assert match is not None

        return int(match.group(0))
