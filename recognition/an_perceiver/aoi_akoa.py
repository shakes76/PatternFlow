"""AOI AKOA Knee MRI Images Dataset.

This dataset has no pre-defined splits, so a 70/15/15 split is used from the population.
Patients (identified by the OAI prefix in the filename) are assigned one split.

@author Anthony North
"""

from itertools import accumulate, groupby, chain
import re
import tensorflow as tf
import tensorflow_datasets as tfds

# TODO(aoi_akoa): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
"""

# TODO(aoi_akoa): BibTeX citation
_CITATION = """
"""

_SPLIT_RATIOS = (0.70, 0.15, 0.15)


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

        # group by patient, ensuring no patient can fall into > 1 split
        examples_by_patient = [
            list(examples)
            for _, examples in groupby(
                self._generate_examples(archive), key=lambda x: x[1]["patient_id"]
            )
        ]

        # cumulative total of images will determine split breaks
        cumulative_lens = accumulate(map(len, examples_by_patient))
        examples_by_patient = list(zip(examples_by_patient, cumulative_lens))

        n_examples = max((x[-1] for x in examples_by_patient))
        split_thresholds = [ratio * n_examples for ratio in accumulate(_SPLIT_RATIOS)]

        # slice examples by begin & end
        slice_examples = lambda begin, end: chain.from_iterable(
            imgs
            for imgs, cumlen in examples_by_patient
            if cumlen > begin and cumlen <= end
        )

        # return train, test & validate splits
        return {
            split: slice_examples(begin, end)
            for split, begin, end in zip(
                ["train", "validation", "test"],
                [-1, *split_thresholds[:-1]],
                split_thresholds,
            )
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

        _name = name.lower()
        any_in = lambda words, text: any(w in text for w in words)

        is_left = any_in(["left.nii", "l_e_f_t.nii"], _name)
        is_right = not is_left and any_in(["right.nii", "r_i_g_h_t.nii"], _name)

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
