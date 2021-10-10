"""AOI AKOA Knee MRI Images Dataset.

@author Anthony North
"""

import tensorflow_datasets as tfds

# TODO(aoi_akoa): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
"""

# TODO(aoi_akoa): BibTeX citation
_CITATION = """
"""


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
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://nda.nih.gov/oai/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        # extract from manual download dir
        path = dl_manager.extract(dl_manager.manual_dir / "akoa_analysis.zip")

        return {
            "train": self._generate_examples(path / "akoa_analysis"),
        }

    def _generate_examples(self, path):
        """Yields examples."""

        for img_path in path.glob("*.png"):
            yield img_path.name, {
                "image": img_path,
                "label": self._get_laterality(img_path.name),
            }

    def _get_laterality(self, name: str) -> str:
        """Get knee laterality from image name."""

        _name = name.lower()
        any_in = lambda words, text: any(w in text for w in words)

        is_left = any_in(["left.nii", "l_e_f_t.nii"], _name)
        is_right = not is_left and any_in(["right.nii", "r_i_g_h_t.nii"], _name)

        assert is_left or is_right
        return "left" if is_left else "right"
