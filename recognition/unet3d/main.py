from nibabel.testing import data_path
import os
import re
data_dir = "/Users/xiaofangchen/data/Labelled_weekly_MR_images_of_the_male_pelvis/HipMRI_study_complete_release_v1/semantic_MRs_anon"

raw_images = os.listdir(data_dir)
raw_images = [fname for fname in raw_images if fname.endswith('nii.gz')]
raw_images.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
print(raw_images)
example_filename = [os.path.join(data_dir, fname) for fname in raw_images]

import nibabel as nib
img = nib.load(example_filename[0])
all_shapes =set([nib.load(img).shape for img in example_filename])

for f in example_filename:
    if nib.load(f).shape != (256, 256, 128):
        print(f)


# print(all)
# print(img.shape)
def load_images():
    print('hi')
