from nibabel.testing import data_path
import os
import re

raw_images = os.listdir(data_dir)
raw_images = [fname for fname in raw_images if fname.endswith('nii.gz')]
raw_images.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
print(raw_images)
example_filename = [os.path.join(data_dir, fname) for fname in raw_images]

import nibabel as nib
img = nib.load(example_filename[0])
all_shapes =set([nib.load(img).shape for img in example_filename])

# Case_019_Week1_LFOV.nii.gz has diff shape ()

for f in example_filename:
    if nib.load(f).shape != (256, 256, 128):
        print(f)

