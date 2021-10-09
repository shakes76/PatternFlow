import nibabel as nib
from model import UNetCSIROMalePelvic

the_model = UNetCSIROMalePelvic("My Model")

print(the_model.mdl.summary())
