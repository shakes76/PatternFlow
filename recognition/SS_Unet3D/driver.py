import nibabel as nib
from model import UNetCSIROMalePelvic

the_model = UNetCSIROMalePelvic("My Model")

print(the_model.mdl.summary())

#for layer in the_model.mdl.layers:
#    print(layer.input_shape, '-->', layer.name, '-->', layer.output_shape)