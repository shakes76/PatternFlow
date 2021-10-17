from Improved_Unet import build_model

model = build_model((384, 512, 3))
model.summary()