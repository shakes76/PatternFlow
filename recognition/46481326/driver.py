# %%
from train import Trainer
from predict import Tester

# %%
trainer = Trainer() # Create Trainer object
if (trainer.is_trained_vqvae() == False):
    trainer.train_vqvae(trainer.model_vqvae, trainer.data.get_dataloader()["X_train"], trainer.fn_optim)
if (trainer.is_trained_dcgan() == False):
    trainer.train_dcgan(trainer.model_vqvae)

# %%
tester = Tester() # Create Tester object
x_real, x_decoded = tester.decode_batch(tester.model_vqvae, tester.data.get_dataloader()["X_test"])
tester.view_batch(x_real, "recognition\\46481326\\output\\x_batch_real.png", False)
tester.view_batch(x_decoded, "recognition\\46481326\\output\\x_batch_decoded.png", False)

x_real, slice_e, slice_e_np = tester.get_slice_e(tester.model_vqvae, tester.data.get_dataloader()["X_test"])
tester.view_single_compare("Real -> Embedding Slice", x_real, slice_e_np, "recognition\\46481326\\output\\x_slice_e.png", False)

_, slice_e_decoded_np = tester.decode_slice_e(tester.model_vqvae, slice_e)
tester.view_single_compare("Embedding Slice -> Slice Reconstruction", slice_e_np, slice_e_decoded_np, "recognition\\46481326\\output\\x_slice_e_decoded.png", False)

slice_e_dcgan, slice_e_dcgan_np = tester.get_slice_e_dcgan(tester.get_generator_dcgan())
slice_e_dcgan_converted, slice_e_dcgan_converted_np = tester.convert_dcgan_slice_e(slice_e_dcgan)
tester.view_single_compare("DCGAN Generated Slice -> Embedding Mapped Slice", slice_e_dcgan_np, slice_e_dcgan_converted_np, "recognition\\46481326\\output\\x_slice_e_dcgan_converted.png", False)

slice_e_dcgan_decoded, slice_e_dcgan_decoded_np = tester.decode_slice_e_dcgan(tester.model_vqvae, slice_e_dcgan_converted)
tester.view_single_compare("Embedding Mapped Slice -> Slice Reconstruction", slice_e_dcgan_converted_np, slice_e_dcgan_decoded_np, "recognition\\46481326\\output\\x_slice_e_dcgan_decoded.png", False)

max_ssim_image = tester.print_ssim(slice_e_dcgan_decoded)
tester.view_single_compare("Slice Reconstruction vs Image w/Max SSIM", slice_e_dcgan_decoded_np, max_ssim_image, "recognition\\46481326\\output\\x_slice_e_dcgan_decoded_ssim_max.png", False)