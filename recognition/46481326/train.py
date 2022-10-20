__author__ = "James Chen-Smith"

# %%
"""Import libraries required for PyTorch"""
from modules import *
from dataset import *
import torch # Import PyTorch
import torch.nn as nn # Import PyTorch Neural Network
import torchvision # Import PyTorch Vision

# %%
class Trainer():
    """Trainer Class
    """
    def __init__(self):
        self.device = self.init_device() # Initialize device
        self.print_device() # Print selected device.
        self.hp = Hyperparameters() # Initialize hyperparameters
        self.data = DataManager(self.hp.channels_image, self.hp.size_batch_vqvae, shuffle=True)
        self.path_state = "recognition\\46481326\\state"
        try: # Try to load last model state
            self.model_vqvae = torch.load(self.path_state + "\\vqvae.txt")
        except (FileNotFoundError): # If model doesn't exist create model
            self.model_vqvae = VQVAE(self.hp.channels_image, self.hp.channels_out, self.hp.channels_out_res, self.hp.len_e, self.hp.size_e, self.hp.threshold_loss)
        self.model_vqvae.to(self.device) # Send model to device
        self.fn_optim = torch.optim.Adam(self.model_vqvae.parameters(), lr=self.hp.rate_learn_vqvae)
        
    def train_vqvae(self, model_vqvae, dataloader, fn_optim):
        """Train and save VQVAE model

        Args:
            model (_type_): Model to be trained
            dataloader (__type__): DataLoader to train on
            fn_optim (_type_): Optimizer function
        """
        size_dataloader = len(dataloader.dataset)
        list_loss = list()
        list_loss_cum = list()
        for epoch in range(self.hp.num_epoch_vqvae):
            for batch, (x, _) in enumerate(dataloader):
                x = x.to(self.device)
                fn_optim.zero_grad()
                loss_vq, x_decoded = model_vqvae(x)
                error_x_decoded = self.hp.fn_loss(x_decoded, x) / self.hp.variance
                loss = error_x_decoded + loss_vq
                loss.backward()
                fn_optim.step()
                
                list_loss.append(error_x_decoded.item())
                list_loss_avg = sum(list_loss)/len(list_loss)
                list_loss_cum.append(list_loss_avg)
                
                print(f"Epoch = [{epoch + 1}], Batch = [{batch}/{size_dataloader}], Reconstruction Loss = [{list_loss_avg}]")
                
                if (batch % self.hp.interval_batch_save_vqvae == 0):
                    torch.save(self.model_vqvae, self.path_state + "\\vqvae.txt")
                    print("Saved VQVAE")
                    
        torch.save(self.model_vqvae, self.path_state + "\\vqvae.txt")
        print("Saved VQVAE")
        
    def train_dcgan(self, model_vqvae):
        """Train and save DCGAN model

        Args:
            model (_type_): VQVAE model
        """
        transform_dcgan = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        dataset_X_train_dcgan = DatasetDCGAN("recognition\\46481326\\oasis\\train", model_vqvae, self.device, transform_dcgan)
        dataloader_X_train_dcgan = torch.utils.data.DataLoader(dataset_X_train_dcgan, self.hp.size_batch_dcgan)
        
        try: # Try to load last model state
            generator = torch.load(self.path_state + "\\generator.txt")
        except (FileNotFoundError): # If model doesn't exist create model
            generator = Generator(self.hp.size_z_dcgan, 3).to(self.device)
            self.init_weights_dcgan(generator)
        try: # Try to load last model state
            discriminator = torch.load(self.path_state + "\\discriminator.txt")
        except (FileNotFoundError): # If model doesn't exist create model
            discriminator = Discriminator(3).to(self.device)
            self.init_weights_dcgan(discriminator)
            
        fn_optim_generator = torch.optim.Adam(generator.parameters(), lr=self.hp.rate_learn_vqvae, betas=(0.5, 0.999))
        fn_optim_discriminator = torch.optim.Adam(discriminator.parameters(), lr=self.hp.rate_learn_vqvae, betas=(0.5, 0.999))
        fn_loss = nn.BCELoss()
        
        for epoch in range(self.hp.num_epoch_dcgan):
            for batch, x in enumerate(dataloader_X_train_dcgan):
                x = x.to(self.device)
                z = torch.randn(self.hp.size_batch_dcgan, self.hp.size_z_dcgan, 1, 1).to(self.device)
                
                """Train Discriminator"""
                generator_fake = generator(z) # Generate fake
                discriminator_real = discriminator(x).reshape(-1) # Discriminate real
                loss_discriminator_real = fn_loss(discriminator_real, torch.ones_like(discriminator_real)) # Compute discriminated real loss
                discriminator_fake = discriminator(generator_fake.detach()).reshape(-1) # Discriminate fake
                loss_discriminator_fake = fn_loss(discriminator_fake, torch.zeros_like(discriminator_fake)) # Compute discriminated fake loss
                loss_discriminator = (loss_discriminator_real + loss_discriminator_fake) / 2
                discriminator.zero_grad()
                loss_discriminator.backward()
                fn_optim_discriminator.step()
                
                """Train Generator"""
                discriminator_fake = discriminator(generator_fake).reshape(-1)
                loss_generator = fn_loss(discriminator_fake, torch.ones_like(discriminator_fake))
                generator.zero_grad()
                loss_generator.backward()
                fn_optim_generator.step()
                
            print(
                f"Epoch = [{epoch + 1}/{self.hp.num_epoch_dcgan}]\n"
                f"Discriminator Loss = [{loss_discriminator:.4f}], Generator Loss = [{loss_generator:.4f}]"
            ) # Print stats to console
            torch.save(generator, self.path_state + "\\generator.txt")
            torch.save(discriminator, self.path_state + "\\discriminator.txt")
            print("Saved DCGAN")
                    
        torch.save(generator, self.path_state + "\\generator.txt")
        torch.save(discriminator, self.path_state + "\\discriminator.txt")
        print("Saved DCGAN")
        
    def init_weights_dcgan(self, dcgan_model):
        for m in dcgan_model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
        
    def init_device(self):
        """Initiates and returns the PyTorch device that is being used

        Returns:
            device: The device that is used for computation
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device
    
    def print_device(self):
        """Prints the PyTorch device that is being used.
        """
        print(f"Device Selected = [{self.device}]")
        
    def is_trained_vqvae(self):
        try: # Try to load last model state
            torch.load(self.path_state + "\\vqvae.txt")
        except (FileNotFoundError): # If model doesn't exist create model
            return False
        return True
    
    def is_trained_dcgan(self):
        try: # Try to load last model state
            torch.load(self.path_state + "\\generator.txt")
        except (FileNotFoundError): # If model doesn't exist create model
            return False
        try: # Try to load last model state
            torch.load(self.path_state + "\\discriminator.txt")
        except (FileNotFoundError): # If model doesn't exist create model
            return False
        return True