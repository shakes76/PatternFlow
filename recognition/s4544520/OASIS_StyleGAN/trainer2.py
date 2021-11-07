import os
import math
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import Dataset
from torchvision import transforms, utils, datasets
from PIL import Image
# from PGNetwork import Generator, Discriminator
from StyleGANNetwork import Generator, Discriminator
from tensorboardX import SummaryWriter

# Path config
img_dir = os.path.join("./", "brainMRI")
img_path = [os.path.join(img_dir, i) for i in os.listdir(img_dir)]


class Trainset(Dataset):
    """
    build torch's Dataset to load and do some transform about images.
    """

    def __init__(self, img_size):
        """
        Dataset's constructor
        Args:
            img_size: The image size in different stage (1 stage -> 4; 2 stage -> 8)
        """
        self.images_path = img_path
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])  # do some preprocessing

    def __getitem__(self, index):
        """
        get each batch size's images
        Args:
            index: the index of batch size

        Returns:
        one batch size' images
        """
        img = Image.open(self.images_path[index]).convert("L")
        img_tensor = self.transform(img)
        return img_tensor

    def __len__(self):
        """
        get the batch size's number
        Returns: the number of batch size

        """
        return len(self.images_path)


class Trainer:
    """
    train and eval network
    """

    def __init__(self, arg):
        self.nc = arg.nc
        self.nz = arg.nz
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.init_size = arg.init_size
        self.output_size = arg.size
        self.stage_epoch = arg.stage_epoch
        self.lr = arg.lr
        self.batch_size = arg.batch_size
        # build generator and Discriminator
        self.G = Generator(nc=self.nc, nz=self.nz, size=self.output_size).to(self.device)
        self.D = Discriminator(nc=self.nc, nz=self.nz, size=self.output_size).to(self.device)
        self.G_cpu_eval = copy.deepcopy(self.G).to('cpu')  # copy G and deliver it to cpu to do eval
        # init optimizer
        self.g_optim = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0, 0.99), eps=1e-8, weight_decay=0.)
        self.d_optim = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0, 0.99), eps=1e-8, weight_decay=0.)
        self.writer = SummaryWriter(self.output_size)  # init tensorboard
        self.dataset = Trainset(self.init_size)
        # length = len(self.dataset)
        # train_size, validate_size = int(24), int(length - 24)
        # self.dataset, validate_set = torch.utils.data.random_split(self.dataset, [train_size, validate_size])
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                      shuffle=True, pin_memory=True, drop_last=True
                                                      )
        self.counts = self.stage_epoch * len(self.dataloader)  # progressive growth's counter

    def gradient_penalty(self, real, fake):
        """
        To implement the gradient penalty for WGAN-GP
        Args:
            real: dataset's real label
            fake: generate by Generator

        Returns:
            the value of gradient_penalty

        """
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        interpolates = alpha * real.detach() + (1 - alpha) * fake.detach()
        interpolates = torch.autograd.Variable(
            interpolates, requires_grad=True)

        descision_interpolates = self.D(interpolates)
        gradients = torch.autograd.grad(outputs=descision_interpolates,
                                        inputs=interpolates,
                                        grad_outputs=torch.ones_like(descision_interpolates).to(self.device),
                                        create_graph=True,
                                        retain_graph=True,
                                        )
        gradients = gradients[0].view(batch_size, -1) # resize
        gradients = gradients.norm(2, dim=1) # weight norm
        gradient_penalty = (gradients - 1.0).pow(2.).mean()
        return gradient_penalty

    def update_moving_average(self, decay=0.999):
        # update exponential running average (EMA) for the weights of the generator
        # W_EMA_t = decay * W_EMA_{t-1} + (1-decay) * W_G
        with torch.no_grad():
            # param_dict_G = dict(self.G.module.named_parameters())
            param_dict_G = dict(self.G.named_parameters())

            for name, param_EMA in self.G_cpu_eval.named_parameters():
                param_G = param_dict_G[name]
                param_EMA.copy_(decay * param_EMA + (1. - decay) * param_G.detach().cpu())

    def loss(self, label):
        """
         Loss zoo.This function can be used to calculate network's loss (G loss and D loss both use WGAN loss)
         and a metric that measures the distance between the generated distribution and the true distribution
         named Wasserstein distance (W_dis).
        Args:
            label: The label to train (real img)

        Returns:
            generator's loss, discriminator' loss and Wasserstein distance
        """
        self.G.train()
        self.D.train()
        # D loss setting:
        self.D.zero_grad()
        self.d_optim.zero_grad()
        # real label D
        real_d = self.D(label)
        real_d_loss = real_d.mean().mul(-1.)  # WGAN loss
        # fake D
        z = torch.FloatTensor(self.batch_size, 512, 4, 4).normal_(0.0, 1.0).to(self.device)
        fake = self.G(z)
        fake_d = self.D(fake.detach())
        loss_d_fake = fake_d.mean()
        gp = self.gradient_penalty(label, fake)
        # WGAN Loss + gradient_penalty + epsilon_penalty
        # https://github.com/tkarras/progressive_growing_of_gans/blob/master/loss.py
        D_loss = real_d_loss + loss_d_fake + 10.0 * gp + real_d.pow(2.).mean() * 0.001
        W_dist = real_d_loss.item() + real_d_loss.item()  # Wasserstein distance
        D_loss.backward()
        self.d_optim.step()
        # G loss setting
        self.G.zero_grad()
        self.g_optim.zero_grad()

        z = torch.FloatTensor(self.batch_size, 512, 4, 4).normal_(0.0, 1.0).to(self.device)
        fake = self.G(z)
        fake_d = self.D(fake)
        G_loss = fake_d.mean().mul(-1.)  # WGAN loss
        G_loss = G_loss.to(self.device)
        G_loss.backward()
        self.g_optim.step()
        return G_loss.item(), D_loss.item(), W_dist

    def concat_factor_update(self, counter):
        """
        Use progressive growth to concat last stage's img and this stage's result by one factor (varies dynamically
        with the training process)
        Args:
            counter: a counter to record the progressive timming
        Returns:
            Every timming's factor (0<=factor<=1)

        """
        dfactor = 1. / self.counts  # ensure progressive growth
        if counter < self.counts:
            self.G.model.residual_module.increase_dfactor(dfactor)
            self.D.model.residual_module.increase_dfactor(dfactor)
            self.G_cpu_eval.model.residual_module.increase_dfactor(dfactor)

        factor = self.G.model.residual_module.get_factor()
        return factor

    def train(self):
        """
        This function can be called to start train and eval the network.
        How to grow can read my ReadMe.md

        """
        x_axis = 0  # record the train
        num_stage = int(math.log2(self.output_size) - 1)  # calculate all stages need to grow
        for stage in range(1, num_stage + 1):
            current_stage_size = int(4 * pow(2, (stage - 1)))  # stage's img size
            stage_epoch = self.stage_epoch
            self.dataset = Trainset(current_stage_size)  # rebuild dataset to suit for different stages
            # length = len(self.dataset)
            # train_size, validate_size = int(24), int(length - 24)
            # self.dataset, validate_set = torch.utils.data.random_split(self.dataset, [train_size, validate_size])
            self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,num_workers=8,
                                                          shuffle=True, pin_memory=True, drop_last=True
                                                          )

            if stage == 1:  # stage 1 don't need to grow
                pass
            elif stage == 2:
                self.G.grow_network()
                self.D.grow_network()
                self.G_cpu_eval.grow_network()
                self.G = self.G.cuda()
                self.D = self.D.cuda()
                # create a new optim to suit new params
                self.g_optim = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0, 0.99), eps=1e-8, weight_decay=0.)
                self.d_optim = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0, 0.99), eps=1e-8, weight_decay=0.)
            else:
                # move to next stage(>=2) need to remove residual concat module first and add new RGB module
                self.G.flush_network()
                self.D.flush_network()
                self.G_cpu_eval.flush_network()
                # normal grow
                self.G.grow_network()
                self.D.grow_network()
                self.G_cpu_eval.grow_network()
                # create new optim
                self.G = self.G.cuda()
                self.D = self.D.cuda()
                self.g_optim = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0, 0.99), eps=1e-8, weight_decay=0.)
                self.d_optim = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0, 0.99), eps=1e-8, weight_decay=0.)

            count = 0  # record the timming about concat factor
            real_epoch = stage_epoch
            if stage == 1:
                real_epoch = stage_epoch
            else:
                real_epoch = 2 * stage_epoch  #last 10 stage to use factor=1 to grow
            for epoch in range(real_epoch):  # each stage should be trained some epochs
                for index, current_stage_label in enumerate(self.dataloader):  # this stage 's img_dataloder
                    concat_factor = 0
                    img = current_stage_label

                    if stage != 1:
                        concat_factor = self.concat_factor_update(count)
                        last_stage_size_img = F.avg_pool2d(img, 2)  # avg_pool to get more general info
                        last_stage_for_current_stage = F.interpolate(last_stage_size_img, scale_factor=2.,
                                                                     mode='nearest')  # resize
                        label = (1 - concat_factor) * last_stage_for_current_stage + concat_factor * img
                        label = label.to(self.device)
                    else:
                        label = img.to(self.device)

                    # label = label.mul(2.).sub(1.)
                    G_loss, D_loss, W_dis = self.loss(label)
                    count += 1
                    x_axis += 1
                    # self.update_moving_average()

                    # val the result every 10 index
                    if index % 10 == 0:
                        print(
                            "Stage {}/{} epoch {}/{} G_loss {:4f} D_loss{:4f} W_dis {:4f}" \
                                .format(stage, num_stage, epoch + 1, stage_epoch, G_loss, D_loss, W_dis)
                        )
                        self.writer.add_scalar('train/G_loss', G_loss, x_axis)
                        self.writer.add_scalar('train/D_loss', D_loss, x_axis)
                        self.writer.add_scalar('train/W_dis', W_dis, x_axis)

                # val
                with torch.no_grad():
                    print("start to do eval")
                    fixed_z = torch.FloatTensor(self.batch_size, 512, 4, 4).normal_(0.0, 1.0)
                    self.G_cpu_eval.eval()
                    fake = self.G_cpu_eval(fixed_z)
                    generate = utils.make_grid(fake, nrow=4, normalize=True, scale_each=True)  # create grid to save
                    self.writer.add_image('stage_{}/fake'.format(stage), generate, epoch)
                    print("Save params")
                    state = {
                        'G': self.G.state_dict()
                    }
                    torch.save(state, os.path.join("./logs", 'stage{}Params'.format(stage)))
