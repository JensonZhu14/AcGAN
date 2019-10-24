# -*- coding: utf-8 -*-
import time, os
import torchvision
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from .networks.AcGANs import Generator, Discriminator
from .utils import evaluator, ops
from torchvision.utils import save_image
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import tqdm
from models.utils.data_loader import AgeDataset
import torch.utils.data as tordata
import random
import os.path as osp
import torch.multiprocessing as mp


class AcGANsModel(object):
    def __init__(self,
                 G_lr,
                 D_lr,
                 max_epoch,
                 num_epoch_test,
                 lr_decay_epoch,
                 lr_decay,
                 D_prob_weight,
                 D_cond_weight,
                 gradient_penalty_weight,
                 mask_weight,
                 mask_smooth_weight,
                 update_netG_every_n_iter,
                 begin_save_epoch,
                 restore_epoch,
                 resume_time,
                 label_range,
                 generate_age_list,
                 model_name,
                 dataset_name,
                 save_time,
                 test_size,
                 data_root,
                 age_group,
                 batch_size,
                 image_size,
                 train_save_batch,
                 test_save_batch,
                 is_ordinal_reg,
                 **kwargs
                 ):

        self.G_lr = G_lr
        self.D_lr = D_lr
        self.max_epoch = max_epoch
        self.num_epoch_test = num_epoch_test
        self.lr_decay_epoch = lr_decay_epoch
        self.lr_decay = lr_decay
        # loss weight
        self.D_prob_weight = D_prob_weight
        self.D_cond_weight = D_cond_weight
        self.gradient_penalty_weight = gradient_penalty_weight
        self.mask_weight = mask_weight
        self.mask_smooth_weight = mask_smooth_weight
        self.update_netG_every_n_iter = update_netG_every_n_iter
        self.begin_save_epoch = begin_save_epoch

        # others
        self.restore_epoch = restore_epoch
        self.resume_time = resume_time
        self.label_range = label_range
        self.generate_age_list = generate_age_list

        self.test_size = test_size
        self.data_root = data_root
        self.age_group = age_group
        self.batch_size = batch_size
        self.image_size = image_size
        self.train_save_batch = train_save_batch
        self.test_save_batch = test_save_batch
        self.is_ordinal_reg = is_ordinal_reg
        self.dataset_name = dataset_name
        # load data
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        train_dataset = AgeDataset(self.test_size, self.dataset_name, self.age_group, self.data_root, split='train',
                                   transforms=transform)
        self.train_dataloader = tordata.DataLoader(train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   pin_memory=True)
        test_dataset = AgeDataset(self.test_size, self.dataset_name, self.age_group, self.data_root, split='test',
                                  transforms=transform)
        self.test_dataloader = tordata.DataLoader(test_dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=False,
                                                  num_workers=8,
                                                  pin_memory=True)

        self.model_name = model_name
        self.save_time = save_time
        self.cur_time = time.time()
        # test sample
        self.train_samples = []
        self.test_samples = []
        # save path for fake images
        self.images_save_dir = os.path.join('./results', self.model_name, self.dataset_name, 'save_images',
                                            self.save_time)
        os.makedirs(self.images_save_dir, exist_ok=True)
        # save path for models
        self.save_model_path = os.path.join('./results', self.model_name, self.dataset_name, 'save_models',
                                            self.save_time)
        os.makedirs(self.save_model_path, exist_ok=True)

        # networks
        self.netG = Generator(c_dim=self.age_group)
        self.netG = nn.DataParallel(self.netG).cuda()
        self.netD = Discriminator(c_dim=self.age_group, image_size=self.image_size,is_ordinal_reg=self.is_ordinal_reg)
        self.netD = nn.DataParallel(self.netD).cuda()
        # initial networks
        self.netG.apply(ops.weights_init)
        self.netD.apply(ops.weights_init)
        # optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.G_lr, betas=[0.5, 0.999], weight_decay=0)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.D_lr, betas=[0.5, 0.999], weight_decay=0)

        # lr decay
        self.scheduler_G = StepLR(self.optimizer_G, step_size=self.lr_decay_epoch, gamma=self.lr_decay)
        self.scheduler_D = StepLR(self.optimizer_D, step_size=self.lr_decay_epoch, gamma=self.lr_decay)

        # loss functions
        if self.is_ordinal_reg:
            self.D_cond_loss = evaluator.OrderRegressionLoss().cuda()
        else:
            self.D_cond_loss = evaluator.CrossEntropyLoss().cuda()



        # save loss
        self.loss_d_prob = []
        self.loss_d_cond = []
        self.loss_d_gp = []
        self.loss_g_masked_fake = []
        self.loss_g_masked_cond = []
        self.loss_g_mask = []
        self.loss_g_mask_smooth = []

    def fit(self):
        if self.restore_epoch != 0:
            self.load()

        print(
            "Loss = [loss_d_prob, loss_d_cond, loss_d_gp, loss_g_masked_fake, loss_g_masked_cond, loss_g_mask, loss_g_mask_smooth]")
        print(
            "---------------------------------------------------------------------------------------------------------------------")
        for e in range(self.restore_epoch, self.max_epoch):
            self.train(epoch=e)
            self.print_info(e, type="train")
            # if (e + 1) % self.num_epoch_test == 0:
            #     self.test()
            #     self.print_info(e, type="test")
            #     print("-----------------------------------------------------------------------------------------------")
            #     # generate fake images
            #     # self.generate_images(self.train_samples, e, sample_type='train')
            #     # self.generate_images(self.test_samples, e, sample_type='test')
            #     if (e + 1) > self.begin_save_epoch:
            #         self.save(e)
            self.scheduler_G.step()
            self.scheduler_D.step()

    def train(self, epoch):
        self.netG.train()
        self.netD.train()
        # iter_count = 0
        for iter_count, (inputs, label) in enumerate(self.train_dataloader, 1):
            # iter_count += 1
            n_iter = epoch * len(self.train_dataloader) + iter_count
            # convert tensor to variables
            real_img = inputs.cuda().float()
            real_cond = ops.group_to_one_hot(label, age_group=self.age_group).cuda().float()
            real_ordinal_cond = ops.group_to_binary(label, age_group=self.age_group).cuda().float()
            desired_cond = ops.desired_group_to_one_hot(label, age_group=self.age_group).cuda().float()
            desired_ordinal_cond = ops.desired_group_to_binary(label, age_group=self.age_group).cuda().float()

            # train D
            loss_d_prob, loss_d_cond, fake_imgs_masked = self.forward_D(real_img, real_cond, real_ordinal_cond, desired_cond)
            # combine losses
            loss_D = loss_d_prob + loss_d_cond
            # save losses
            self.loss_d_prob.append(loss_d_prob.data.cpu().numpy())
            self.loss_d_cond.append(loss_d_cond.data.cpu().numpy())
            # backward
            self.optimizer_D.zero_grad()
            loss_D.backward()
            self.optimizer_D.step()

            # gradient penalty loss
            loss_d_gp = self.gradient_penalty_D(real_img, fake_imgs_masked)
            # save loss
            self.loss_d_gp.append(loss_d_gp.data.cpu().numpy())
            # backward
            self.optimizer_D.zero_grad()
            loss_d_gp.backward()
            self.optimizer_D.step()
            # print('{}/{}/{}, loss_d_prob {:.5f}, loss_d_cond {:.5}'.format(epoch + 1, len(self.train_dataloader),
            #                                                                iter_count, loss_d_prob.item(),
            #                                                                loss_d_cond.item()))

            # train G
            if iter_count % self.update_netG_every_n_iter == 0:
                loss_g_masked_fake, loss_g_masked_cond, loss_g_mask, loss_g_mask_smooth = self.forward_G(real_img,
                                                                                                         real_cond,
                                                                                                         desired_cond,
                                                                                                         desired_ordinal_cond)
                loss_G = loss_g_masked_fake + loss_g_masked_cond + loss_g_mask + loss_g_mask_smooth
                # save losses
                self.loss_g_masked_fake.append(loss_g_masked_fake.data.cpu().numpy())
                self.loss_g_masked_cond.append(loss_g_masked_cond.data.cpu().numpy())
                self.loss_g_mask.append(loss_g_mask.data.cpu().numpy())
                self.loss_g_mask_smooth.append(loss_g_mask_smooth.data.cpu().numpy())
                # backward
                self.optimizer_G.zero_grad()
                loss_G.backward()
                self.optimizer_G.step()

            if n_iter % self.train_save_batch == 0:
                self.generate_images(real_img, n_iter, 'train')

            if n_iter % self.test_save_batch == 0:
                num = 3
                for idx, (inputs, label) in enumerate(self.test_dataloader):
                    # convert tensor to variables
                    real_img = inputs.cuda().float()
                    self.generate_images(real_img, n_iter, 'test', index=idx)
                    # print(idx)
                    if idx > num:
                        break
                # if (epoch + 1) > self.begin_save_epoch:
                self.save(n_iter)

    def test(self):
        self.netG.eval()
        self.netD.eval()
        for inputs, label in tqdm.tqdm(self.test_dataloader, desc='test'):
            # convert tensor to variables
            real_img = inputs.cuda().float()
            real_cond = ops.group_to_one_hot(label, age_group=self.age_group).cuda().float()
            real_ordinal_cond = ops.group_to_binary(label, age_group=self.age_group).cuda().float()
            desired_cond = ops.desired_group_to_one_hot(label, self.age_group).cuda().float()

            # test D
            loss_d_prob, loss_d_cond, fake_imgs_masked = self.forward_D(real_img, real_cond, real_ordinal_cond, desired_cond)
            loss_d_gp = self.gradient_penalty_D(real_img, fake_imgs_masked)
            # save losses
            self.loss_d_prob.append(loss_d_prob.data.cpu().numpy())
            self.loss_d_cond.append(loss_d_cond.data.cpu().numpy())
            self.loss_d_gp.append(loss_d_gp.data.cpu().numpy())

            # test G
            loss_g_masked_fake, loss_g_masked_cond, loss_g_mask, loss_g_mask_smooth = self.forward_G(real_img,
                                                                                                     real_cond,
                                                                                                     desired_cond,
                                                                                                     desired_ordinal_cond)
            loss_G = loss_g_masked_fake + loss_g_masked_cond + loss_g_mask + loss_g_mask_smooth
            # save losses
            self.loss_g_masked_fake.append(loss_g_masked_fake.data.cpu().numpy())
            self.loss_g_masked_cond.append(loss_g_masked_cond.data.cpu().numpy())
            self.loss_g_mask.append(loss_g_mask.data.cpu().numpy())
            self.loss_g_mask_smooth.append(loss_g_mask_smooth.data.cpu().numpy())
        # self.test_samples = real_img

    def forward_D(self, real_img, real_cond, real_ordinal_cond, desired_cond):
        # generate fake images
        fake_imgs, fake_img_mask = self.netG(real_img, desired_cond)
        fake_imgs_masked = fake_img_mask * real_img + (1 - fake_img_mask) * fake_imgs

        # D(real_I)
        d_real_img_prob, d_real_img_cond = self.netD(real_img)
        loss_d_real = self.compute_loss_D(d_real_img_prob, True) * self.D_prob_weight
        if self.is_ordinal_reg:
            loss_d_cond = self.D_cond_loss(d_real_img_cond, real_ordinal_cond) * self.D_cond_weight
        else:
            loss_d_cond = self.D_cond_loss(d_real_img_cond, real_cond) * self.D_cond_weight

        # D(fake_I)
        d_fake_desired_img_prob, _ = self.netD(fake_imgs_masked.detach())
        loss_d_fake = self.compute_loss_D(d_fake_desired_img_prob, False) * self.D_prob_weight

        # combine d_prob_loss
        loss_d_prob = loss_d_real + loss_d_fake

        return loss_d_prob, loss_d_cond, fake_imgs_masked

    def forward_G(self, real_img, real_cond, desired_cond, desired_ordinal_cond):
        # generate fake images
        fake_imgs, fake_img_mask = self.netG(real_img, desired_cond)
        fake_imgs_masked = fake_img_mask * real_img + (1 - fake_img_mask) * fake_imgs

        # D(G(Ic1, c2))
        d_fake_desired_img_prob, d_fake_desired_img_masked_cond = self.netD(fake_imgs_masked)
        loss_g_masked_fake = self.compute_loss_D(d_fake_desired_img_prob, True) * self.D_prob_weight
        if self.is_ordinal_reg:
            loss_g_masked_cond = self.D_cond_loss(d_fake_desired_img_masked_cond, desired_ordinal_cond) * self.D_cond_weight
        else:
            loss_g_masked_cond = self.D_cond_loss(d_fake_desired_img_masked_cond, desired_cond) * self.D_cond_weight

        # loss mask
        loss_g_mask = torch.mean(fake_img_mask) * self.mask_weight
        loss_g_mask_smooth = self.compute_loss_smooth(fake_img_mask) * self.mask_smooth_weight

        # combine losses
        return loss_g_masked_fake, loss_g_masked_cond, loss_g_mask, loss_g_mask_smooth

    # compute Discriminator loss
    def compute_loss_D(self, estim, is_real):
        return -torch.mean(estim) if is_real else torch.mean(estim)

    def compute_loss_smooth(self, mat):
        loss_smooth = torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + torch.sum(
            torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))
        return loss_smooth / mat.size(0)

    def gradient_penalty_D(self, real_img, fake_imgs_masked):
        batch_size = fake_imgs_masked.size(0)
        # interpolate sample
        alpha = torch.rand(batch_size, 1, 1, 1).cuda().expand_as(real_img)
        interpolated = Variable(alpha * real_img.data + (1 - alpha) * fake_imgs_masked.data, requires_grad=True)
        interpolated_prob, _ = self.netD(interpolated)

        # computer gradients
        grad = torch.autograd.grad(outputs=interpolated_prob,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(interpolated_prob.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        # penalize gradients
        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        loss_d_gp = torch.mean((grad_l2norm - 1) ** 2) * self.gradient_penalty_weight
        return loss_d_gp

    def print_info(self, e, type="train"):
        str = 'Epoch[{epoch[0]}/{epoch[1]}] '.format(epoch=((e + 1), self.max_epoch))
        if type == "train":
            str += 'TrainTime:{train_time:7.2f} sec, '.format(train_time=(time.time() - self.cur_time))
        else:
            str += 'TestTime:{train_time:8.2f} sec, '.format(train_time=(time.time() - self.cur_time))
        self.cur_time = time.time()
        loss_str = '[{:7.2f}'.format(np.mean(self.loss_d_prob), end='')
        loss_str += ', {:6.2f}'.format(np.mean(self.loss_d_cond), end='')
        loss_str += ', {:5.2f}'.format(np.mean(self.loss_d_gp), end='')
        loss_str += ', {:7.2f}'.format(np.mean(self.loss_g_masked_fake), end='')
        loss_str += ', {:6.2f}'.format(np.mean(self.loss_g_masked_cond), end='')
        loss_str += ', {:5.2f}'.format(np.mean(self.loss_g_mask), end='')
        loss_str += ', {:5.2f}'.format(np.mean(self.loss_g_mask_smooth), end='')
        loss_str += ']'
        str += 'Loss' + loss_str
        if len(str) > 0:
            print(str)
        # clear loss
        self.loss_d_prob = []
        self.loss_d_cond = []
        self.loss_d_gp = []
        self.loss_g_masked_fake = []
        self.loss_g_masked_cond = []
        self.loss_g_mask = []
        self.loss_g_mask_smooth = []

    # generate images
    def generate_images(self, real_img, n_iter, sample_type='test', index=None):
        st = time.time()
        self.netG.eval()
        bs, ch, w, h = real_img.size()
        # generate fake samples
        num_age = self.age_group
        fake_group_list = torch.arange(num_age).unsqueeze(0).repeat((bs, 1))
        fake_group_list = ops.group_to_one_hot(fake_group_list).cuda().float()
        real_imgs = real_img.unsqueeze(1)
        real_imgs = real_imgs.repeat(1, num_age, 1, 1, 1)
        real_imgs = real_imgs.view(-1, ch, w, h)

        with torch.no_grad():
            fake_imgs, fake_img_mask = self.netG(real_imgs, fake_group_list)
        fake_imgs_masked = fake_img_mask * real_imgs + (1 - fake_img_mask) * fake_imgs

        # add mask image
        fake_img_mask = fake_img_mask.repeat(1, 3, 1, 1)
        fake_images = torch.cat([real_imgs, fake_imgs_masked, fake_img_mask, fake_imgs], dim=0)
        fake_images = torchvision.utils.make_grid(fake_images, nrow=fake_img_mask.size(0), padding=0)
        fake_images = torch.split(fake_images, self.age_group * self.image_size, dim=2)
        fake_images = torch.stack(fake_images)
        fake_images = fake_images * 0.5 + 0.5
        if sample_type == 'test':
            images_save_dir = os.path.join(self.images_save_dir, 'test_{}'.format(n_iter))
            os.makedirs(images_save_dir, exist_ok=True)
        else:
            images_save_dir = self.images_save_dir
        index = '' if index is None else '{}_'.format(index)
        fake_images = torch.split(fake_images, bs // torch.cuda.device_count(), dim=0)
        for i, fake_img in enumerate(fake_images):
            save_image(fake_img, os.path.join(images_save_dir,
                                              index + 'fake_{:s}_image_{:05d}_iter_{}.png'.format(sample_type, n_iter,
                                                                                                  i)),
                       nrow=1)
        # print(time.time()-st)

    # save models
    def save(self, e):
        torch.save(self.netG.state_dict(), osp.join(self.save_model_path, '{:05d}_netG.pth'.format(e + 1)))
        torch.save(self.netD.state_dict(), osp.join(self.save_model_path, '{:05d}_netD.pth'.format(e + 1)))
        torch.save(self.optimizer_G.state_dict(),
                   osp.join(self.save_model_path, '{:05d}_optimizer_G.pth'.format(e + 1)))
        torch.save(self.optimizer_D.state_dict(),
                   osp.join(self.save_model_path + '{:05d}_optimizer_D.pth'.format(e + 1)))

    # load models
    def load(self):
        self.netG.load_state_dict(
            torch.load(osp.join(self.save_model_path, '{:05d}_netG.pth'.format(self.restore_epoch))))
        self.netD.load_state_dict(
            torch.load(osp.join(self.save_model_path, '{:05d}_netD.pth'.format(self.restore_epoch))))
        self.optimizer_G.load_state_dict(
            torch.load(osp.join(self.save_model_path, '{:05d}_optimizer_G.pth'.format(self.restore_epoch))))
        self.optimizer_D.load_state_dict(
            torch.load(osp.join(self.save_model_path, '{:05d}_optimizer_D.pth'.format(self.restore_epoch))))
