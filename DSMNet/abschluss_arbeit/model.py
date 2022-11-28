import my_nets
import torch
import torch.nn as nn


class GenRGB(nn.Module):
    def __init__(self, output_dim_rgb, nz):
        super(GenRGB, self).__init__()
        self.nz = nz
        ini_tch = 256
        tch_add = ini_tch
        tch = ini_tch
        self.tch_add = tch_add

        self.decRGB1 = my_nets.MisINSResBlock(tch, tch_add)
        self.decRGB2 = my_nets.MisINSResBlock(tch, tch_add)
        self.decRGB3 = my_nets.MisINSResBlock(tch, tch_add)
        self.decRGB4 = my_nets.MisINSResBlock(tch, tch_add)

        decRGB5 = []
        decRGB5 += [my_nets.ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        tch = tch//2
        decRGB5 += [my_nets.ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        tch = tch//2
        decRGB5 += [nn.ConvTranspose2d(tch, output_dim_rgb, kernel_size=1, stride=1, padding=0)]
        decRGB5 += [nn.Tanh()]
        self.decRGB5 = nn.Sequential(*decRGB5)

        self.mlpRGB = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, tch_add*4))

    def forward(self, x, z):
        z = self.mlpRGB(z)
        z1, z2, z3, z4 = torch.split(z, self.tch_add, dim=1)
        z1, z2, z3, z4 = z1.contiguous(), z2.contiguous(), z3.contiguous(), z4.contiguous()
        out1 = self.decRGB1(x, z1)
        out2 = self.decRGB2(out1, z2)
        out3 = self.decRGB3(out2, z3)
        out4 = self.decRGB4(out3, z4)
        out = self.decRGB5(out4)
        return out


class GenHSI(nn.Module):
    def __init__(self, output_dim_hsi, nz):
        super(GenHSI, self).__init__()
        self.nz = nz
        ini_tch = 256
        tch_add = ini_tch
        tch = ini_tch
        self.tch_add = tch_add

        self.decHSI1 = my_nets.MisINSResBlock(tch, tch_add)
        self.decHSI2 = my_nets.MisINSResBlock(tch, tch_add)
        self.decHSI3 = my_nets.MisINSResBlock(tch, tch_add)
        self.decHSI4 = my_nets.MisINSResBlock(tch, tch_add)

        decHSI5 = []
        decHSI5 += [my_nets.ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        tch = tch // 2
        decHSI5 += [my_nets.ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        tch = tch // 2
        decHSI5 += [nn.ConvTranspose2d(tch, output_dim_hsi, kernel_size=1, stride=1, padding=0)]
        decHSI5 += [nn.Tanh()]
        self.decHSI5 = nn.Sequential(*decHSI5)

        self.mlpHSI = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, tch_add * 4))

    def forward(self, x, z):
        z = self.mlpHSI(z)
        z1, z2, z3, z4 = torch.split(z, self.tch_add, dim=1)
        z1, z2, z3, z4 = z1.contiguous(), z2.contiguous(), z3.contiguous(), z4.contiguous()
        out1 = self.decHSI1(x, z1)
        out2 = self.decHSI2(out1, z2)
        out3 = self.decHSI3(out2, z3)
        out4 = self.decHSI4(out3, z4)
        out = self.decHSI5(out4)
        return out


class MSMT(nn.Module):
    def __init__(self, device, input_dim_rgb=3, input_dim_hsi=50, nz=8):
        super(MSMT, self).__init__()
        self.device = device
        self.nz= nz
        # encoder_rgb_content
        enc_rgb_c = []
        tch = 64
        enc_rgb_c += [my_nets.LeakyReLUConv2d(input_dim_rgb, tch, kernel_size=7, stride=1, padding=3)]
        for _ in range(2):
            enc_rgb_c += [my_nets.ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
            tch *= 2
        for _ in range(3):
            enc_rgb_c += [my_nets.INSResBlock(tch, tch)]
        # encoder_hsi_content
        enc_hsi_c = []
        tch = 64
        enc_hsi_c += [my_nets.LeakyReLUConv2d(input_dim_hsi, tch, kernel_size=7, stride=1, padding=3)]
        for _ in range(2):
            enc_hsi_c += [my_nets.ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
            tch *= 2
        for _ in range(3):
            enc_hsi_c += [my_nets.INSResBlock(tch, tch)]
        # encoder_share_content
        enc_share = []
        for i in range(0, 1):
            enc_share += [my_nets.INSResBlock(tch, tch)]
            enc_share += [my_nets.GaussianNoiseLayer()]
            self.enc_share = nn.Sequential(*enc_share)

        self.enc_rgb_c = nn.Sequential(*enc_rgb_c)
        self.enc_hsi_c = nn.Sequential(*enc_hsi_c)

        # encoder_rgb_atte
        dim = 64
        self.encoder_rgb_atte = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_dim_rgb, dim, 7, 1),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim*2, 4, 2),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim*2, dim*4, 4, 2),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim*4, dim*4, 4, 2),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim*4, dim*4, 4, 2),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim*4, nz, 1, 1, 0))

        self.encoder_hsi_atte = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_dim_hsi, dim, 7, 1),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim*2, 4, 2),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim*2, dim*4, 4, 2),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim*4, dim*4, 4, 2),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim*4, dim*4, 4, 2),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim*4, nz, 1, 1, 0))

        # generator_rgb
        self.genRGB = GenRGB(input_dim_rgb, nz)
        # generator_hsi
        self.genHSI = GenHSI(input_dim_hsi, nz)

        # downstream task
        # dfc sem----out-channel=20
        self.downstream_head_rgb = nn.Sequential(nn.ConvTranspose2d(tch, tch, 3, 2, 1, 1),
                                             nn.ReLU(inplace=False),
                                             nn.ConvTranspose2d(tch, tch//2, 3, 2, 1, 1),
                                             nn.ReLU(inplace=False),
                                             nn.Conv2d(tch//2, 21, 3, stride=1, padding='same'))
        self.downstream_head_hsi = nn.Sequential(nn.ConvTranspose2d(tch, tch, 3, 2, 1, 1),
                                             nn.ReLU(inplace=False),
                                             nn.ConvTranspose2d(tch, tch//2, 3, 2, 1, 1),
                                             nn.ReLU(inplace=False),
                                             nn.Conv2d(tch//2, 21, 3, stride=1, padding='same'))

    def forward(self, rgb, hsi):
        row_rgb, row_hsi = rgb, hsi

        # get content
        rgb_content = self.enc_rgb_c(rgb)
        hsi_content = self.enc_hsi_c(hsi)
        rgb_after_share_c = self.enc_share(rgb_content)
        hsi_after_share_c = self.enc_share(hsi_content)

        # get attr
        rgb_atte = self.encoder_rgb_atte(rgb)
        rgb_atte = rgb_atte.view(rgb_atte.size(0), -1)
        hsi_atte = self.encoder_hsi_atte(hsi)
        hsi_atte = hsi_atte.view(hsi_atte.size(0), -1)

        # reconstruct
        random_z_rgb = self.get_z_random(rgb_after_share_c.size(0), self.nz)
        gen_rgb = self.genRGB(rgb_after_share_c, random_z_rgb)
        random_z_hsi = self.get_z_random(hsi_after_share_c.size(0), self.nz)
        gen_hsi = self.genHSI(hsi_after_share_c, random_z_hsi)

        # transfer reconstruct
        transfer_gen_rgb = self.genRGB(hsi_after_share_c, rgb_atte)
        transfer_gen_hsi = self.genHSI(rgb_after_share_c, hsi_atte)

        # downstream-task
        down_out_rgb = self.downstream_head_rgb(rgb_after_share_c)
        down_out_hsi = self.downstream_head_hsi(hsi_after_share_c)

        # todo: try content not shared or shared
        return row_rgb, row_hsi, rgb_content, hsi_content, gen_rgb, gen_hsi, transfer_gen_rgb, transfer_gen_hsi, down_out_rgb, down_out_hsi

    def get_z_random(self, batchsize, nz, random_type='gauss'):
        z = torch.randn(batchsize, nz).to(self.device)
        return z



class DRIT(nn.Module):
    def __init__(self, opts):
        super(DRIT, self).__init__()

        # parameters
        lr = 1e-4
        lr_dcontent = lr / 2.5
        self.nz = 8
        self.concat = opts.concat
        self.no_ms = opts.no_ms

        # discriminators
        if opts.dis_scale > 1:
            self.disA = my_nets.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
            self.disB = my_nets.MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
            self.disA2 = my_nets.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
            self.disB2 = my_nets.MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
        else:
            self.disA = my_nets.Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
            self.disB = my_nets.Dis(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
            self.disA2 = my_nets.Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
            self.disB2 = my_nets.Dis(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
        self.disContent = my_nets.Dis_content()

        # encoders
        self.enc_c = my_nets.E_content(opts.input_dim_a, opts.input_dim_b)
        if self.concat:
            self.enc_a = my_nets.E_attr_concat(opts.input_dim_a, opts.input_dim_b, self.nz, norm_layer=None, nl_layer=my_nets.get_non_linearity(layer_type='lrelu'))
        else:
            self.enc_a = my_nets.E_attr(opts.input_dim_a, opts.input_dim_b, self.nz)

        # generateor
        if self.concat:
            self.gen = my_nets.G_concat(opts.input_dim_a, opts.input_dim_b, nz=self.nz)
        else:
            self.gen = my_nets.G(opts.input_dim_a, opts.input_dim_b, nz=self.nz)

        # optimizers
        self.disA_opt = torch.optim.Adam(self.disA.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disB_opt = torch.optim.Adam(self.disB.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disA2_opt = torch.optim.Adam(self.disA2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disB2_opt = torch.optim.Adam(self.disB2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disContent_opt = torch.optim.Adam(self.disContent.parameters(), lr=lr_dcontent, betas=(0.5, 0.999), weight_decay=0.0001)
        self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

        # Setup the loss function for training
        self.criterionL1 = torch.nn.L1Loss()

    def initialize(self):
        self.disA.apply(my_nets.gaussian_weights_init)
        self.disB.apply(my_nets.gaussian_weights_init)
        self.disA2.apply(my_nets.gaussian_weights_init)
        self.disB2.apply(my_nets.gaussian_weights_init)
        self.disContent.apply(my_nets.gaussian_weights_init)
        self.gen.apply(my_nets.gaussian_weights_init)
        self.enc_c.apply(my_nets.gaussian_weights_init)
        self.enc_a.apply(my_nets.gaussian_weights_init)

    def set_scheduler(self, opts, last_ep=0):
        self.disA_sch = my_nets.get_scheduler(self.disA_opt, opts, last_ep)
        self.disB_sch = my_nets.get_scheduler(self.disB_opt, opts, last_ep)
        self.disA2_sch = my_nets.get_scheduler(self.disA2_opt, opts, last_ep)
        self.disB2_sch = my_nets.get_scheduler(self.disB2_opt, opts, last_ep)
        self.disContent_sch = my_nets.get_scheduler(self.disContent_opt, opts, last_ep)
        self.enc_c_sch = my_nets.get_scheduler(self.enc_c_opt, opts, last_ep)
        self.enc_a_sch = my_nets.get_scheduler(self.enc_a_opt, opts, last_ep)
        self.gen_sch = my_nets.get_scheduler(self.gen_opt, opts, last_ep)

    def setgpu(self, gpu):
        self.gpu = gpu
        self.disA.cuda(self.gpu)
        self.disB.cuda(self.gpu)
        self.disA2.cuda(self.gpu)
        self.disB2.cuda(self.gpu)
        self.disContent.cuda(self.gpu)
        self.enc_c.cuda(self.gpu)
        self.enc_a.cuda(self.gpu)
        self.gen.cuda(self.gpu)

    def get_z_random(self, batchsize, nz, random_type='gauss'):
        z = torch.randn(batchsize, nz).cuda(self.gpu)
        return z

    def test_forward(self, image, a2b=True):
        self.z_random = self.get_z_random(image.size(0), self.nz, 'gauss')
        if a2b:
            self.z_content = self.enc_c.forward_a(image)
            output = self.gen.forward_b(self.z_content, self.z_random)
        else:
            self.z_content = self.enc_c.forward_b(image)
            output = self.gen.forward_a(self.z_content, self.z_random)
        return output

    def test_forward_transfer(self, image_a, image_b, a2b=True):
        self.z_content_a, self.z_content_b = self.enc_c.forward(image_a, image_b)
        if self.concat:
            self.mu_a, self.logvar_a, self.mu_b, self.logvar_b = self.enc_a.forward(image_a, image_b)
            std_a = self.logvar_a.mul(0.5).exp_()
            eps = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
            self.z_attr_a = eps.mul(std_a).add_(self.mu_a)
            std_b = self.logvar_b.mul(0.5).exp_()
            eps = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
            self.z_attr_b = eps.mul(std_b).add_(self.mu_b)
        else:
            self.z_attr_a, self.z_attr_b = self.enc_a.forward(image_a, image_b)
        if a2b:
            output = self.gen.forward_b(self.z_content_a, self.z_attr_b)
        else:
            output = self.gen.forward_a(self.z_content_b, self.z_attr_a)
        return output

    def forward(self):
        # input images
        half_size = 1
        real_A = self.input_A
        real_B = self.input_B
        self.real_A_encoded = real_A[0:half_size]
        self.real_A_random = real_A[half_size:]
        self.real_B_encoded = real_B[0:half_size]
        self.real_B_random = real_B[half_size:]

        # get encoded z_c
        self.z_content_a, self.z_content_b = self.enc_c.forward(self.real_A_encoded, self.real_B_encoded)

        # get encoded z_a
        if self.concat:
            self.mu_a, self.logvar_a, self.mu_b, self.logvar_b = self.enc_a.forward(self.real_A_encoded, self.real_B_encoded)
            std_a = self.logvar_a.mul(0.5).exp_()
            eps_a = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
            self.z_attr_a = eps_a.mul(std_a).add_(self.mu_a)
            std_b = self.logvar_b.mul(0.5).exp_()
            eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
            self.z_attr_b = eps_b.mul(std_b).add_(self.mu_b)
        else:
            self.z_attr_a, self.z_attr_b = self.enc_a.forward(self.real_A_encoded, self.real_B_encoded)

        # get random z_a
        self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.nz, 'gauss')
        if not self.no_ms:
            self.z_random2 = self.get_z_random(self.real_A_encoded.size(0), self.nz, 'gauss')

        # first cross translation
        if not self.no_ms:
            input_content_forA = torch.cat((self.z_content_b, self.z_content_a, self.z_content_b, self.z_content_b), 0)
            input_content_forB = torch.cat((self.z_content_a, self.z_content_b, self.z_content_a, self.z_content_a), 0)
            input_attr_forA = torch.cat((self.z_attr_a, self.z_attr_a, self.z_random, self.z_random2), 0)
            input_attr_forB = torch.cat((self.z_attr_b, self.z_attr_b, self.z_random, self.z_random2), 0)
            output_fakeA = self.gen.forward_a(input_content_forA, input_attr_forA)
            output_fakeB = self.gen.forward_b(input_content_forB, input_attr_forB)
            self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random, self.fake_A_random2 = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
            self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random, self.fake_B_random2 = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)
        else:
            input_content_forA = torch.cat((self.z_content_b, self.z_content_a, self.z_content_b), 0)
            input_content_forB = torch.cat((self.z_content_a, self.z_content_b, self.z_content_a), 0)
            input_attr_forA = torch.cat((self.z_attr_a, self.z_attr_a, self.z_random), 0)
            input_attr_forB = torch.cat((self.z_attr_b, self.z_attr_b, self.z_random), 0)
            output_fakeA = self.gen.forward_a(input_content_forA, input_attr_forA)
            output_fakeB = self.gen.forward_b(input_content_forB, input_attr_forB)
            self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
            self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)

        # get reconstructed encoded z_c
        self.z_content_recon_b, self.z_content_recon_a = self.enc_c.forward(self.fake_A_encoded, self.fake_B_encoded)

        # get reconstructed encoded z_a
        if self.concat:
            self.mu_recon_a, self.logvar_recon_a, self.mu_recon_b, self.logvar_recon_b = self.enc_a.forward(self.fake_A_encoded, self.fake_B_encoded)
            std_a = self.logvar_recon_a.mul(0.5).exp_()
            eps_a = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
            self.z_attr_recon_a = eps_a.mul(std_a).add_(self.mu_recon_a)
            std_b = self.logvar_recon_b.mul(0.5).exp_()
            eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
            self.z_attr_recon_b = eps_b.mul(std_b).add_(self.mu_recon_b)
        else:
            self.z_attr_recon_a, self.z_attr_recon_b = self.enc_a.forward(self.fake_A_encoded, self.fake_B_encoded)

        # second cross translation
        self.fake_A_recon = self.gen.forward_a(self.z_content_recon_a, self.z_attr_recon_a)
        self.fake_B_recon = self.gen.forward_b(self.z_content_recon_b, self.z_attr_recon_b)

        # for display
        self.image_display = torch.cat((self.real_A_encoded[0:1].detach().cpu(), self.fake_B_encoded[0:1].detach().cpu(),
                                        self.fake_B_random[0:1].detach().cpu(), self.fake_AA_encoded[0:1].detach().cpu(),
                                        self.fake_A_recon[0:1].detach().cpu(), self.real_B_encoded[0:1].detach().cpu(),
                                        self.fake_A_encoded[0:1].detach().cpu(), self.fake_A_random[0:1].detach().cpu(),
                                        self.fake_BB_encoded[0:1].detach().cpu(), self.fake_B_recon[0:1].detach().cpu()), dim=0)

        # for latent regression
        if self.concat:
            self.mu2_a, _, self.mu2_b, _ = self.enc_a.forward(self.fake_A_random, self.fake_B_random)
        else:
            self.z_attr_random_a, self.z_attr_random_b = self.enc_a.forward(self.fake_A_random, self.fake_B_random)

    def forward_content(self):
        half_size = 1
        self.real_A_encoded = self.input_A[0:half_size]
        self.real_B_encoded = self.input_B[0:half_size]
        # get encoded z_c
        self.z_content_a, self.z_content_b = self.enc_c.forward(self.real_A_encoded, self.real_B_encoded)

    def update_D_content(self, image_a, image_b):
        self.input_A = image_a
        self.input_B = image_b
        self.forward_content()
        self.disContent_opt.zero_grad()
        loss_D_Content = self.backward_contentD(self.z_content_a, self.z_content_b)
        self.disContent_loss = loss_D_Content.item()
        nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
        self.disContent_opt.step()

    def update_D(self, image_a, image_b):
        self.input_A = image_a
        self.input_B = image_b
        self.forward()

        # update disA
        self.disA_opt.zero_grad()
        loss_D1_A = self.backward_D(self.disA, self.real_A_encoded, self.fake_A_encoded)
        self.disA_loss = loss_D1_A.item()
        self.disA_opt.step()

        # update disA2
        self.disA2_opt.zero_grad()
        loss_D2_A = self.backward_D(self.disA2, self.real_A_random, self.fake_A_random)
        self.disA2_loss = loss_D2_A.item()
        if not self.no_ms:
            loss_D2_A2 = self.backward_D(self.disA2, self.real_A_random, self.fake_A_random2)
            self.disA2_loss += loss_D2_A2.item()
        self.disA2_opt.step()

        # update disB
        self.disB_opt.zero_grad()
        loss_D1_B = self.backward_D(self.disB, self.real_B_encoded, self.fake_B_encoded)
        self.disB_loss = loss_D1_B.item()
        self.disB_opt.step()

        # update disB2
        self.disB2_opt.zero_grad()
        loss_D2_B = self.backward_D(self.disB2, self.real_B_random, self.fake_B_random)
        self.disB2_loss = loss_D2_B.item()
        if not self.no_ms:
            loss_D2_B2 = self.backward_D(self.disB2, self.real_B_random, self.fake_B_random2)
            self.disB2_loss += loss_D2_B2.item()
        self.disB2_opt.step()

        # update disContent
        self.disContent_opt.zero_grad()
        loss_D_Content = self.backward_contentD(self.z_content_a, self.z_content_b)
        self.disContent_loss = loss_D_Content.item()
        nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
        self.disContent_opt.step()

    def backward_D(self, netD, real, fake):
        pred_fake = netD.forward(fake.detach())
        pred_real = netD.forward(real)
        loss_D = 0
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = nn.functional.sigmoid(out_a)
            out_real = nn.functional.sigmoid(out_b)
            all0 = torch.zeros_like(out_fake).cuda(self.gpu)
            all1 = torch.ones_like(out_real).cuda(self.gpu)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            loss_D += ad_true_loss + ad_fake_loss
        loss_D.backward()
        return loss_D

    def backward_contentD(self, imageA, imageB):
        pred_fake = self.disContent.forward(imageA.detach())
        pred_real = self.disContent.forward(imageB.detach())
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = nn.functional.sigmoid(out_a)
            out_real = nn.functional.sigmoid(out_b)
            all1 = torch.ones((out_real.size(0))).cuda(self.gpu)
            all0 = torch.zeros((out_fake.size(0))).cuda(self.gpu)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
        loss_D = ad_true_loss + ad_fake_loss
        loss_D.backward()
        return loss_D

    def update_EG(self):
        # # update G, Ec, Ea
        # self.enc_c_opt.zero_grad()
        # self.enc_a_opt.zero_grad()
        # self.gen_opt.zero_grad()
        # self.backward_EG()
        # self.enc_c_opt.step()
        # self.enc_a_opt.step()
        # self.gen_opt.step()
        #
        # # update G, Ec
        # self.enc_c_opt.zero_grad()
        # self.gen_opt.zero_grad()
        # self.backward_G_alone()
        # self.enc_c_opt.step()
        # self.gen_opt.step()

        # todo figure out the diff(question in github)
        # update G, Ec, Ea
        self.enc_c_opt.zero_grad()
        self.enc_a_opt.zero_grad()
        self.gen_opt.zero_grad()
        self.backward_EG()
        self.enc_c_opt.step()
        self.enc_a_opt.step()
        self.gen_opt.step()
        # update G, Ec
        self.enc_c_opt.zero_grad()
        self.gen_opt.zero_grad()
        self.forward()  # call forward() to using new network parameters to compute variables
        self.backward_G_alone()
        self.enc_c_opt.step()
        self.gen_opt.step()

    def backward_EG(self):
        # content Ladv for generator
        loss_G_GAN_Acontent = self.backward_G_GAN_content(self.z_content_a)
        loss_G_GAN_Bcontent = self.backward_G_GAN_content(self.z_content_b)

        # Ladv for generator
        loss_G_GAN_A = self.backward_G_GAN(self.fake_A_encoded, self.disA)
        loss_G_GAN_B = self.backward_G_GAN(self.fake_B_encoded, self.disB)

        # KL loss - z_a
        if self.concat:
            kl_element_a = self.mu_a.pow(2).add_(self.logvar_a.exp()).mul_(-1).add_(1).add_(self.logvar_a)
            loss_kl_za_a = torch.sum(kl_element_a).mul_(-0.5) * 0.01
            kl_element_b = self.mu_b.pow(2).add_(self.logvar_b.exp()).mul_(-1).add_(1).add_(self.logvar_b)
            loss_kl_za_b = torch.sum(kl_element_b).mul_(-0.5) * 0.01
        else:
            loss_kl_za_a = self._l2_regularize(self.z_attr_a) * 0.01
            loss_kl_za_b = self._l2_regularize(self.z_attr_b) * 0.01

        # KL loss - z_c
        loss_kl_zc_a = self._l2_regularize(self.z_content_a) * 0.01
        loss_kl_zc_b = self._l2_regularize(self.z_content_b) * 0.01

        # cross cycle consistency loss
        loss_G_L1_A = self.criterionL1(self.fake_A_recon, self.real_A_encoded) * 10
        loss_G_L1_B = self.criterionL1(self.fake_B_recon, self.real_B_encoded) * 10
        loss_G_L1_AA = self.criterionL1(self.fake_AA_encoded, self.real_A_encoded) * 10
        loss_G_L1_BB = self.criterionL1(self.fake_BB_encoded, self.real_B_encoded) * 10

        loss_G = loss_G_GAN_A + loss_G_GAN_B + \
                 loss_G_GAN_Acontent + loss_G_GAN_Bcontent + \
                 loss_G_L1_AA + loss_G_L1_BB + \
                 loss_G_L1_A + loss_G_L1_B + \
                 loss_kl_zc_a + loss_kl_zc_b + \
                 loss_kl_za_a + loss_kl_za_b

        loss_G.backward(retain_graph=True)

        self.gan_loss_a = loss_G_GAN_A.item()
        self.gan_loss_b = loss_G_GAN_B.item()
        self.gan_loss_acontent = loss_G_GAN_Acontent.item()
        self.gan_loss_bcontent = loss_G_GAN_Bcontent.item()
        self.kl_loss_za_a = loss_kl_za_a.item()
        self.kl_loss_za_b = loss_kl_za_b.item()
        self.kl_loss_zc_a = loss_kl_zc_a.item()
        self.kl_loss_zc_b = loss_kl_zc_b.item()
        self.l1_recon_A_loss = loss_G_L1_A.item()
        self.l1_recon_B_loss = loss_G_L1_B.item()
        self.l1_recon_AA_loss = loss_G_L1_AA.item()
        self.l1_recon_BB_loss = loss_G_L1_BB.item()
        self.G_loss = loss_G.item()

    def backward_G_GAN_content(self, data):
        outs = self.disContent.forward(data)
        for out in outs:
            outputs_fake = nn.functional.sigmoid(out)
            all_half = 0.5 * torch.ones((outputs_fake.size(0))).cuda(self.gpu)
            ad_loss = nn.functional.binary_cross_entropy(outputs_fake, all_half)
        return ad_loss

    def backward_G_GAN(self, fake, netD=None):
        outs_fake = netD.forward(fake)
        loss_G = 0
        for out_a in outs_fake:
            outputs_fake = nn.functional.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
            loss_G += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
        return loss_G

    def backward_G_alone(self):
        # Ladv for generator
        loss_G_GAN2_A = self.backward_G_GAN(self.fake_A_random, self.disA2)
        loss_G_GAN2_B = self.backward_G_GAN(self.fake_B_random, self.disB2)
        if not self.no_ms:
            loss_G_GAN2_A2 = self.backward_G_GAN(self.fake_A_random2, self.disA2)
            loss_G_GAN2_B2 = self.backward_G_GAN(self.fake_B_random2, self.disB2)

        # mode seeking loss for A-->B and B-->A
        if not self.no_ms:
            lz_AB = torch.mean(torch.abs(self.fake_B_random2 - self.fake_B_random)) / torch.mean(torch.abs(self.z_random2 - self.z_random))
            lz_BA = torch.mean(torch.abs(self.fake_A_random2 - self.fake_A_random)) / torch.mean(torch.abs(self.z_random2 - self.z_random))
            eps = 1 * 1e-5
            loss_lz_AB = 1 / (lz_AB + eps)
            loss_lz_BA = 1 / (lz_BA + eps)
        # latent regression loss
        if self.concat:
            loss_z_L1_a = torch.mean(torch.abs(self.mu2_a - self.z_random)) * 10
            loss_z_L1_b = torch.mean(torch.abs(self.mu2_b - self.z_random)) * 10
        else:
            loss_z_L1_a = torch.mean(torch.abs(self.z_attr_random_a - self.z_random)) * 10
            loss_z_L1_b = torch.mean(torch.abs(self.z_attr_random_b - self.z_random)) * 10

        loss_z_L1 = loss_z_L1_a + loss_z_L1_b + loss_G_GAN2_A + loss_G_GAN2_B
        if not self.no_ms:
            loss_z_L1 += (loss_G_GAN2_A2 + loss_G_GAN2_B2)
            loss_z_L1 += (loss_lz_AB + loss_lz_BA)
        loss_z_L1.backward()
        self.l1_recon_z_loss_a = loss_z_L1_a.item()
        self.l1_recon_z_loss_b = loss_z_L1_b.item()
        if not self.no_ms:
            self.gan2_loss_a = loss_G_GAN2_A.item() + loss_G_GAN2_A2.item()
            self.gan2_loss_b = loss_G_GAN2_B.item() + loss_G_GAN2_B2.item()
            self.lz_AB = loss_lz_AB.item()
            self.lz_BA = loss_lz_BA.item()
        else:
            self.gan2_loss_a = loss_G_GAN2_A.item()
            self.gan2_loss_b = loss_G_GAN2_B.item()

    def update_lr(self):
        self.disA_sch.step()
        self.disB_sch.step()
        self.disA2_sch.step()
        self.disB2_sch.step()
        self.disContent_sch.step()
        self.enc_c_sch.step()
        self.enc_a_sch.step()
        self.gen_sch.step()

    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir)
        # weight
        if train:
            self.disA.load_state_dict(checkpoint['disA'])
            self.disA2.load_state_dict(checkpoint['disA2'])
            self.disB.load_state_dict(checkpoint['disB'])
            self.disB2.load_state_dict(checkpoint['disB2'])
            self.disContent.load_state_dict(checkpoint['disContent'])
        self.enc_c.load_state_dict(checkpoint['enc_c'])
        self.enc_a.load_state_dict(checkpoint['enc_a'])
        self.gen.load_state_dict(checkpoint['gen'])
        # optimizer
        if train:
            self.disA_opt.load_state_dict(checkpoint['disA_opt'])
            self.disA2_opt.load_state_dict(checkpoint['disA2_opt'])
            self.disB_opt.load_state_dict(checkpoint['disB_opt'])
            self.disB2_opt.load_state_dict(checkpoint['disB2_opt'])
            self.disContent_opt.load_state_dict(checkpoint['disContent_opt'])
            self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
            self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
            self.gen_opt.load_state_dict(checkpoint['gen_opt'])
        return checkpoint['ep'], checkpoint['total_it']

    def save(self, filename, ep, total_it):
        state = {
                'disA': self.disA.state_dict(),
                'disA2': self.disA2.state_dict(),
                'disB': self.disB.state_dict(),
                'disB2': self.disB2.state_dict(),
                'disContent': self.disContent.state_dict(),
                'enc_c': self.enc_c.state_dict(),
                'enc_a': self.enc_a.state_dict(),
                'gen': self.gen.state_dict(),
                'disA_opt': self.disA_opt.state_dict(),
                'disA2_opt': self.disA2_opt.state_dict(),
                'disB_opt': self.disB_opt.state_dict(),
                'disB2_opt': self.disB2_opt.state_dict(),
                'disContent_opt': self.disContent_opt.state_dict(),
                'enc_c_opt': self.enc_c_opt.state_dict(),
                'enc_a_opt': self.enc_a_opt.state_dict(),
                'gen_opt': self.gen_opt.state_dict(),
                'ep': ep,
                'total_it': total_it
                }
        torch.save(state, filename)
        return

    def assemble_outputs(self):
        images_a = self.normalize_image(self.real_A_encoded).detach()
        images_b = self.normalize_image(self.real_B_encoded).detach()
        images_a1 = self.normalize_image(self.fake_A_encoded).detach()
        images_a2 = self.normalize_image(self.fake_A_random).detach()
        images_a3 = self.normalize_image(self.fake_A_recon).detach()
        images_a4 = self.normalize_image(self.fake_AA_encoded).detach()
        images_b1 = self.normalize_image(self.fake_B_encoded).detach()
        images_b2 = self.normalize_image(self.fake_B_random).detach()
        images_b3 = self.normalize_image(self.fake_B_recon).detach()
        images_b4 = self.normalize_image(self.fake_BB_encoded).detach()
        row1 = torch.cat((images_a[0:1, ::], images_b1[0:1, ::], images_b2[0:1, ::], images_a4[0:1, ::], images_a3[0:1, ::]), 3)
        row2 = torch.cat((images_b[0:1, ::], images_a1[0:1, ::], images_a2[0:1, ::], images_b4[0:1, ::], images_b3[0:1, ::]), 3)
        return torch.cat((row1, row2), 2)

    def normalize_image(self, x):
        return x[:, 0:3, :, :]
