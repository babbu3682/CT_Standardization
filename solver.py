from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, simens_loader, ge_loader, philips_loader, toshiba_loader, valid_simens_loader, valid_ge_loader, config):
        """Initialize configurations."""
        # Multi gpu
        self.multi_gpu_mode = config.multi_gpu_mode

        # Data loader.
        self.simens_loader  = simens_loader
        self.ge_loader      = ge_loader
        self.philips_loader = philips_loader
        self.toshiba_loader = toshiba_loader

        self.valid_simens_loader = valid_simens_loader
        self.valid_ge_loader     = valid_ge_loader                

        # Model configurations.
        self.c1_dim       = config.c1_dim
        self.c2_dim       = config.c2_dim 
        self.c3_dim       = config.c3_dim 
        self.c4_dim       = config.c4_dim 

        self.image_size   = config.image_size
        self.g_conv_dim   = config.g_conv_dim
        self.d_conv_dim   = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls   = config.lambda_cls
        self.lambda_rec   = config.lambda_rec
        self.lambda_gp    = config.lambda_gp

        # Training configurations.
        # self.dataset         = config.dataset
        self.batch_size      = config.batch_size
        self.num_iters       = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr            = config.g_lr
        self.d_lr            = config.d_lr
        self.n_critic        = config.n_critic
        self.beta1           = config.beta1
        self.beta2           = config.beta2
        self.resume_iters    = config.resume_iters
        # self.selected_attrs  = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir        = config.log_dir
        self.sample_dir     = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir     = config.result_dir

        # Step size.
        self.log_step        = config.log_step
        self.sample_step     = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step  = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.c1_dim+self.c2_dim+self.c3_dim+self.c4_dim+4, self.g_repeat_num)   # 4 for mask vector.
        self.D = Discriminator(image_size=self.image_size, conv_dim=self.d_conv_dim, c_dim=self.c1_dim+self.c2_dim+self.c3_dim+self.c4_dim, repeat_num=self.d_repeat_num)

        #### Multi-GPU
        if self.multi_gpu_mode == 'DataParallel':
            print("Multi GPU model = DataParallel")
            self.G = torch.nn.DataParallel(self.G)
            self.D = torch.nn.DataParallel(self.D)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))

        #### Multi-GPU
        if self.multi_gpu_mode == 'DataParallel':
            print("Multi GPU model = DataParallel")
            self.G.module.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
            self.D.module.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))            
        else:
            self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
            self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        # labels is integer label [1,2,1,5] : batch:4, class number:6
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=3):
        """Generate target domain labels for debugging and testing."""
        # c_trg_list에 OneHot 방식의 형태로 인가됨, batch 4에 첫 loop의 경우 [[1,0,0], [1,0,0], [1,0,0], [1,0,0]] 로 인가가됨.
        # 우리는 애초에 모든 데이터셋이 onehot 이기에 simple 하게 가능.
        # list에는 batch size 만큼의 c_dim=3 일때, [1, 0, 0], [0, 1, 0], [0, 0, 1] 이 존재.
        c_trg_list = []
        for i in range(c_dim):
            c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)   
            c_trg_list.append(c_trg.to(self.device))

        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)
        else :
            return F.cross_entropy(logit, target)            

    def train_multi(self):
        """Train StarGAN with multiple datasets."""        
        # Data iterators.
        simens_iter  = iter(self.simens_loader)
        ge_iter      = iter(self.ge_loader)
        philips_iter = iter(self.philips_loader)
        toshiba_iter = iter(self.toshiba_loader)

        valid_simens_iter  = iter(self.valid_simens_loader)

        # Fetch fixed inputs for debugging.
        data_dict    = next(valid_simens_iter)
        x_fixed      = data_dict['image'].squeeze(4)
        c_org        = data_dict['label']
        # x_fixed      = torch.cat([ data_dict[i]['image'].squeeze(4)  for i in range(4) ]).to(self.device)  # 8 is patch_nums
        # c_org        = torch.cat([ data_dict[i]['label'] for i in range(4) ]).to(self.device)  # (8*batch, C(=1), 64, 64) or (8*batch, C(=1), D(=3), H(=64), W(=64))

        print("C1 === ", x_fixed.shape)
        print("C2 === ", c_org.shape)

        x_fixed         = x_fixed.to(self.device)
        c_simens_list   = self.create_labels(c_org, self.c1_dim)
        c_ge_list       = self.create_labels(c_org, self.c2_dim)
        c_philips_list  = self.create_labels(c_org, self.c3_dim)
        c_toshiba_list  = self.create_labels(c_org, self.c4_dim)
        
        zero_f_simens    = torch.zeros(x_fixed.size(0), self.c1_dim).to(self.device)             # Zero vector for CelebA.
        zero_f_ge        = torch.zeros(x_fixed.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
        zero_f_philips   = torch.zeros(x_fixed.size(0), self.c3_dim).to(self.device)             # Zero vector for RaFD.
        zero_f_toshiba   = torch.zeros(x_fixed.size(0), self.c4_dim).to(self.device)             # Zero vector for RaFD.
        
        mask_f_simens    = self.label2onehot(torch.ones(x_fixed.size(0))*0, dim=4).to(self.device)    # Mask vector: [1, 0].
        mask_f_ge        = self.label2onehot(torch.ones(x_fixed.size(0))*1, dim=4).to(self.device)     # Mask vector: [0, 1].
        mask_f_philips   = self.label2onehot(torch.ones(x_fixed.size(0))*2, dim=4).to(self.device)     # Mask vector: [0, 1].
        mask_f_toshiba   = self.label2onehot(torch.ones(x_fixed.size(0))*3, dim=4).to(self.device)     # Mask vector: [0, 1].

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            # for dataset in ['Simens', 'GE', 'Philips', 'TOSHIBA']:
            for dataset in ['Simens', 'GE']:                

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                
                # Fetch real images and labels.
                if dataset == 'Simens':
                    data_iter = simens_iter
                elif dataset == 'GE':
                    data_iter = ge_iter
                elif dataset == 'Philips':
                    data_iter = philips_iter
                elif dataset == 'TOSHIBA':
                    data_iter = toshiba_iter
                else :
                    print("Error...!")


                try:
                    data_dict = next(data_iter)
                    # x_real    = data_dict['image'].squeeze(4)
                    # label_org = data_dict['label']
                    x_real    = torch.cat([ data_dict[i]['image'].squeeze(4) for i in range(4) ]).to(self.device)  # 8 is patch_nums
                    label_org = torch.cat([ data_dict[i]['label'] for i in range(4) ]).to(self.device)  # 8 is patch_nums

                except:
                    if dataset == 'Simens':
                        simens_iter = iter(self.simens_loader)
                        data_dict   = next(simens_iter)
                        # x_real      = data_dict['image'].squeeze(4)
                        # label_org   = data_dict['label']           
                        x_real    = torch.cat([ data_dict[i]['image'].squeeze(4) for i in range(4) ]).to(self.device)  # 8 is patch_nums
                        label_org = torch.cat([ data_dict[i]['label'] for i in range(4) ]).to(self.device)  # 8 is patch_nums                                     

                    elif dataset == 'GE':
                        ge_iter    = iter(self.ge_loader)
                        data_dict  = next(ge_iter)
                        # x_real     = data_dict['image'].squeeze(4)
                        # label_org  = data_dict['label']    
                        x_real    = torch.cat([ data_dict[i]['image'].squeeze(4) for i in range(4) ]).to(self.device)  # 8 is patch_nums
                        label_org = torch.cat([ data_dict[i]['label'] for i in range(4) ]).to(self.device)  # 8 is patch_nums                            

                    elif dataset == 'Philips':
                        philips_iter = iter(self.philips_loader)
                        data_dict    = next(philips_iter)
                        # x_real      = data_dict['image'].squeeze(4)
                        # label_org   = data_dict['label']    
                        x_real    = torch.cat([ data_dict[i]['image'].squeeze(4) for i in range(4) ]).to(self.device)  # 8 is patch_nums
                        label_org = torch.cat([ data_dict[i]['label'] for i in range(4) ]).to(self.device)  # 8 is patch_nums    

                    elif dataset == 'TOSHIBA':
                        toshiba_iter = iter(self.toshiba_loader)
                        data_dict    = next(toshiba_iter)
                        # x_real      = data_dict['image'].squeeze(4)
                        # label_org   = data_dict['label']   
                        x_real    = torch.cat([ data_dict[i]['image'].squeeze(4) for i in range(4) ]).to(self.device)  # 8 is patch_nums
                        label_org = torch.cat([ data_dict[i]['label'] for i in range(4) ]).to(self.device)  # 8 is patch_nums                            

                # Generate target domain labels randomly.
                rand_idx  = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                if dataset == 'Simens':
                    c_org = self.label2onehot(label_org, self.c1_dim)
                    c_trg = self.label2onehot(label_trg, self.c1_dim)
                    
                    # zero_simens  = torch.zeros(x_real.size(0), self.c1_dim)
                    zero_ge      = torch.zeros(x_real.size(0), self.c2_dim)
                    zero_philips = torch.zeros(x_real.size(0), self.c3_dim)
                    zero_toshiba = torch.zeros(x_real.size(0), self.c4_dim)

                    mask_simens = self.label2onehot(torch.ones(x_real.size(0))*0, dim=4)
                    # mask_ge       = self.label2onehot(torch.ones(x_real.size(0))*1, dim=4)
                    # mask_philips  = self.label2onehot(torch.ones(x_real.size(0))*2, dim=4)
                    # mask_toshiba  = self.label2onehot(torch.ones(x_real.size(0))*3, dim=4)

                    c_org = torch.cat([c_org, zero_ge, zero_philips, zero_toshiba, mask_simens], dim=1)
                    c_trg = torch.cat([c_trg, zero_ge, zero_philips, zero_toshiba, mask_simens], dim=1)

                elif dataset == 'GE':
                    c_org = self.label2onehot(label_org, self.c2_dim)
                    c_trg = self.label2onehot(label_trg, self.c2_dim)

                    zero_simens  = torch.zeros(x_real.size(0), self.c1_dim)
                    # zero_ge      = torch.zeros(x_real.size(0), self.c2_dim)
                    zero_philips = torch.zeros(x_real.size(0), self.c3_dim)
                    zero_toshiba = torch.zeros(x_real.size(0), self.c4_dim)

                    # mask_simens = self.label2onehot(torch.ones(x_real.size(0))*0, dim=4)
                    mask_ge       = self.label2onehot(torch.ones(x_real.size(0))*1, dim=4)
                    # mask_philips  = self.label2onehot(torch.ones(x_real.size(0))*2, dim=4)
                    # mask_toshiba  = self.label2onehot(torch.ones(x_real.size(0))*3, dim=4)

                    c_org = torch.cat([zero_simens, c_org, zero_philips, zero_toshiba, mask_ge], dim=1)
                    c_trg = torch.cat([zero_simens, c_trg, zero_philips, zero_toshiba, mask_ge], dim=1)

                elif dataset == 'Philips':
                    c_org = self.label2onehot(label_org, self.c3_dim)
                    c_trg = self.label2onehot(label_trg, self.c3_dim)

                    zero_simens  = torch.zeros(x_real.size(0), self.c1_dim)
                    zero_ge      = torch.zeros(x_real.size(0), self.c2_dim)
                    # zero_philips = torch.zeros(x_real.size(0), self.c3_dim)
                    zero_toshiba = torch.zeros(x_real.size(0), self.c4_dim)

                    # mask_simens = self.label2onehot(torch.ones(x_real.size(0))*0, dim=4)
                    # mask_ge       = self.label2onehot(torch.ones(x_real.size(0))*1, dim=4)
                    mask_philips  = self.label2onehot(torch.ones(x_real.size(0))*2, dim=4)
                    # mask_toshiba  = self.label2onehot(torch.ones(x_real.size(0))*3, dim=4)

                    c_org = torch.cat([zero_simens, zero_ge, c_org, zero_toshiba, mask_philips], dim=1)
                    c_trg = torch.cat([zero_simens, zero_ge, c_trg, zero_toshiba, mask_philips], dim=1)

                elif dataset == 'TOSHIBA':
                    c_org = self.label2onehot(label_org, self.c4_dim)
                    c_trg = self.label2onehot(label_trg, self.c4_dim)

                    zero_simens  = torch.zeros(x_real.size(0), self.c1_dim)
                    zero_ge      = torch.zeros(x_real.size(0), self.c2_dim)
                    zero_philips = torch.zeros(x_real.size(0), self.c3_dim)
                    # zero_toshiba = torch.zeros(x_real.size(0), self.c4_dim)

                    # mask_simens  = self.label2onehot(torch.ones(x_real.size(0))*0, dim=4)
                    # mask_ge      = self.label2onehot(torch.ones(x_real.size(0))*1, dim=4)
                    # mask_philips = self.label2onehot(torch.ones(x_real.size(0))*2, dim=4)
                    mask_toshiba   = self.label2onehot(torch.ones(x_real.size(0))*3, dim=4)

                    c_org = torch.cat([zero_simens, zero_ge, zero_philips, c_org, mask_toshiba], dim=1)
                    c_trg = torch.cat([zero_simens, zero_ge, zero_philips, c_trg, mask_toshiba], dim=1)


                x_real    = x_real.to(self.device)             # Input images.
                c_org     = c_org.to(self.device)               # Original domain labels.
                c_trg     = c_trg.to(self.device)               # Target domain labels.
                label_org = label_org.to(self.device)       # Labels for computing classification loss.
                label_trg = label_trg.to(self.device)       # Labels for computing classification loss.

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                out_src, out_cls = self.D(x_real)

                if dataset == 'Simens':
                    out_cls = out_cls[:, :self.c1_dim]
                elif dataset == 'GE':
                    out_cls = out_cls[:, self.c1_dim:self.c1_dim+self.c2_dim]
                elif dataset == 'Philips':
                    out_cls = out_cls[:, self.c1_dim+self.c2_dim:self.c1_dim+self.c2_dim+self.c3_dim]
                elif dataset == 'TOSHIBA':
                    out_cls = out_cls[:, self.c1_dim+self.c2_dim+self.c3_dim:self.c1_dim+self.c2_dim+self.c3_dim+self.c4_dim]
                else :
                    print("Error2...!")

                d_loss_real = -torch.mean(out_src)
                
                # print("DATA SET == ", dataset)
                # print("check1...!", out_cls.shape)   # input
                # print("check2...!", label_org.shape) # target
                # print("check3...!", label_org) # target
                

                d_loss_cls  = self.classification_loss(out_cls, label_org, dataset)

                # Compute loss with fake images.
                x_fake      = self.G(x_real, c_trg)
                out_src, _  = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha      = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)  # (B, C, H, W)
                x_hat      = (alpha*x_real.data + (1-alpha)*x_fake.data).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp  = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls*d_loss_cls + self.lambda_gp*d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls']  = d_loss_cls.item()
                loss['D/loss_gp']   = d_loss_gp.item()
            
                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                if (i+1) % self.n_critic == 0:
                    # Original-to-target domain.
                    x_fake           = self.G(x_real, c_trg)
                    out_src, out_cls = self.D(x_fake)

                    if dataset == 'Simens':
                        out_cls = out_cls[:, :self.c1_dim]
                    elif dataset == 'GE':
                        out_cls = out_cls[:, self.c1_dim:self.c1_dim+self.c2_dim]
                    elif dataset == 'Philips':
                        out_cls = out_cls[:, self.c1_dim+self.c2_dim:self.c1_dim+self.c2_dim+self.c3_dim]
                    elif dataset == 'TOSHIBA':
                        out_cls = out_cls[:, self.c1_dim+self.c2_dim+self.c3_dim:self.c1_dim+self.c2_dim+self.c3_dim+self.c4_dim]
                    else :
                        print("Error3...!")

                    # print("G] DATA SET == ", dataset)
                    # print("G]check1...!", out_cls.shape)   # input
                    # print("G]check2...!", label_trg.shape) # target
                    # print("G]check3...!", label_trg) # target

                    g_loss_fake = -torch.mean(out_src)
                    g_loss_cls  = self.classification_loss(out_cls, label_trg, dataset)



                    # Target-to-original domain.
                    x_reconst  = self.G(x_fake, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec*g_loss_rec + self.lambda_cls*g_loss_cls

                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls'] = g_loss_cls.item()

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                # Print out training info.
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}], Dataset [{}]".format(et, i+1, self.num_iters, dataset)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    # for num, loader in enumerate([self.simens_loader, self.ge_loader, self.philips_loader, self.toshiba_loader]):
                    for num, loader in enumerate([self.valid_simens_loader, self.valid_ge_loader]):

                        for i, data_dict in enumerate(loader):
                            x_real = data_dict['image'].squeeze(4).to(self.device)
                            c_org  = data_dict['label']                            
                                
                            x_fake_list = [x_real]

                            # Translate images Checking for intra/inter class - Simens .
                            for c_fixed in c_simens_list:
                                c_trg = torch.cat([c_fixed, zero_f_ge, zero_f_philips, zero_f_toshiba, mask_f_simens], dim=1)
                                x_fake_list.append(self.G(x_real, c_trg))
                        
                            # Save the translated images.
                            x_concat = torch.cat(x_fake_list, dim=3)
                            if num == 0:
                                sample_path = os.path.join(self.sample_dir, 'B{0}_S{1}_to_S[B30f]_iter_{2}-images.jpg'.format(x_real.size(0), str(c_org.numpy()), i+1))
                            elif num == 1:
                                sample_path = os.path.join(self.sample_dir, 'B{0}_G{1}_to_S[B30f]_iter_{2}-images.jpg'.format(x_real.size(0), str(c_org.numpy()), i+1))
                            elif num == 2:
                                sample_path = os.path.join(self.sample_dir, 'B{0}_P{1}_to_S[B30f]_iter_{2}-images.jpg'.format(x_real.size(0), str(c_org.numpy()), i+1))
                            elif num == 3:
                                sample_path = os.path.join(self.sample_dir, 'B{0}_T{1}_to_S[B30f]_iter_{2}-images.jpg'.format(x_real.size(0), str(c_org.numpy()), i+1))
                            else : 
                                raise Exception('Error...! num')    

                            # sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                            save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                            print('Saved real and fake images into {}...'.format(sample_path))

                # Save model checkpoints.
                if (i+1) % self.model_save_step == 0:
                    G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                    D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))

                    if hasattr(self.G, 'module'):
                        torch.save(self.G.module.state_dict(), G_path)
                        torch.save(self.D.module.state_dict(), D_path)
                    else :
                        torch.save(self.G.state_dict(), G_path)
                        torch.save(self.D.state_dict(), D_path)

                    print('Saved model checkpoints into {}...'.format(self.model_save_dir))


                # Decay learning rates.
                if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                    g_lr -= (self.g_lr / float(self.num_iters_decay))
                    d_lr -= (self.d_lr / float(self.num_iters_decay))
                    self.update_lr(g_lr, d_lr)
                    print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    def test_multi(self):
        """Translate images using StarGAN trained on multiple datasets."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
    
        with torch.no_grad():
            
            for num, loader in enumerate([self.simens_loader, self.ge_loader, self.philips_loader, self.toshiba_loader]):
                # Translate images Checking for intra class - Simens .
                for i, data_dict in enumerate(loader):

                    x_real = data_dict['image'].squeeze(4)
                    c_org  = data_dict['label']

                    # Prepare input images and target domain labels.
                    x_real        = x_real.to(self.device)
                    # c_org ===  tensor([0, 0, 1, 2, 2, 2, 1, 0])
                    
                    c_simens_list = self.create_labels(c_org, self.c1_dim)

                    zero_ge       = torch.zeros(x_real.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
                    zero_philips  = torch.zeros(x_real.size(0), self.c3_dim).to(self.device)             # Zero vector for RaFD.
                    zero_toshiba  = torch.zeros(x_real.size(0), self.c4_dim).to(self.device)             # Zero vector for RaFD.

                    mask_simens   = self.label2onehot(torch.ones(x_real.size(0))*0, 4).to(self.device)     # Mask vector: [1, 0, 0, 0].

                    x_fake_list = [x_real]
                    for c_simens in c_simens_list:
                        c_trg = torch.cat([c_simens, zero_ge, zero_philips, zero_toshiba, mask_simens], dim=1)  # to simens
                        x_fake_list.append(self.G(x_real, c_trg))

                    # Save the translated images.
                    x_concat = torch.cat(x_fake_list, dim=3)
                    if num == 0:
                        result_path = os.path.join(self.result_dir, 'B{0}_S{1}_to_S[B30f]_iter_{2}-images.jpg'.format(x_real.size(0), str(c_org.numpy()), i+1))
                    elif num == 1:
                        result_path = os.path.join(self.result_dir, 'B{0}_G{1}_to_S[B30f]_iter_{2}-images.jpg'.format(x_real.size(0), str(c_org.numpy()), i+1))
                    elif num == 2:
                        result_path = os.path.join(self.result_dir, 'B{0}_P{1}_to_S[B30f]_iter_{2}-images.jpg'.format(x_real.size(0), str(c_org.numpy()), i+1))
                    elif num == 3:
                        result_path = os.path.join(self.result_dir, 'B{0}_T{1}_to_S[B30f]_iter_{2}-images.jpg'.format(x_real.size(0), str(c_org.numpy()), i+1))
                    else : 
                        raise Exception('Error...! num')    

                    # print("Check === ", self.denorm(x_concat.data.cpu()).shape, self.denorm(x_concat.data.cpu()).max(), self.denorm(x_concat.data.cpu()).min())
                    # Check ===  torch.Size([2, 1, 512, 2048]) tensor(0.9981) tensor(0.)
                    
                    # t = torch.clamp(self.denorm(x_concat.data.cpu()), min=0.2195, max=0.3050)
                    # t = (t - 0.2195)
                    # t = t / t.max()
                    # save_image(t, result_path, nrow=1, padding=0)

                    save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(result_path))
