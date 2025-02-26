import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

from skimage.exposure import rescale_intensity

import torchvision.utils as tvu

import nibabel as nib
import imageio

def proc_month(month):
    if month == '4yr':
        month = '48mo'
    elif month == '2wk':
        month = '0.5mo'
    elif month == '09mo':
        month = '9mo'

    bins = [4, 7, 13, 25, 100]
    for idx in range(len(bins)):
        if float(month[:-2]) <= bins[idx]:
            return idx
        

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps, s=0.008):  #获得β
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == 'cosine':
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = num_diffusion_timesteps + 1
        x = torch.linspace(0, num_diffusion_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_diffusion_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0] #这求得是哪一个值呢？
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999).numpy()
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def scale_data(data): #需要吗？
    p10 = np.percentile(data, 10)
    data[data<p10] = p10
    data -= p10
    
    p99 = np.percentile(data, 99.5)
    data[data>p99] = p99
    data /= p99
    data *= 255

    return data

def save_gif(data, idx, save_path):
    frames = []
    for i in range(len(data[0])):
        data_one = scale_data(data[0][i][idx][0].detach().cpu().numpy())

        fig=np.zeros([data_one.shape[1]+data_one.shape[2], data_one.shape[0]+data_one.shape[1]], dtype=np.uint8)

        fig[:data_one.shape[2],:data_one.shape[0]] = data_one[:,data_one.shape[1]//2,::-1].T
        fig[data_one.shape[2]:data_one.shape[2]+data_one.shape[1],:data_one.shape[0]] = data_one[:,::-1,data_one.shape[2]//2].T
        fig[:data_one.shape[2],data_one.shape[0]:data_one.shape[0]+data_one.shape[1]] = data_one[data_one.shape[0]//2,:,::-1].T

        frames.append(fig)
    
    frames.append(fig)
    frames.append(fig)
    frames.append(fig)
    
    imageio.mimsave(save_path, frames, 'GIF', duration=0.05)

def save_nifti(data, ref_path, save_path):
    data = data[0].detach().cpu().numpy()
    data *= 1000
    ref_data = nib.load(ref_path)
    nib.Nifti1Image(data, ref_data.affine).to_filename(save_path)


def proc_nib_data(nib_data):
    p10 = np.percentile(nib_data, 10)
    p99 = np.percentile(nib_data, 99.9)

    nib_data[nib_data<p10] = p10
    nib_data[nib_data>p99] = p99

    m = np.mean(nib_data, axis=(0, 1, 2))
    s = np.std(nib_data, axis=(0, 1, 2))
    nib_data = (nib_data - m) / s

    nib_data = torch.tensor(nib_data, dtype=torch.float32)

    return nib_data


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
            
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        self.num_denoising_timesteps=config.diffusion.num_denoising_timesteps
        self.num_diffusion_addingsteps=config.diffusion.num_diffusion_addingsteps

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        def check_requires_grad(module):
            for name, param in module.named_parameters():
                if param.requires_grad:
                    print(f"Parameter {name} requires gradient.")
                else:
                    print(f"Parameter {name} does not require gradient.")
        args, config = self.args, self.config
        # tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        model = Model(config) #Unet

        model = model.to(self.device)
        model = torch.nn.DataParallel(model) #多卡

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load("/public/home/wangkd2023/Denghw/2024Codes/CBCP_UnDPM_with_age_finetune/exp/logs/Fusion_diffusion/ckpt_150000.pth")
            new_model_dict = model.state_dict()
            new_model_dict_con = model.module.control_model.state_dict()
            pretrain_state = states[0]
            for k in list(new_model_dict.keys()):
                if k in pretrain_state:
                    new_model_dict[k] = pretrain_state[k]

            for k in list(model.module.control_model.state_dict().keys()):
                if "module."+k in pretrain_state:
                    new_model_dict_con[k] = pretrain_state["module."+k]

            model.load_state_dict(new_model_dict)
            model.module.control_model.load_state_dict(new_model_dict_con)
            for name, param in model.named_parameters():
                if "control_model" in name:
                    param.requires_grad = True
                
                else:
                    param.requires_grad = False
            
            check_requires_grad(model.module)



            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            
            # pretrain_optimizer_state = states[1]
            # for i in list(new_model_dict.keys()):
            #     if i in pretrain_optimizer_state:
            #         new_optimizer_dict[i] = pretrain_optimizer_state[i]

            # optimizer.load_state_dict(new_optimizer_dict)
            # optimizer.load_state_dict(states[1])
            
            # start_epoch = states[2]
            # step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])
            print("load weight sucess") #加载权重成功
        # model.eval()
        # model.control_model.train()
        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, c1, c2, fn) in enumerate(train_loader):
                # print('shape', x.shape, c1.shape, c2.shape)
                n = x.size(0)
                data_time += time.time() - data_start
                
                step += 1

                x = x.to(self.device)
                c1 = c1.to(self.device)
                c2 = c2.to(self.device)
                # x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, c1, c2, t, e, b)

                # tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item():12.4f}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

    def inference(self):
        model = Model(self.config) #无条件的Unet

        states = torch.load(
            "/public/home/wangkd2023/Denghw/2024Codes/UnDPM_with_age/exp/logs/Age_diffusion/ckpt_160000.pth",
            map_location=self.config.device,
        )

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
        else:
            ema_helper = None
        
        model.eval()

        #c1 = proc_nib_data(nib.load(self.args.reference_img).get_fdata())#读入有伪影的图片
        c1 = nib.load(self.args.reference_img).get_fdata()
        p10 = np.percentile(c1, 10)
        p99 = np.percentile(c1, 99)
        c1 = rescale_intensity(c1, in_range=(p10, p99), out_range=(0, 1))
        m = np.mean(c1, axis=(0, 1, 2))
        s = np.std(c1, axis=(0, 1, 2))
        c1 = (c1 - m) / s
        c1 = torch.tensor(c1, dtype=torch.float32)
        c1 = c1.unsqueeze(0).unsqueeze(0)
        c1 = c1.to(self.device)


        b = self.betas
        e = torch.randn_like(c1)#与增加维度后的c1一样
        logging.info(self.num_timesteps)
        t = torch.randint(
            low=0, high=self.num_timesteps, size=(1 // 2 + 1,)#batch size为1不会报错
        ).to(self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:1]        
        a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1, 1)
        logging.info(a)
        logging.info(t)
        x = c1 * a.sqrt() + e * (1.0 - a).sqrt()
        save_nifti(
            x[0], self.args.reference_img, os.path.join(self.args.inference_folder, f"noise_xt.nii.gz"))
        logging.info("噪声已输出")
        x = self.inference_image(x, model, (not self.args.inference_gif)) #这里的x直接改为加了噪声后的x就行了
        #记得把时间t给改一下
        save_nifti(
            x[0], self.args.reference_img, os.path.join(self.args.inference_folder, f"fake_initial.nii.gz"))
        logging.info("结果已输出")
    def inference_all(self):
        args, config = self.args, self.config
        # tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        inference_loader = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        model = Model(self.config)

        states = torch.load(
            "/public/home/wangkd2023/Denghw/2024Codes/CBCP_UnDPM_with_age_finetune/exp/logs/finetuneDPM_with_age/ckpt_100000.pth",
            map_location=self.config.device,
        )

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
        else:
            ema_helper = None
        
        model.eval()
        b = self.betas
        t = torch.tensor([self.num_diffusion_addingsteps], device=self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:1]        
        a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1, 1)#这里t自己定义一下
        logging.info(a)
        logging.info(t)
        for i, (t1, age, fn) in enumerate(inference_loader):
            logging.info("运行到：")
            logging.info(i)
            fn1 = "/".join(fn[0].split("/")[:-1])
            logging.info("当前结果将保存到：")
            t1 = t1.to(self.device)
            e = torch.randn_like(t1)#与增加维度后的c1一样
            x = t1 * a.sqrt() + e * (1.0 - a).sqrt()
            x = self.inference_image(x, age, t1, model, (not self.args.inference_gif)) #这里的x直接改为加了噪声后的x就行了
            logging.info(f"x的形状是:{x.shape}")
            fn3 = "/".join(fn[0].split("/")[:6])
            fn4 = "/".join(fn[0].split("/")[7:-1])
            logging.info(os.path.join(fn3, "Results/T2_norm12", fn4))
            os.makedirs(os.path.join(fn3, "Results/T2_norm12", fn4), exist_ok='True')
            #记得把时间t给改一下
            save_nifti(
                x[0], fn[0], os.path.join(fn3, "Results/T2_norm12", fn4, f"T1_age.nii.gz"))
            logging.info("结果已输出")
    def generate(self):
        model = Model(self.config)

        states = torch.load(
            "/public/home/zhuzh2023/infant_barin_generate/code_20230228_inference_copy/generate_model_6_from_20221217_model7_copy/exp/logs/diffusion3/ckpt_455000.pth",
            map_location=self.config.device,
        )

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
        else:
            ema_helper = None
        
        model.eval()
        main_month = self.args.inference_folder.split("/")[-1]
        x = torch.randn(
            [1, 1, 80, 96, 80],
            device=self.device,
        )

        c1 = proc_nib_data(nib.load(self.args.reference_img).get_fdata())
        c1 = c1.reshape(1, 1, 80, 96, 80).repeat(1, 1, 1, 1, 1)
        c2 = torch.tensor(proc_month1(main_month)).reshape(1)

        c1 = c1.to(self.device)
        c2 = c2.to(self.device)

        # print('c1.shape', c1.shape)
        # print('c2.shape', c2.shape)

        x = self.inference_image(x, c1, c2, model)
    
        save_nifti(
            x[0], os.path.join(self.args.inference_folder, f"lr_croped.nii.gz"))
        # save_nifti(
        #     x[proc_month1(main_month)], os.path.join(self.args.inference_folder, f"{main_month}.nii.gz"))

    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 400
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x
    
    def inference_image(self, x, age, t1, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":#true
            if self.args.skip_type == "uniform":#true
                skip = self.num_denoising_timesteps // self.args.timesteps# 改这里不久完事了吗？
                seq = range(0, self.num_denoising_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_denoising_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import inference_steps

            xs = inference_steps(x, age, seq, t1, model, self.betas, eta=self.args.eta)
            x = xs
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1] #返回第一列最后一个
        return x

    def test(self):
        pass
