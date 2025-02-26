import torch
from numpy.fft import fftshift, ifftshift, fftn, ifftn
import logging
import numpy as np


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1, 1)
    return a


def generalized_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds

def inference_steps(x, age, seq, t1, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            et = model(xt, age, t, t1)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )

            c2 = ((1 - at_next)).sqrt()
            x0_t[t1 == 0] = 0
            x0_t = x0_t[0].cpu()
            t1 = t1[0].cpu()
            k_space = fftshift(fftn(t1[0]))
            gt_k_space = fftshift(fftn(x0_t[0]))

            k_space_cut = np.zeros_like(k_space)
            k_space_cut2 = np.zeros_like(gt_k_space)
            k_space_cut[:, :, :] = k_space[:, :, :]
            k_space_cut2[:, :, :] = gt_k_space[:, :, :]
            k_space_cut[:, 110:131, :] = 0
            k_space_cut2[:, 110:131, :] = 0

            k_space_leave = np.zeros_like(k_space)
            k_space_leave2 = np.zeros_like(gt_k_space)
            k_space_leave[:, 110:131, :] = k_space[:, 110:131, :]
            k_space_leave2[:, 110:131, :] = gt_k_space[:, 110:131, :]

            alpha = 1
            k_space_add = k_space_cut2 + (1 - alpha) * k_space_leave2 + alpha * k_space_leave
            reconstructed_img_data = np.abs(ifftn(ifftshift(k_space_add)))
            reconstructed_img_data = torch.tensor(reconstructed_img_data, dtype=torch.float32)
            reconstructed_img_data = reconstructed_img_data.to(x.device)
            reconstructed_img_data = reconstructed_img_data.unsqueeze(0)
            logging.info(f"reconstructed_img_data's shape:{reconstructed_img_data.shape}")#torch.Size([1, 192, 240, 192])
            t1 = t1.unsqueeze(0)
            xt_next = at_next.sqrt() * reconstructed_img_data  + c2 * et + c1 * torch.randn_like(et)#torch.Size([1, 1, 192, 240, 192])
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds

def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
