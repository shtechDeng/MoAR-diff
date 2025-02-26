import torch


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          c1: torch.Tensor,
                          c2: torch.LongTensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, c2, t.float(), c1)
    # print('x:', x.shape)
    # print('e:', e.shape)
    # print('out:', output.shape)
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3, 4))
    else:
        return (e - output).square().sum(dim=(1, 2, 3, 4)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}
