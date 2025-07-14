"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
import torch 
from torchdiffeq import odeint
import numpy as np
import torch.distributed as dist
from tools.nn import mean_flat
from tools import logger
import torch.nn.functional as F
from einops import rearrange
from functools import partial



class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon
    VELOCITY = enum.auto() # the model predicts velocity alpha_t * epsilon - sigma_t * x_0
    VECTOR = enum.auto() # the model predicts v in flow matching d_sigma_t * epsilon - d_alpha_t * x_0
    SCORE = enum.auto()  # the model predicts the score function
    

# def _extract_into_tensor(arr, timesteps, broadcast_shape):
#     """
#     Extract values from a 1-D numpy array for a batch of indices.

#     :param arr: the 1-D numpy array.
#     :param timesteps: a tensor of indices into the array to extract.
#     :param broadcast_shape: a larger shape of K dimensions with the batch
#                             dimension equal to the length of timesteps.
#     :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
#     """
#     res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
#     while len(res.shape) < len(broadcast_shape):
#         res = res[..., None]
#     return res.expand(broadcast_shape)

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not dist.is_initialized():
        return output

    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def stopgrad(x):
    return x.detach()

def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()

class FlowMatching:
    def __init__(
        self,
        *,
        model_mean_type,
        mse_loss_weight_type='constant',
        path_type,         
        sampler_type="sde",   
        p2_gamma=1,
        p2_k=1,
        atol=1e-6,
        rtol=1e-5,
    ):
        self.path_type = path_type        
        self.model_mean_type = model_mean_type
        self.mse_loss_weight_type = mse_loss_weight_type
        self.sampler_type = sampler_type
        # P2 weighting
        self.p2_gamma = p2_gamma
        self.p2_k = p2_k
        #sample tolenace
        self.atol = atol
        self.rtol = rtol
        
    def expand_t_like_x(self, t, x):
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        dims = [1] * (len(x.size()) - 1)
        return t.view(t.size(0), *dims).to(x)
    
    def float_equal(self, num1, num2, eps=1e-8):
        return abs(num1 - num2) < eps
    
    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = torch.full_like(t, -1.0)  
            d_sigma_t = torch.full_like(t, 1.0)  
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def convert_model_output_to_vector(self, model_output, x_t, t):
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t)

        if self.model_mean_type == ModelMeanType.START_X:
            start_x = model_output
            noise = (x_t - alpha_t * start_x) / sigma_t
            
        elif self.model_mean_type == ModelMeanType.EPSILON:
            noise = model_output
            start_x = (x_t - sigma_t * noise) / alpha_t
            
        elif self.model_mean_type == ModelMeanType.VELOCITY:
            # v = α_t * ε - σ_t * x₀ → solve ε and x₀
            start_x = (alpha_t * x_t - sigma_t * model_output) / (alpha_t**2 + sigma_t**2)
            noise = (sigma_t * x_t + alpha_t * model_output) / (alpha_t**2 + sigma_t**2)
            
        elif self.model_mean_type == ModelMeanType.VECTOR:
            return model_output
        
        else:
            raise NotImplementedError("Unsupported model_mean_type for vector")

        vector = d_alpha_t * start_x + d_sigma_t * noise
        return vector

    def convert_model_output_to_score(self, model_output, x_t, t):
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t)

        if self.model_mean_type == ModelMeanType.START_X:
            start_x = model_output
            score = -(x_t - alpha_t * start_x) / (sigma_t ** 2)

        elif self.model_mean_type == ModelMeanType.EPSILON:
            noise = model_output
            score = -noise / sigma_t

        elif self.model_mean_type == ModelMeanType.VELOCITY:
            # v = α_t * ε - σ_t * x₀ → solve ε and x₀
            noise = (sigma_t * x_t + alpha_t * model_output) / (alpha_t**2 + sigma_t**2)
            score = -noise / sigma_t

        elif self.model_mean_type == ModelMeanType.VECTOR:
            # start_x = (sigma_t * model_output - d_sigma_t * x_t) / (sigma_t * d_alpha_t - alpha_t * d_sigma_t)
            noise = (d_alpha_t * x_t - alpha_t * model_output) / (sigma_t * d_alpha_t - alpha_t * d_sigma_t)
            score = -noise / sigma_t
        elif self.model_mean_type == ModelMeanType.SCORE:
            return model_output
        
        else:
            raise NotImplementedError("Unsupported model_mean_type for score")

        return score
    
    def q_sample(self, x_start, noise, t,):
        t = self.expand_t_like_x(t, x_start)
        alpha_t, sigma_t, _, _ = self.interpolant(t)
        x_t = alpha_t * x_start + sigma_t * noise
        return x_t
    
    #taining
    def training_losses(self, model, x_start, t=None, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        if t is None:
            t = torch.rand(x_start.shape[0], device=x_start.device)
        
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t)     
                     
        x_t = self.q_sample(x_start, noise, t)
        
        terms = {}
        
        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
            ModelMeanType.VELOCITY: alpha_t[:, None, None, None] * noise - sigma_t[:, None, None, None] * x_start,
            ModelMeanType.VECTOR: d_alpha_t[:, None, None, None] * x_start + d_sigma_t[:, None, None, None] * noise,
            ModelMeanType.SCORE: - noise / sigma_t[:, None, None, None],
            # ModelMeanType.UNRAVEL: unravel,
        }[self.model_mean_type]
        
        model_output = model(x_t, t, **model_kwargs)
        
        assert model_output.shape == target.shape == x_start.shape

        raw_mse = mean_flat((target - model_output) ** 2)
            
        terms["loss"] = raw_mse

        return terms

    #sampling
    def forward_with_cfg(self, model, x, t_in, guidance_scale, **model_kwargs):
        t = t_in.view(x.shape[0]) # make sure the shape fo t inputs mdoel is [batch_dim]
        model_output = model(x, t, **model_kwargs)
        guidance_scale = guidance_scale(t_in.mean().item()) if callable(guidance_scale) else guidance_scale
        if not self.float_equal(guidance_scale, 1.0):
            cond, uncond = torch.split(model_output, len(model_output) // 2, dim=0)
            cond = uncond + guidance_scale * (cond - uncond)
            model_output = torch.cat([cond, cond], dim=0)
        # convert to score or vector field
        return model_output #self.convert_model_output(model_output, x, t_in)

    def ode_sample(self, model, noise, device, num_steps=50, solver='dopri5', guidance_scale=1.0, **model_kwargs):
        timesteps = torch.linspace(1.0, 0.0, num_steps, device=device)
        
        def guided_drift(t, x):
            t_in = self.expand_t_like_x(t, x)
            model_output = self.forward_with_cfg(model, x, t_in, guidance_scale, **model_kwargs)
            return self.convert_model_output_to_vector(model_output, x, t_in)
        
        samples = odeint(
            func=guided_drift,
            y0=noise,
            t=timesteps,
            method=solver,
            rtol=self.rtol,
            atol=self.atol,
        )
        return samples[-1]
    
    def compute_diffusion(self, t_cur):
        # return self.interpolant(t_cur)[1]
        return self.interpolant(t_cur)[1] ** 2
    
    def sde_sample(self, model, noise, device, num_steps=50, solver='heun', guidance_scale=1.0, **model_kwargs):
        """
        SDE sampler with Euler or Heun method and final deterministic step.
        x_t is the initial latent (x_T), denoised to x_0.
        """
        def compute_drift(x, t_tensor, diffusion):
            out = self.forward_with_cfg(model, x, t_tensor, guidance_scale, **model_kwargs)
            s = self.convert_model_output_to_score(out, x, t_tensor)
            v = self.convert_model_output_to_vector(out, x, t_tensor)
            return v - 0.5 * diffusion * s

        t_steps = torch.linspace(1.0, 0.04, num_steps, dtype=torch.float64, device=device)
        t_steps = torch.cat([t_steps, torch.tensor([0.0], dtype=torch.float64, device=device)])
        x_t = noise

        with torch.no_grad():
            for t_cur, t_next in zip(t_steps[:-2], t_steps[1:-1]):
                dt = t_next - t_cur
                t_tensor = self.expand_t_like_x(t_cur, x_t)
                diffusion = self.compute_diffusion(t_tensor)

                d_cur = compute_drift(x_t, t_tensor, diffusion)

                eps = torch.randn_like(x_t)
                noise_term = torch.sqrt(diffusion) * eps * torch.sqrt(torch.abs(dt))

                if solver == 'euler':
                    x_t = x_t + d_cur * dt + noise_term

                elif solver == 'heun':
                    x_pred = x_t + d_cur * dt + noise_term
                    t_next_tensor = self.expand_t_like_x(t_next, x_pred)
                    diffusion_next = self.compute_diffusion(t_next_tensor)
                    d_next = compute_drift(x_pred, t_next_tensor, diffusion_next)
                    x_t = x_t + 0.5 * (d_cur + d_next) * dt + noise_term

                else:
                    raise ValueError(f"Unknown solver: {solver}")

            # Final deterministic step
            t_cur, t_next = t_steps[-2], t_steps[-1]
            dt = t_next - t_cur
            t_tensor = self.expand_t_like_x(t_cur, x_t)
            diffusion = self.compute_diffusion(t_tensor)

            d_cur = compute_drift(x_t, t_tensor, diffusion)
            mean_x = x_t + d_cur * dt  # no noise

        return mean_x

    def sample(self, model, noise, device, num_steps=50, solver='heun', guidance_scale=1.0, **model_kwargs):
        if self.sampler_type == "ode": 
            return self.ode_sample(model, noise, device, num_steps, solver=solver, guidance_scale=guidance_scale, **model_kwargs)
        elif self.sampler_type == "sde": 
            return self.sde_sample(model, noise, device, num_steps, solver=solver, guidance_scale=guidance_scale, **model_kwargs)
        else: 
            raise NotImplementedError(f"Unsupported sampler_type: {self.sampler_type}")






class MeanFlow:
    def __init__(
        self,

        # mean flow settings
        flow_ratio=0.50,
        # time distribution, mu, sigma
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0.10,
        # set scale as none to disable CFG distill
        cfg_scale=2.0,
        # experimental
        cfg_uncond='v',
        jvp_api='autograd',
    ):
        super().__init__()
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.cfg_ratio = cfg_ratio
        self.w = cfg_scale

        self.cfg_uncond = cfg_uncond
        self.jvp_api = jvp_api

        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

    # fix: r should be always not larger than t
    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

        # Assign t = max, r = min, for each pair
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    def loss(self, model, x, c=None):
        batch_size = x.shape[0]
        device = x.device

        t, r = self.sample_t_r(batch_size, device)

        t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
        r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

        e = torch.randn_like(x)

        z = (1 - t_) * x + t_ * e
        v = e - x

        if c is not None:
            assert self.cfg_ratio is not None
            uncond = torch.ones_like(c) * self.num_classes
            cfg_mask = torch.rand_like(c.float()) < self.cfg_ratio
            c = torch.where(cfg_mask, uncond, c)
            if self.w is not None:
                with torch.no_grad():
                    u_t = model(z, t, t, uncond)
                v_hat = self.w * v + (1 - self.w) * u_t
                if self.cfg_uncond == 'v':
                    # offical JAX repo uses original v for unconditional items
                    cfg_mask = rearrange(cfg_mask, "b -> b 1 1 1").bool()
                    v_hat = torch.where(cfg_mask, v, v_hat)
            else:
                v_hat = v

        # forward pass
        # u = model(z, t, r, y=c)
        model_partial = partial(model, y=c)
        jvp_args = (
            lambda z, t, r: model_partial(z, t, r),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )

        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        u_tgt = v_hat - (t_ - r_) * dudt

        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)
        # loss = F.mse_loss(u, stopgrad(u_tgt))

        mse_val = (stopgrad(error) ** 2).mean()
        return loss, mse_val

    @torch.no_grad()
    def sample_each_class(self, model, n_per_class, classes=None,
                          sample_steps=5, device='cuda'):
        model.eval()

        if classes is None:
            c = torch.arange(self.num_classes, device=device).repeat(n_per_class)
        else:
            c = torch.tensor(classes, device=device).repeat(n_per_class)

        z = torch.randn(c.shape[0], self.channels,
                        self.image_size, self.image_size,
                        device=device)

        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)

        # print(t_vals)

        for i in range(sample_steps):
            t = torch.full((z.size(0),), t_vals[i], device=device)
            r = torch.full((z.size(0),), t_vals[i + 1], device=device)

            # print(f"t: {t[0].item():.4f};  r: {r[0].item():.4f}")

            t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
            r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

            v = model(z, t, r, c)
            z = z - (t_-r_) * v

        z = self.normer.unnorm(z)
        return z