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


class FlowMatching:
    def __init__(
        self,
        *,
        model_mean_type,
        path_type,         
        flow_ratio=0.50, #control how many time steps satisfy r=t, when flow ratio = 1.0,  mean flow degrades as flow matching
        time_dist=['lognorm', -0.4, 1.0], #sampling timestep interval [r, t].
        mf_cfg_ratio=0.10,
        # set scale as none to disable CFG distill
        mf_cfg_scale=2.0,
        # experimental
        # mf_cfg_uncond='v',
        jvp_api='autograd',
        
        sampler_type="sde",   
        interval=(1.0, 0.0), # for mean flow sampling, the interval of t and r
        atol=1e-6,
        rtol=1e-5,
    ):
        # flow matching settings
        self.path_type = path_type        
        self.model_mean_type = model_mean_type
        
        self.use_mean_flow = flow_ratio < 1.0
        
        # mean flow settings
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.mf_cfg_ratio = mf_cfg_ratio
        # self.mf_cfg_uncond = mf_cfg_uncond        
        self.w = mf_cfg_scale
        self.jvp_api = jvp_api
        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True
        
        #sample settings
        self.atol = atol
        self.rtol = rtol
        self.interval = interval
        self.sampler_type = sampler_type
                    
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

    # fix: r should be always not larger than t
    def sample_t_r(self, x_start):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(x_start.shape[0], 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(x_start.shape[0], 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid, ensure samples are in [0, 1]

        # Assign t = max, r = min, for each pair
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * x_start.shape[0])
        indices = np.random.permutation(x_start.shape[0])[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=x_start.device)
        r = torch.tensor(r_np, device=x_start.device)
        return t, r

    
    def apply_cfg_for_mf(self, model, x_t, t, v_hat, c):
        """
        Apply classifier-free guidance and return modified v_hat.
        """
        if c is None or self.mf_cfg_ratio is None:
            return v_hat

        uncond = torch.ones_like(c) * getattr(self, "num_classes", 1000)
        cfg_mask = torch.rand_like(c.float()) < self.mf_cfg_ratio
        c_cfg = torch.where(cfg_mask, uncond, c)

        if self.w is not None:
            with torch.no_grad():
                u_t = model(x_t, t, t, c_cfg)
            v_hat_guided = self.w * v_hat + (1 - self.w) * u_t

            # if self.mf_cfg_uncond == 'v':
            cfg_mask = rearrange(cfg_mask, "b -> b 1 1 1").bool()
            v_hat_guided = torch.where(cfg_mask, v_hat, v_hat_guided)
            return v_hat_guided
        else:
            return v_hat
        
    def compute_u_target(self, model, x_t, t, r, v_hat, c):
        """
        Compute model output and u_target = v_hat - (t - r) * ∂u/∂t
        """
        v_hat = self.apply_cfg_for_mf(model, x_t, t, v_hat, c)
        model_partial = partial(model, y=c) if c is not None else model

        jvp_args = (
            lambda z, t, r: model_partial(z, t, r),
            (x_t, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )

        u, dudt = self.jvp_fn(*jvp_args, create_graph=self.create_graph)
        # t_ = self.expand_t_like_x(t, x_t)
        # r_ = self.expand_t_like_x(r, x_t)
        t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
        r_ = rearrange(r, "b -> b 1 1 1").detach().clone()
        u_target = v_hat - (t_ - r_) * dudt
        return u, u_target

    def q_sample(self, x_start, noise, t,):
        t = self.expand_t_like_x(t, x_start)
        alpha_t, sigma_t, _, _ = self.interpolant(t)
        x_t = alpha_t * x_start + sigma_t * noise
        return x_t
    
    #taining
    def training_losses(self, model, x_start, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
            
        t, r = self.sample_t_r(x_start)
        if self.use_mean_flow:
            model_kwargs["r"] = r
            
        x_t = self.q_sample(x_start, noise, t)
        # _, _, d_alpha_t, d_sigma_t = self.interpolant(t)      
        # velocity = d_alpha_t[:, None, None, None] * x_start + d_sigma_t[:, None, None, None] * noise     
           
        _, _, d_alpha_t, d_sigma_t = self.interpolant(self.expand_t_like_x(t, x_t)) 
        velocity = d_alpha_t * x_start + d_sigma_t * noise
        
        y = model_kwargs.get("y", None)
        
        terms = {}
        # compute the model output and target according to flow ratio
        if self.use_mean_flow:
            u, u_target = self.compute_u_target(model, x_t, t, r, velocity, c=y)
            model_output = u            
            target = u_target.detach()
        else:
            model_output = model(x_t, t, **model_kwargs)
            target = velocity

        raw_mse = mean_flat((target - model_output) ** 2)
        
        if self.use_mean_flow:
            # if flow ratio is less than 1.0, we compute adaptive weight
            mse_loss_weight = self.adaptive_weight(raw_mse.detach())
        else:
            mse_loss_weight = torch.ones_like(t)
            
        terms["loss"] = mse_loss_weight * raw_mse

        return terms

    def adaptive_weight(self, raw_mse, gamma=0.5, c=1e-3):
        """
        Adaptive weight: w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
        Args:
            raw_mse: Tensor of shape (B, C, W, H)
            gamma: Power used in original ||Δ||^{2γ} loss
            c: Small constant for stability
        Returns:
            weight
        """
        p = 1.0 - gamma
        w = 1.0 / (raw_mse + c).pow(p)
        return w
    
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
            # return self.convert_model_output_to_vector(model_output, x, t_in)
            return model_output
        
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

    def convert_velocity_to_score(self, model_output, x_t, t):
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t)
        # start_x = (sigma_t * model_output - d_sigma_t * x_t) / (sigma_t * d_alpha_t - alpha_t * d_sigma_t)
        noise = (d_alpha_t * x_t - alpha_t * model_output) / (sigma_t * d_alpha_t - alpha_t * d_sigma_t)
        score = -noise / sigma_t
        return score
        
    def sde_sample(self, model, noise, device, num_steps=50, solver='heun', guidance_scale=1.0, **model_kwargs):
        """
        SDE sampler with Euler or Heun method and final deterministic step.
        x_t is the initial latent (x_T), denoised to x_0.
        """
        def compute_drift(x, t_tensor, diffusion):
            out = self.forward_with_cfg(model, x, t_tensor, guidance_scale, **model_kwargs)
            s = self.convert_velocity_to_score(out, x, t_tensor)
            v = out
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
    
    def mean_flow_sample(self, model, z, device, num_steps=1, classes=None,):
        start, end = self.interval
        t_vals = torch.linspace(start, end, num_steps + 1, device=device)
        # print(t_vals)
        for i in range(num_steps):
            t = torch.full((z.size(0),), t_vals[i], device=device)
            r = torch.full((z.size(0),), t_vals[i + 1], device=device)
            # print(f"t: {t[0].item():.4f};  r: {r[0].item():.4f}")
            t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
            r_ = rearrange(r, "b -> b 1 1 1").detach().clone()
            u = model(z, t, r, classes)
            z = z - (t_-r_) * u

        return z
    
    def sample(self, model, noise, device, num_steps=50, solver='heun', guidance_scale=1.0, **model_kwargs):
        if self.use_mean_flow:
            return self.mean_flow_sample(model, noise, device, num_steps, **model_kwargs)
        else:
            if self.sampler_type == "ode": 
                return self.ode_sample(model, noise, device, num_steps, solver=solver, guidance_scale=guidance_scale, **model_kwargs)
            elif self.sampler_type == "sde": 
                return self.sde_sample(model, noise, device, num_steps, solver=solver, guidance_scale=guidance_scale, **model_kwargs)
            else: 
                raise NotImplementedError(f"Unsupported sampler_type: {self.sampler_type}")

