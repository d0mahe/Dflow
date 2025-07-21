import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tools import dist_util, logger
# from .resample import LossAwareSampler, UniformSampler, create_named_schedule_sampler
import csv
import os
import torch.distributed as dist

def ema(source, target, decay):
    with torch.no_grad():
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * decay + source_dict[key].data * (1 - decay))

class Trainer:
    def __init__(self, args, device, model, ema_model, optimizer, scheduler, diffusion, train_loader, start_step, pbar=None):
        self.args = args
        self.device = device        
        self.model = model
        self.ema_model = ema_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.datalooper = iter(train_loader)
        # self.schedule_sampler = create_named_schedule_sampler(args.sampler_type, diffusion)
        self.scaler = GradScaler() if args.amp else None
        self.start_step = start_step        
        self.pbar = pbar

    def _get_next_batch(self):
        try:
            images, labels = next(self.datalooper)
        except StopIteration:
            self.datalooper = iter(self.train_loader)
            images, labels = next(self.datalooper)
        return images.to(self.device), labels.to(self.device) if self.args.class_cond else None
            
    def _compute_loss(self, images, labels):
        model_kwargs = {"y": labels} if self.args.class_cond else {}
        loss_dict = self.diffusion.training_losses(self.model, images, model_kwargs=model_kwargs)
        return (loss_dict["loss"]).mean()

    def _apply_gradient_clipping(self):
        if self.args.grad_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

    def _update_ema(self):
        if dist_util.is_main_process():
            ema(self.model, self.ema_model, self.args.ema_decay)
                   
    def _sample_from_latent(self, latent, latent_scale=1.):
        mean, std = torch.chunk(latent, 2, dim=1)
        latent_samples = mean + std * torch.randn_like(mean)
        latent_samples = latent_samples * latent_scale 
        return latent_samples 

    def train_step(self, step):
        self.model.train()
        if self.args.parallel:
            self.train_loader.sampler.set_epoch(step)
        
        grad_accumulation = max(1, self.args.grad_accumulation)  # Ensure cumulative steps are least 1
        loss_accumulated = 0.0

        for accumulation_step in range(grad_accumulation):
            images, labels = self._get_next_batch()
            
            if self.args.in_chans == 4:
                images = self._sample_from_latent(images, self.args.latent_scale)  
                
            if self.args.amp:
                with autocast():
                # with autocast(dtype=torch.bfloat16):
                    loss = self._compute_loss(images, labels) / grad_accumulation  # Scale loss for accumulation
                self.scaler.scale(loss).backward()
            else:
                loss = self._compute_loss(images, labels) / grad_accumulation  # Scale loss for accumulation
                loss.backward()
                
            loss_accumulated += loss.item()

            # Perform optimization step only after grad_accumulation steps
            if (accumulation_step + 1) % grad_accumulation == 0:
                if self.args.amp:
                    if self.args.grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        self._apply_gradient_clipping()
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self._apply_gradient_clipping()
                    
                    self.optimizer.step()
                self.optimizer.zero_grad()
        
        # Update scheduler
        self.scheduler.step()
        
        if dist_util.is_main_process():
            self._update_ema()
            self.pbar.update(1)
            self.pbar.set_postfix(loss=loss_accumulated)

        return loss_accumulated  # Return scalar accumulated loss 
