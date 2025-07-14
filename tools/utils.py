import argparse
import csv
import os
import re
import math
import random
import torch
import numpy as np
import torch.distributed as dist
from torchvision.utils import make_grid, save_image
from tools import dist_util
from tools.sampler import Sampler


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_random_seed(args, seed):
    rank = dist.get_rank() if args.parallel else 0
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def get_lr_lambda(args):
    return lambda step: warmup_cosine_lr(
        step, args.warmup_steps, args.total_steps,
        args.lr, args.final_lr, args.cosine_decay)


def warmup_cosine_lr(step, warmup_steps, total_steps, lr, final_lr, cosine_decay):
    if step < warmup_steps:
        return min(step, warmup_steps) / warmup_steps
    else:
        if cosine_decay:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return (final_lr + (lr - final_lr) * cosine_decay) / lr
        else:
            return 1
        
        
def save_checkpoint(args, step, model, optimizer, ema_model=None):
    if dist_util.is_main_process():
        checkpoint_dir = os.path.join('checkpoint', args.dataset, args.model)
        os.makedirs(checkpoint_dir, exist_ok=True)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step
        }
        if ema_model is not None:
            state['ema_model'] = ema_model.state_dict()
        filename = f"{args.mean_type}_{args.weight_type}_{args.beta_schedule}"
        
        if args.beta_schedule == "power":
            filename += f"_{args.p}"
            
        filename += f"_{step}.pth"
        filename = os.path.join(checkpoint_dir, filename)
        torch.save(state, filename)
        print(f"Checkpoint saved: {filename}")


def load_checkpoint(ckpt_path, model=None, optimizer=None, ema_model=None):
    if dist_util.is_main_process():
        print('==> Resuming from checkpoint..')
    assert os.path.exists(ckpt_path), 'Error: checkpoint {} not found'.format(ckpt_path)
    checkpoint = torch.load(ckpt_path)
    if model:
        model.load_state_dict(checkpoint['model'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if ema_model and 'ema_model' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_model'])
    return checkpoint


def generate_samples(args, step, device, eval_model, flow, save_grid=False):
    """Sample images from the model and either save them as a grid or for evaluation."""
    sampler = Sampler(args, device, eval_model, flow,)
    
    with torch.no_grad():
        all_samples, all_labels = sampler.sample(
            num_samples=args.num_samples if not save_grid else 64, 
            sample_size=args.sample_size, 
            image_size=args.image_size, 
            num_classes=args.num_classes, 
            progress_bar=not save_grid,)
        
    return save_images(args, step, all_samples,all_labels, save_grid)    
    
    
def save_images(args, step, samples, labels, save_grid=False):
    """Save sampled images as a grid."""
    if dist_util.is_main_process():
        arr = np.concatenate(samples, axis=0)
        arr = arr[: args.num_samples if not save_grid else 64]    
        
        if save_grid:
            # Save as grid image if 'save_grid' is True
            torch_samples = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
            grid = make_grid(torch_samples, pad_value=0.5)
            sample_dir = os.path.join(args.logdir, args.dataset, 'sample')
            os.makedirs(sample_dir, exist_ok=True)
            path = os.path.join(sample_dir, f'{step}.png')
            save_image(grid, path)
        else:
            # Save for evaluation purposes
            sample_dir = os.path.join(args.logdir, args.dataset, 'generate_sample', args.mean_type)
            os.makedirs(sample_dir, exist_ok=True)
            shape_str = "x".join([str(x) for x in arr.shape[1:3]])
            p = f"_{args.p}" if args.beta_schedule == "power" else ''
            out_path = os.path.join(sample_dir, f"{args.dataset}_{shape_str}_{args.model}_{args.weight_type}_{args.beta_schedule}{p}_samples.npz")
            
            if args.class_cond:
                label_arr = np.concatenate(labels, axis=0)[: args.num_samples]
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)
            print(f"Evaluation samples saved at {out_path}")

        return arr  # Return the sampled images array for evaluation
        
    return None    


def calculate_metrics(args, eval_model, **kwargs):
    
    step, device, flow, evaluator, ref_acts, ref_stats, ref_stats_spatial = (
        kwargs['step'], kwargs['device'],  kwargs['flow'], kwargs['evaluator'], 
        kwargs['ref_acts'], kwargs['ref_stats'], kwargs['ref_stats_spatial'])
    
    # Sample images and get the array
    arr = generate_samples(args, step, device, eval_model, flow)
    if dist_util.is_main_process():
        # Calculate metrics if in evaluation mode
        sample_batch = [np.array(arr[i:i + args.sample_size]) for i in range(0, len(arr), args.sample_size)]
        sample_acts = evaluator.compute_activations(sample_batch)

        sample_stats, sample_stats_spatial = tuple(evaluator.compute_statistics(x) for x in sample_acts)
        is_score = evaluator.compute_inception_score(sample_acts[0])   
        fid = sample_stats.frechet_distance(ref_stats)
        sfid = sample_stats_spatial.frechet_distance(ref_stats_spatial)
        pre, rec = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
        return is_score, fid, sfid, pre, rec
    
    return None, None, None, None, None


def save_metrics_to_csv(args, eval_dir, metrics, step):
    params = (
        f"{args.dataset}_{args.model}_"
        + (f"patch_{args.patch_size}_" if args.patch_size else "")
        + f"lr_{args.lr}_"  
        + f"betas_{args.betas}_"  
        + (f"lr_decay_{args.cosine_decay}_" if args.cosine_decay else "")        
        + f"dropout_{args.dropout}_"
        + f"drop_label_{args.drop_label_prob}_"
        + f"sample_t_{args.sample_steps}_"
        + f"cfg_{args.guidance_scale}_"
        + f"beta_sched_{args.beta_schedule}_" + (f"{args.p}_" if args.beta_schedule == 'power' else "")
        + f"target_{args.mean_type}_"
        + f"weight_{args.weight_type}_"  
        + ("cond_" if args.class_cond else "")
        )

    params = re.sub(r'[^\w\-_\. ]', '_', params).rstrip('_')

    csv_filename = os.path.join(eval_dir, f"{params}.csv")
    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Step'] + list(metrics.keys()))
        # writer.writerow([step] + list(metrics.values()))
        formatted_values = [f"{value:.2f}" if isinstance(value, (float, int)) else value for value in metrics.values()]
        writer.writerow([step] + formatted_values)

