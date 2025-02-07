import time
import torch
from functools import wraps
import torch.distributed as dist
from evaluate import evaluate

def only_rank_0(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if dist.get_rank() == 0:
            return func(*args, **kwargs)
    return wrapper

@only_rank_0
def print_log_iter(optimizer, iter_counter, iter_loss, logger):
    current_lr = optimizer.param_groups[0]['lr']
    aver_loss_im_heatmap = sum(iter_loss[0]) / len(iter_loss[0])
    aver_loss_im_score = sum(iter_loss[1]) / len(iter_loss[1])
    aver_loss_sematic_heatmap = sum(iter_loss[2]) / len(iter_loss[2])
    # aver_loss_mis_score = sum(iter_loss[3]) / len(iter_loss[3])
    aver_loss_mis_heatmap,aver_loss_mis_score = 0,0
    logger.info(f"Iteration {iter_counter}: Learning Rate = {current_lr}\n"
        f"Implausibility Heatmap Loss = {aver_loss_im_heatmap}, Implausibility Score Loss = {aver_loss_im_score}\n"
        f"Sematic Heatmap Loss = {aver_loss_sematic_heatmap}, Misalignment Score Loss = {aver_loss_mis_score}")
    print(f"Iteration {iter_counter}: Learning Rate = {current_lr}\n"
        f"Implausibility Heatmap Loss = {aver_loss_im_heatmap}, Implausibility Score Loss = {aver_loss_im_score}\n"
        f"Sematic Heatmap Loss = {aver_loss_sematic_heatmap}, Misalignment Score Loss = {aver_loss_mis_score}")

@only_rank_0
def eval_in_training(model, val_dataloader, device, criterion, iter_counter, logger):
    # val_loss = evaluate(model=model.module, dataloader=val_dataloader, device=device, criterion=criterion,criterion2=criterion2)
    val_loss = evaluate(model=model.module, dataloader=val_dataloader, device=device, criterion=criterion)
    logger.info(f"Iteration {iter_counter} Validation:\n"
                  f"Implausibility Heatmap Loss = {val_loss[0]}, Implausibility Score Loss = {val_loss[1]}\n"
                  f"Sematic Heatmap Loss = {val_loss[2]}, Misalignment Score Loss = {val_loss[3]}")
    print(f"Iteration {iter_counter} Validation:\n"
            f"Implausibility Heatmap Loss = {val_loss[0]}, Implausibility Score Loss = {val_loss[1]}\n"
            f"Sematic Heatmap Loss = {val_loss[2]}, Misalignment Score Loss = {val_loss[3]}")

@only_rank_0
def save_in_training(model, optimizer, scheduler, iter_counter, save_path):
    checkpoint = {
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(checkpoint, f'{save_path}/{iter_counter}.pth')
    print(f"Model weights saved to {save_path}/{iter_counter}.pth")

@only_rank_0
def final_save(model, optimizer, scheduler, start_time, save_path):
    end_time = time.time()
    checkpoint = {
      'model': model.module.state_dict(),
      'optimizer': optimizer.state_dict(),
      'scheduler': scheduler.state_dict()
    }
    total_minutes = (end_time - start_time) / 60
    torch.save(checkpoint, f'{save_path}/last.pth')
    print(f"Model weights saved to {save_path}/last.pth. Total training time: {total_minutes:.2f} minutes")