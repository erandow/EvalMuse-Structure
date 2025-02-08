import torch
from model.model_final import RAHF
from PIL import Image
import numpy as np
from scipy.stats import spearmanr
from dataset import RAHFDataset
from torch.utils.data import DataLoader, Dataset
import pickle
import os

def get_plcc_srcc(output_scores, gt_scores):
    # for (output_scores, gt_scores) in zip(output_scores_list, gt_scores_list):
    output_scores = np.array(output_scores)
    gt_scores = np.array(gt_scores)
    # Calculate PLCC (Pearson Linear Correlation Coefficient)
    plcc = np.corrcoef(gt_scores, output_scores)[0, 1]

    # Calculate SRCC (Spearman Rank Correlation Coefficient)
    srcc, _ = spearmanr(gt_scores, output_scores)

    print(f'PLCC: {plcc}')
    print(f'SRCC: {srcc}')

def ignore_edge(heatmap):
    heatmap[0:5, :] = 0            # 顶部边缘
    heatmap[-1:-5, :] = 0           # 底部边缘
    heatmap[:, 0:5] = 0            # 左侧边缘
    heatmap[:, -1:-5] = 0           # 右侧边缘
    return heatmap

def compute_num_params(model):
    import collections
    params = collections.defaultdict(int)
    bytes_per_param = 4
    for name, module in model.named_modules():
      model_name = name.split('.')[0]
      if list(module.parameters()):  # 只处理有参数的模块
          total_params = sum(p.numel() for p in module.parameters())
          memory_usage_mb = (total_params * bytes_per_param) / (1024 * 1024)
        #   print(f"模块: {name}, 参数总量: {total_params}, 显存占用: {memory_usage_mb:.2f} MB")
          params[model_name] += memory_usage_mb
    
    for k, v in params.items():
        print(k, v,"MB")

def save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def save_heatmap_mask(input_tensor, threshold, img_name, save_path, process_edge=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    vis_path = f'{save_path}_vis'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)   
    input_tensor = torch.where(input_tensor > threshold, 1, 0)
    input_numpy = input_tensor.squeeze(0).cpu().numpy().astype(np.uint8)
    if process_edge:
        input_numpy = ignore_edge(input_numpy)
    vis_numpt = input_numpy * 255
    # Convert to PIL Image
    pil_image = Image.fromarray(input_numpy[0])
    # Save the PIL Image
    pil_image.save(f"{save_path}/{img_name}.png")
    pil_vis = Image.fromarray(vis_numpt[0])
    pil_vis.save(f"{vis_path}/{img_name}.png")

    
def process_segment_output(outputs):
    normed = torch.softmax(outputs,dim=1)
    foreground = normed[:,1,:,:]
    binary_mask = (foreground>0.5).float().squeeze(0)
    return binary_mask

def compute_badcase_detect_rate(output, target):
    if not output:
        return 0
    assert len(output) == len(target), "output and target must have the same length"
    det_count = 0
    for out_score, tar_score in zip(output, target):
        out_score = out_score*4 + 1
        tar_score = tar_score*4 + 1
        if tar_score <3 and out_score < 3:
            det_count += 1

    return det_count / len(output)

def evaluate(model, dataloader, device, criterion):
    model.eval()
    loss_heatmap_im, loss_score_im, loss_heatmap_mis, loss_score_mis = 0, 0, 0, 0
    with torch.no_grad():
        sum_heatmap_im, sum_heatmap_mis = 0.0, 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)

            outputs_im = model(inputs['pixel_values'].squeeze(1), inputs['input_ids'][:, 0, :])  # implausibility
            output_heatmap, target_heatmap = outputs_im[0].to(device), targets['artifact_map'].float().to(device)
            output_score, target_score = outputs_im[1].to(device), targets['artifact_score'].float().to(device)
            cur_loss_heatmap_im = criterion(output_heatmap, target_heatmap).item()
            loss_heatmap_im += cur_loss_heatmap_im
            loss_score_im += criterion(output_score, target_score).item()
            sum_heatmap_im += (output_heatmap * 255.0).sum().item()

            if targets['img_name'][0].startswith('finetune'):   # check finetune data loss
                print(f"{targets['img_name']} artifact loss: {cur_loss_heatmap_im}")
        scale_factor = (255 ** 2, 4)
    print(f'Sum of heatmap: {sum_heatmap_im}, {sum_heatmap_mis}')
    return [loss_heatmap_im / len(dataloader) * scale_factor[0], loss_score_im / len(dataloader) * scale_factor[1],
            loss_heatmap_mis / len(dataloader), loss_score_mis / len(dataloader) * scale_factor[1]]


def evaluate_test(model, dataloader, device, criterion, save_root):
    '''
    推理并保存用于计算指标分数的pkl文件
    '''
    model.eval()
    loss_heatmap_im, loss_score_im, loss_heatmap_mis, loss_score_mis, ori_loss_heatmap_im, ori_loss_heatmap_mis = 0, 0, 0, 0, 0, 0
    print(f"Length of dataloader: {len(dataloader)}")
    output_im_scores, output_mis_scores, gt_im_scores, gt_mis_scores = [], [], [], []
    outputs = {}
    preds = {}
    heatmap_threshold = 40
    with torch.no_grad():
        sum_heatmap_im = 0.0
        counter = 0

        for idx,(inputs, targets) in enumerate(dataloader):
            img_name = targets['img_name'][0]
            inputs = inputs.to(device)
            outputs_im = model(inputs['pixel_values'].squeeze(1), inputs['input_ids'][:, 0, :])  # implausibility
            output_heatmap, target_heatmap = outputs_im[0].to(device), targets['artifact_map'].float().to(device)
            output_score, target_score = outputs_im[1].to(device), targets['artifact_score'].float().to(device)
            cur_loss_heatmap_im = criterion(output_heatmap, target_heatmap).item()

            loss_heatmap_im += cur_loss_heatmap_im
            sum_heatmap_im += (output_heatmap * 255.0).sum().item()
            ori_outputs_im_map = torch.round(outputs_im[0] * 255.0).to(device)
            ori_target_im_map = (targets['artifact_map'].float() * 255.0).to(device)

            output_im_scores.append(output_score.item())
            gt_im_scores.append(targets['artifact_score'].float().to(device).item())
            cur_loss_score_im = criterion(outputs_im[1].to(device), targets['artifact_score'].float().to(device)).item()
            loss_score_im += cur_loss_score_im

            # compute loss
            cur_loss_heatmap_im = criterion(outputs_im[0].to(device), targets['artifact_map'].float().to(device)).item()
            loss_heatmap_im += cur_loss_heatmap_im
            
            print(f'Sum of ori_heatmap: {ori_outputs_im_map.sum()}')
            
            outputs[img_name] = [
                ori_outputs_im_map.cpu().numpy() , ori_target_im_map.cpu().numpy()]
            
            # 保存用于计算指标分数的pkl文件
            input_tensor = torch.where(ori_outputs_im_map > heatmap_threshold, 1, 0)
            saved_output_im_map = input_tensor.squeeze(0).cpu().numpy().astype(np.uint8)
            preds[img_name[:-4]] = {
                "score":output_score.item(),
                "pred_area": saved_output_im_map
            }
            target_im_score = targets['artifact_score']
            output_im_score = outputs_im[1].to(device)
            print(f'Counter {counter} implausibity heatmap loss: {cur_loss_heatmap_im}, score&gt: {output_im_score}&{target_im_score}')

    print(f"Aver loss: {loss_heatmap_im / len(dataloader)}, {loss_score_im / len(dataloader)}, \
          {loss_heatmap_mis / len(dataloader)}, {loss_score_mis / len(dataloader)},")
    
    with open(f'{save_root}/baseline_results.pkl', 'wb') as f:
        pickle.dump(preds, f)

    # compute plcc srcc
    get_plcc_srcc(output_im_scores, gt_im_scores)
    return outputs

if  __name__ == '__main__':
    '''** code for dataset **'''
    gpu = "cuda:0"
    pretrained_processor_path = 'altclip_processor'
    pretrained_model_path = 'altclip_model'
    datapath = 'xxx' # save path of datasets
    save_root = 'xxx' # save path of the evaluate results
    load_checkpoint = 'xxx' # save path of the model weight

    val_dataset = RAHFDataset(datapath, 'val', pretrained_processor_path)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2)
    criterion = torch.nn.MSELoss(reduction='mean').to(gpu)

    model = RAHF(pretrained_model_path=pretrained_model_path, freeze=True)
    model.cuda(gpu)
    print(f'Load checkpoint {load_checkpoint}')
    checkpoint = torch.load(f'{load_checkpoint}', map_location=gpu)
    model.load_state_dict(checkpoint['model'])
    outputs = evaluate_test(model=model, dataloader=val_dataloader, device=gpu, criterion=criterion, save_root=save_root)





