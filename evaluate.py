import torch
from model.model_final import RAHF
from PIL import Image
import numpy as np
from scipy.stats import spearmanr
from dataset import RAHFDataset
from torch.utils.data import DataLoader, Dataset
import pickle
import os
import json
from transformers import AutoProcessor

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


if  __name__ == '__main__':

    '''推理并保存计算分数的pkl文件'''
    gpu = "cuda:0"
    pretrained_processor_path = 'altclip_processor'
    pretrained_model_path = 'altclip_model'
    save_root = 'xxx' # save path of the evaluate results
    load_checkpoint = 'xxx' # data path of the model weight

    val_info_path = 'val_info.json' # data path of the val anno info
    val_info = json.load(open(val_info_path, 'r'))
    name2prompt = {k:v['prompt_en'] for k,v in val_info.items()}

    img_root = 'xxx' # data path of the val images
    img_files = os.listdir(img_root)
    
    gpu = "cuda:0"
    print(f'Load checkpoint {load_checkpoint}')
    checkpoint = torch.load(f'{load_checkpoint}', map_location='cpu')
    model = RAHF(pretrained_model_path=pretrained_model_path,freeze=True)
    model.load_state_dict(checkpoint['model'])
    model.cuda(gpu)
    model.eval()
    processor = AutoProcessor.from_pretrained(pretrained_processor_path)
    tag_word = ['human artifact', 'human segmentation']

    with torch.no_grad():
        preds = {}
        for img_file in img_files:
            img_name = img_file.split('.')[0]
            prompt = name2prompt[img_name]
            img_path = f'{img_root}/{img_file}'
            img = Image.open(img_path)
            image = img.resize((448, 448), Image.LANCZOS)

            cur_input = processor(images=image, text=[f"{tag_word[0]} {prompt}", f"{tag_word[1]} {prompt}"],
                    padding="max_length", return_tensors="pt", truncation=True)
            inputs_pixel_values, inputs_ids_im = cur_input['pixel_values'].to(gpu), cur_input['input_ids'][0, :].unsqueeze(0).to(gpu)
            
            heatmap, score = model(inputs_pixel_values, inputs_ids_im, need_score=True)
            print(f'heatmap: {heatmap.shape}, score: {score}')
            
            ori_heatmap = torch.round(heatmap * 255.0)
            heatmap_treshold = 40
            input_tensor = torch.where(ori_heatmap > heatmap_treshold, 1, 0)
            saved_output_im_map = input_tensor.squeeze(0).cpu().numpy().astype(np.uint8)
            preds[img_name[:-4]] = {
                "score":score.item(),
                "pred_area": saved_output_im_map
            }

    with open(f'{save_root}/baseline_results.pkl', 'wb') as f:
        pickle.dump(preds, f)








