import pickle
import os
import json
import cv2
import numpy as np
'''
Process the labeled data into the baseline training format train_info.pkl
Note: The competition does not provide a validation set. If necessary, please divide it yourself and save it as val_info.pkl in the same path
'''

def overlay_heatmap_opencv(mask, image):
    heatmap = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
    overlayed_image = cv2.addWeighted(image, 0.3, heatmap, 0.7, 0)
    return overlayed_image

train_path = 'xxx' # Training set image path
vis_path = 'xxx' # Heatmap visualization save path
save_path = 'xxx' # Training data save path
info_path = 'train_info.json'# Training set annotation data
with open(info_path, 'r') as f:
    info = json.load(f)
os.makedirs(save_path, exist_ok=True)

vis = False
if vis:
    os.makedirs(vis_path, exist_ok=True)
data_info = {}
for k,v in info.items():
    try:
        cur_info = {}
        img_path = os.path.join(train_path, k+'.jpg')
        image = cv2.imread(img_path)
        height, width, _ = image.shape

        prompt = v['prompt_en']
        score = v['mos']
        bbox_info = v['bbox_info']
        part_mask = np.zeros([height, width])
        bbox_infos = []

        for person in bbox_info:
            cur_mask =  np.zeros([height, width])
            if not person:
                continue
            else:
                for bbox in person:
                    if bbox['bbox_type'] == 1:
                        top_left, bottom_right = bbox['bbox']
                        cur_mask[top_left['y']:bottom_right['y'], top_left['x']:bottom_right['x']] = 1
                    elif bbox['bbox_type'] == 2:
                        points = bbox['bbox']
                        cv2.fillPoly(cur_mask, [np.array(points)], 1)
                    else:
                        import pdb;pdb.set_trace()
            part_mask = part_mask + cur_mask

        error_mask = part_mask.astype(np.uint8)
        error_mask[error_mask==1] = 64
        error_mask[error_mask==2] = 127
        error_mask[error_mask==3] = 180     
        error_mask[error_mask==4] = 255

        # resize for baseline model
        resized_heatmap = cv2.resize(error_mask, (512, 512))
        cur_info['heat_map'] = resized_heatmap
        cur_info['prompt'] = prompt
        cur_info['score'] = score
        data_info[k+'.jpg'] = cur_info
        if vis:
            heatmap = overlay_heatmap_opencv(error_mask, image)
            mask_path = os.path.join(vis_path, k+'.jpg')
            cv2.imwrite(mask_path, heatmap)

    except Exception as e:
        print(k,e)

print(len(data_info))
with open(os.path.join(save_path, 'train_info.pkl'), 'wb') as f:
    pickle.dump(data_info, f)


