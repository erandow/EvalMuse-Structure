
import random
import time
import io
import torch
import pickle
from transformers import AutoProcessor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageOps
from skimage.transform import resize
import numpy as np

random.seed(time.time())

def add_jpeg_noise(img):
  # Randomly add JPEG noise with quality between 70 and 100
  quality = random.randint(70, 100)
  buffer = io.BytesIO()
  img.save(buffer, format='JPEG', quality=quality)
  noisy_img = Image.open(io.BytesIO(buffer.getvalue()))
  return noisy_img


class RAHFDataset(Dataset):
  def __init__(self, datapath, data_type, pretrained_processor_path, finetune=False, img_len=448):
    self.img_len = img_len
    # self.tag_word = ['human artifact', 'human mask']  # 使用siglip时需要
    self.tag_word = ['human artifact', 'human segmentation']  # 使用AltCLIP时需要
    self.finetune = finetune
    self.processor = AutoProcessor.from_pretrained(pretrained_processor_path)
    self.processor.image_processor.do_resize = False
    self.processor.image_processor.do_center_crop = False   # 保持图片原大小
    self.to_tensor = transforms.ToTensor()
    self.datapath = datapath
    self.data_type = data_type
    # 加载pkl文件
    self.data_info = self.load_info()
    self.images = []
    self.prompts_en = []
    self.prompts_cn = []
    self.heatmaps = []
    self.scores = []
    self.img_name = list(self.data_info.keys())
    for i in range(len(self.img_name)):
      cur_img = self.img_name[i]
      img = Image.open(f"{self.datapath}/{self.data_type}/images/{cur_img}")
      self.images.append(img.resize((self.img_len, self.img_len), Image.LANCZOS))
      # prompt_cn = self.data_info[cur_img]['prompt_cn']
      prompt_en = self.data_info[cur_img]['prompt']
      self.prompts_en.append(prompt_en)
      # self.prompts_cn.append(prompt_cn)

      artifact_map = self.data_info[cur_img]['heat_map'].astype(float)
      artifact_map = artifact_map/255.0  # 热力图归一化到0-1

      # misalignment_map = self.data_info[cur_img]['human_mask'].astype(float)  # 使用0-1二值的人体mask
      misalignment_map = np.zeros((512,512))
      self.heatmaps.append([artifact_map, misalignment_map])

      norm_score = (self.data_info[cur_img]['score'] - 1.0)/4.0
      self.scores.append((norm_score, 0))   # 人体mask没有分数
      if i % 1000 == 0:
        print(f"Processed {i} images.")

    if data_type == 'train' and self.finetune:
      self.finetune_info = self.load_info(specific_name='finetune')
      self.finetune_images = []
      self.finetune_prompts = []
      self.finetune_heatmaps = []
      self.finetune_scores = []
      self.finetune_img_names = list(self.finetune_info.keys())
      for i in range(len(self.finetune_img_names)):
        cur_img = self.finetune_img_names[i]
        img = Image.open(f"{self.datapath}/{self.data_type}/images/{cur_img}")
        self.finetune_images.append(img.resize((self.img_len, self.img_len), Image.LANCZOS))
        self.finetune_prompts.append(self.finetune_info[cur_img]['prompt'])
        artifact_map = self.finetune_info[cur_img]['artifact_map'].astype(float)
        misalignment_map = self.finetune_info[cur_img]['misalignment_map'].astype(float)
        self.finetune_heatmaps.append([artifact_map, misalignment_map])
        self.finetune_scores.append((self.finetune_info[cur_img]['artifact_score'], self.finetune_info[cur_img]['misalignment_score']))
        if i % 1000 == 0:
          print(f"Processed {i} finetuning images.")

  def __len__(self):
    return len(self.img_name)

  def __getitem__(self, idx):
    if self.data_type == 'train' and self.finetune and random.random() < 0.5:   # choose finetune image to train with probability of 0.1
      finetune_idx = idx % len(self.finetune_img_names)
      finetune_img_name = self.finetune_img_names[finetune_idx]
      input_img = self.finetune_images[finetune_idx]
      input_prompt = self.finetune_prompts[finetune_idx]
      target_heatmaps = self.finetune_heatmaps[finetune_idx]
      input_img, target_heatmaps, img_pos = self.finetune_augment(input_img, target_heatmaps)
      cur_input = self.processor(images=input_img, text=[f"{self.tag_word[0]} {input_prompt}", f"{self.tag_word[1]} {input_prompt}"],
                            padding="max_length", return_tensors="pt", truncation=True)

      cur_target = {}
      cur_target['artifact_map'] = (self.to_tensor(target_heatmaps[0]))
      cur_target['misalignment_map'] = (self.to_tensor(target_heatmaps[1]))
      cur_target['artifact_score'] = self.finetune_scores[finetune_idx][0]
      cur_target['misalignment_score'] = self.finetune_scores[finetune_idx][1]
      cur_target['img_name'] = finetune_img_name
      cur_target['img_pos'] = torch.tensor(img_pos)
      return cur_input, cur_target

    else:
      img_name = self.img_name[idx]
      input_img = self.images[idx]
      input_prompt = self.prompts_en[idx]
      target_heatmaps = self.heatmaps[idx]
      if self.data_type == 'train':
          input_img, target_heatmaps = self.data_augment(input_img, target_heatmaps)
      cur_input = self.processor(images=input_img, text=[f"{self.tag_word[0]} {input_prompt}", f"{self.tag_word[1]} {input_prompt}"],
                            padding="max_length", return_tensors="pt", truncation=True)
      cur_target = {}
      cur_target['artifact_map'] = (self.to_tensor(target_heatmaps[0]))
      cur_target['misalignment_map'] = (self.to_tensor(target_heatmaps[1]))
      cur_target['artifact_score'] = self.scores[idx][0]
      cur_target['misalignment_score'] = self.scores[idx][1]
      cur_target['img_name'] = img_name
      cur_target['img_pos'] = torch.tensor((0,0,self.img_len))
      return cur_input, cur_target

  def load_info(self, specific_name=None):
      if specific_name:
          print(f'Loading {specific_name} data info...')
          data_info = pickle.load(open(f'{self.datapath}/{specific_name}_info.pkl', 'rb'))
      else:
          print(f'Loading {self.data_type} data info...')
          data_info = pickle.load(open(f'{self.datapath}/{self.data_type}_info.pkl', 'rb'))
      return data_info

  def data_augment(self, img, heatmaps):

      if random.random() < 0.5: # 50% chance to crop
          crop_size = int(img.height * random.uniform(0.8, 1.0)), int(img.width * random.uniform(0.8, 1.0))
          crop_region = transforms.RandomCrop.get_params(img, crop_size)
          img = transforms.functional.crop(img, crop_region[0], crop_region[1], crop_region[2], crop_region[3])
          heatmaps = [resize(heatmap, (self.img_len, self.img_len), mode='reflect', anti_aliasing=True)
                             for heatmap in heatmaps]
          heatmaps = [heatmap[crop_region[0]:crop_region[0]+crop_region[2],
                              crop_region[1]:crop_region[1]+crop_region[3]]
                              for heatmap in heatmaps]
          img = img.resize((self.img_len, self.img_len), Image.LANCZOS)
          heatmaps = [resize(heatmap, (512, 512), mode='reflect', anti_aliasing=True)
                             for heatmap in heatmaps]
      data_transforms = transforms.Compose([
          transforms.RandomApply([
              transforms.ColorJitter(brightness=0.05, contrast=(0.8, 1), hue=0.025, saturation=(0.8, 1)),
              add_jpeg_noise
          ], p=0.1),
          transforms.RandomApply([transforms.Grayscale(3)], p=0.1)
      ])

      img = data_transforms(img)
      return img, heatmaps

  def finetune_augment(self, img, heatmaps):

    data_transforms = transforms.Compose([
      transforms.RandomApply([
        transforms.ColorJitter(brightness=0.05, contrast=(0.8, 1), hue=0.025, saturation=(0.8, 1)),
        add_jpeg_noise
      ], p=0.2),
      transforms.RandomApply([transforms.Grayscale(3)], p=0.2)
    ])
    img = data_transforms(img)
    # rescale gt, do nothing to heatmaps
    if random.random() < 0.9:   # very small image
      scale = random.uniform(0.2, 0.5)
    else:
      scale = random.uniform(0.5, 1.0)
    new_len = int(scale * self.img_len)
    small_img = img.resize((new_len, new_len), Image.LANCZOS)
    top_left_x, top_left_y = random.randint(0, self.img_len-new_len), random.randint(0, self.img_len-new_len)
    pad_left, pad_top = top_left_x, top_left_y
    pad_right, pad_bottom = self.img_len - new_len - pad_left, self.img_len - new_len - pad_top
    pad_color = (255, 255, 255)   # white padding
    pad_img = ImageOps.expand(small_img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=pad_color)
    return pad_img, heatmaps, (top_left_x, top_left_y, new_len)
  