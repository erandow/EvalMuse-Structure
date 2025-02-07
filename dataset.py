import copy
import random
import time
import io
import torch
import pickle
from transformers import AutoProcessor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from skimage.transform import resize
import json
import os
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
      # img = Image.open(f"{self.datapath}/images/{cur_img}")
      img = Image.open(f"{self.datapath}/{self.data_type}/images/{cur_img}")
      self.images.append(img.resize((self.img_len, self.img_len), Image.LANCZOS))
      # prompt_cn = self.data_info[cur_img]['prompt_cn']
      prompt_en = self.data_info[cur_img]['prompt']
      # 超出文字编码器max_len，重置prompt为'human'
      # 标识符human artifact与human mask的token长度相同

      # 使用siglip时需要
      # cur_text_token = self.processor(text=[f"{self.tag_word[0]} {input_prompt}", f"{self.tag_word[1]} {input_prompt}"], 
      #                       padding="max_length", return_tensors="pt", truncation=False)
      # if cur_text_token['input_ids'].shape[1] > 64:  

      self.prompts_en.append(prompt_en)
      # self.prompts_cn.append(prompt_cn)

      # artifact_map = self.data_info[cur_img]['artifact_map'].astype(float)
      artifact_map = self.data_info[cur_img]['heat_map'].astype(float)
      artifact_map = artifact_map/255.0  # 归一化到0-1

      # misalignment_map = self.data_info[cur_img]['human_mask'].astype(float)  # 使用人体01mask代替图文匹配热力图
      misalignment_map = np.zeros((512,512))
      self.heatmaps.append([artifact_map, misalignment_map])
      # self.scores.append((self.data_info[cur_img]['artifact_score'], self.data_info[cur_img]['misalignment_score']))
      # self.scores.append((self.data_info[cur_img]['artifact_score'], None))  
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
      # target_heatmaps = [heatmap / 255.0 for heatmap in target_heatmaps]
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
      # if random.random() < 1/3: # 33%的概率使用英文prompt
      #   input_prompt = self.prompts_en[idx]
      # else:
      #   input_prompt = self.prompts_cn[idx]
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



class RAHFDataset_old(Dataset):
  def __init__(self, datapath, data_type, siglip_processor_path, finetune=False):
    self.img_W = 512
    self.img_H = 512
    self.finetune = finetune
    self.processor = AutoProcessor.from_pretrained(siglip_processor_path)

    self.to_tensor = transforms.ToTensor()
    self.datapath = datapath
    self.data_type = data_type

    self.data_info = self.load_info(specific_name='checked')
    # self.data_info = self.load_info()
    self.images = []
    self.prompts = []
    self.heatmaps = []
    self.scores = []
    self.img_name = []
    self.imgs = os.listdir(f"{self.datapath}/{self.data_type}")

    for i in range(len(self.imgs)):
      cur_img = self.imgs[i]
      if cur_img == '4f9177be654d11efb45852fe1602b694.png':continue
      img = Image.open(f"{self.datapath}/{self.data_type}/{cur_img}")
      self.images.append(img.resize((512, 512), Image.LANCZOS))
      # 临时改为448
      # self.images.append(img.resize((448, 448), Image.LANCZOS))
      # self.prompts.append(self.data_info[cur_img]['prompt'])
      self.prompts.append('human')
      self.img_name.append(cur_img)

      # 不带热力图则注释掉
      artifact_map = self.data_info[cur_img]['heat_map']
      # 临时resize, 根据图像尺寸修改 
      # artifact_map = artifact_map.astype(np.uint8)
      # artifact_image = Image.fromarray(artifact_map)
      # artifact_resized = artifact_image.resize((224, 224), Image.BOX)

      # 添加语义分割分支
      sematic_map = self.data_info[cur_img]['human_mask']
      sematic_map[sematic_map==255] = 1

      # 临时修改size
      # sematic_image = Image.fromarray(sematic_map)
      # sematic_resized = sematic_image.resize((224, 224), Image.BOX)
      # sematic_map = np.array(sematic_map)

      self.heatmaps.append([artifact_map, sematic_map])
      score = (self.data_info[cur_img]['score'] - 1) / 4
      self.scores.append(score)

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
        img = Image.open(f"{self.datapath}/images/{cur_img}")
        self.finetune_images.append(img.resize((512, 512), Image.LANCZOS))
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
      input_img, img_pos = self.finetune_augment(input_img)
      cur_input = self.processor(images=input_img, text=[f"implausibility {input_prompt}", f"misalignment {input_prompt}"],
                            padding="max_length", return_tensors="pt", truncation=True)

      target_heatmap, sematic_map = target_heatmaps
      target_heatmap = target_heatmap / 255.0
      cur_target = {}
      cur_target['artifact_map'] = (self.to_tensor(target_heatmap))
      cur_target['artifact_score'] = self.finetune_scores[finetune_idx]
      cur_target['img_name'] = finetune_img_name
      cur_target['img_pos'] = torch.tensor(img_pos)
      return cur_input, cur_target

    else:
      img_name = self.img_name[idx]
      input_img = self.images[idx]
      input_prompt = self.prompts[idx]
      target_heatmaps = self.heatmaps[idx]
      
      if self.data_type == 'train':
        input_img, target_heatmaps = self.data_augment(input_img,target_heatmaps)
        
      cur_input = self.processor(images=input_img, text=[f"implausibility {input_prompt}", f"misalignment {input_prompt}"],
                            padding="max_length", return_tensors="pt", truncation=True)

      target_heatmap, sematic_map = target_heatmaps
      # heatmap转换到[0,1]
      target_heatmap = target_heatmap / 255.0
      cur_target = {}
      cur_target['artifact_map'] = (self.to_tensor(target_heatmap))
      sematic_map = self.to_tensor(sematic_map).squeeze(0)
      cur_target['sematic_map'] = (sematic_map)
      cur_target['artifact_score'] = self.scores[idx]
      cur_target['img_name'] = img_name
      cur_target['img_pos'] = torch.tensor((0,0,512))
      return cur_input, cur_target

  def load_info(self, specific_name=None):
      if specific_name:
          print(f'Loading {specific_name} data info...')
          data_info = pickle.load(open(f'{self.datapath}/{self.data_type}_info_{specific_name}.pkl', 'rb'))
      else:
          print(f'Loading {self.data_type} data info...')
          data_info = pickle.load(open(f'{self.datapath}/{self.data_type}_info.pkl', 'rb'))

      print(len(data_info))
      return data_info

  def data_augment(self, img, heatmaps=None):

      if random.random() < 0.5: # 50% chance to crop
          crop_size = int(img.height * random.uniform(0.8, 1.0)), int(img.width * random.uniform(0.8, 1.0))
          crop_region = transforms.RandomCrop.get_params(img, crop_size)
          img = transforms.functional.crop(img, crop_region[0], crop_region[1], crop_region[2], crop_region[3])
          ##  gt heatmap resize
          if heatmaps is not None:
            heatmaps = [heatmap[crop_region[0]:crop_region[0]+crop_region[2],
                                crop_region[1]:crop_region[1]+crop_region[3]]
                                for heatmap in heatmaps]
            heatmaps = [resize(heatmap, (512, 512), mode='reflect', anti_aliasing=True)
                      for heatmap in heatmaps]
          # 修改size
          img = img.resize((512, 512), Image.LANCZOS)
          # img = img.resize((448, 448), Image.LANCZOS)

      data_transforms = transforms.Compose([
          transforms.RandomApply([
              transforms.ColorJitter(brightness=0.05, contrast=(0.8, 1), hue=0.025, saturation=(0.8, 1)),
              add_jpeg_noise
          ], p=0.1),
          transforms.RandomApply([transforms.Grayscale(3)], p=0.1)
      ])

      img = data_transforms(img)
      return img, heatmaps

  def finetune_augment(self, img):

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
    new_len = int(scale * 512)
    small_img = img.resize((new_len, new_len), Image.LANCZOS)
    top_left_x, top_left_y = random.randint(0, 512-new_len), random.randint(0, 512-new_len)
    pad_left, pad_top = top_left_x, top_left_y
    pad_right, pad_bottom = 512 - new_len - pad_left, 512 - new_len - pad_top
    pad_color = (255, 255, 255)   # white padding
    pad_img = ImageOps.expand(small_img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=pad_color)
    return pad_img, (top_left_x, top_left_y, new_len)


class RAHFDataset_Score(Dataset):
  def __init__(self, datapath, data_type, siglip_processor_path, finetune=False):
    self.img_W = 512
    self.img_H = 512
    self.finetune = finetune
    self.processor = AutoProcessor.from_pretrained(siglip_processor_path)
    self.to_tensor = transforms.ToTensor()
    self.datapath = datapath
    self.data_type = data_type

    self.data_info = self.load_info(specific_name='data_score')
    self.images = []
    self.prompts = []
    self.heatmaps = []
    self.scores = []
    self.img_name = []
    self.imgs = os.listdir(f"{self.datapath}/{self.data_type}")
    for i in range(len(self.imgs)):
      cur_img = self.imgs[i]
      img = Image.open(f"{self.datapath}/score_images/{self.data_type}/{cur_img}")
      self.images.append(img.resize((512, 512), Image.LANCZOS))
      self.prompts.append(self.data_info[cur_img]['prompt'])
      self.img_name.append(cur_img)
      # 不带热力图则注释掉
      # artifact_map = self.data_info[cur_img]['artifact_map'].astype(float)
      # misalignment_map = self.data_info[cur_img]['misalignment_map'].astype(float)
      # self.heatmaps.append([artifact_map, misalignment_map])
      score = (self.data_info[cur_img]['score'] - 1) / 4
      self.scores.append(score)

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
        img = Image.open(f"{self.datapath}/images/{cur_img}")
        self.finetune_images.append(img.resize((512, 512), Image.LANCZOS))
        self.finetune_prompts.append(self.finetune_info[cur_img]['prompt'])
        # artifact_map = self.finetune_info[cur_img]['artifact_map'].astype(float)
        # misalignment_map = self.finetune_info[cur_img]['misalignment_map'].astype(float)
        # self.finetune_heatmaps.append([artifact_map, misalignment_map])
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
      # target_heatmaps = self.finetune_heatmaps[finetune_idx]
      input_img, img_pos = self.finetune_augment(input_img)
      cur_input = self.processor(images=input_img, text=[f"implausibility {input_prompt}", f"misalignment {input_prompt}"],
                            padding="max_length", return_tensors="pt", truncation=True)
      # target_heatmaps = [heatmap / 255.0 for heatmap in target_heatmaps]
      cur_target = {}
      # cur_target['artifact_map'] = (self.to_tensor(target_heatmaps[0]))
      # cur_target['misalignment_map'] = (self.to_tensor(target_heatmaps[1]))
      cur_target['artifact_score'] = self.finetune_scores[finetune_idx]
      # cur_target['misalignment_score'] = self.finetune_scores[finetune_idx][1]
      cur_target['img_name'] = finetune_img_name
      cur_target['img_pos'] = torch.tensor(img_pos)
      return cur_input, cur_target

    else:
      img_name = self.img_name[idx]
      input_img = self.images[idx]
      input_prompt = self.prompts[idx]
      # target_heatmaps = self.heatmaps[idx]
      if self.data_type == 'train':
          input_img, target_heatmaps = self.data_augment(input_img)
      cur_input = self.processor(images=input_img, text=[f"implausibility {input_prompt}", f"misalignment {input_prompt}"],
                            padding="max_length", return_tensors="pt", truncation=True)
      # target_heatmaps = [heatmap / 255.0 for heatmap in target_heatmaps]
      cur_target = {}
      # cur_target['artifact_map'] = (self.to_tensor(target_heatmaps[0]))
      # cur_target['misalignment_map'] = (self.to_tensor(target_heatmaps[1]))
      cur_target['artifact_score'] = self.scores[idx]
      # cur_target['misalignment_score'] = self.scores[idx][1]
      cur_target['img_name'] = img_name
      cur_target['img_pos'] = torch.tensor((0,0,512))
      return cur_input, cur_target

  def load_info(self, specific_name=None):
      # if specific_name:
      #     print(f'Loading {specific_name} data info...')
      #     data_info = pickle.load(open(f'{self.datapath}/{specific_name}_info.pkl', 'rb'))
      # else:
      #     print(f'Loading {self.data_type} data info...')
      #     data_info = pickle.load(open(f'{self.datapath}/{self.data_type}_info.pkl', 'rb'))
      json_path = f'{specific_name}.json'
      with open(json_path, 'r', encoding='utf-8') as f:
          data_info = json.load(f)
      return data_info

  def data_augment(self, img, heatmaps=None):

      if random.random() < 0.5: # 50% chance to crop
          crop_size = int(img.height * random.uniform(0.8, 1.0)), int(img.width * random.uniform(0.8, 1.0))
          crop_region = transforms.RandomCrop.get_params(img, crop_size)
          img = transforms.functional.crop(img, crop_region[0], crop_region[1], crop_region[2], crop_region[3])
          if heatmaps is not None:
            heatmaps = [heatmap[crop_region[0]:crop_region[0]+crop_region[2],
                                crop_region[1]:crop_region[1]+crop_region[3]]
                                for heatmap in heatmaps]
            heatmaps = [resize(heatmap, (512, 512), mode='reflect', anti_aliasing=True)
                      for heatmap in heatmaps]
          img = img.resize((512, 512), Image.LANCZOS)

      data_transforms = transforms.Compose([
          transforms.RandomApply([
              transforms.ColorJitter(brightness=0.05, contrast=(0.8, 1), hue=0.025, saturation=(0.8, 1)),
              add_jpeg_noise
          ], p=0.1),
          transforms.RandomApply([transforms.Grayscale(3)], p=0.1)
      ])

      img = data_transforms(img)
      return img, heatmaps

  def finetune_augment(self, img):

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
    new_len = int(scale * 512)
    small_img = img.resize((new_len, new_len), Image.LANCZOS)
    top_left_x, top_left_y = random.randint(0, 512-new_len), random.randint(0, 512-new_len)
    pad_left, pad_top = top_left_x, top_left_y
    pad_right, pad_bottom = 512 - new_len - pad_left, 512 - new_len - pad_top
    pad_color = (255, 255, 255)   # white padding
    pad_img = ImageOps.expand(small_img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=pad_color)
    return pad_img, (top_left_x, top_left_y, new_len)


if __name__ == '__main__':
  datapath = '/mnt/bn/rahf/mlx/users/jincheng.liang/repo/12094/RAHF/data/human'
  siglip_processor_path = '/mnt/bn/rahf/mlx/users/kouhongwei/repo/13650/RAHF/siglip_processor'
  train_dataset = RAHFDataset(datapath, 'val', siglip_processor_path)
  train_dataloader = DataLoader(dataset=train_dataset, 
                                batch_size=2, 
                                shuffle=False, 
                                num_workers=1,
                                pin_memory=True,)
  for batch_id, (inputs, targets) in enumerate(train_dataloader):
    # print('sem:',targets['sematic_map'])
    # print('heat',targets['artifact_map'])
    print(targets['sematic_map'].shape)
    print(targets['artifact_map'].shape)
    break