import torch
import torchvision.transforms as T

from PIL import Image

def env_to_img(arr):
  to_img = T.ToPILImage()
  arr = torch.from_numpy(arr).permute(2,0,1)
  img = to_img(arr)
  return img

def save_replay(loc, name, buf, dur):
  buf = list(map(env_to_img, buf))
  buf[0].save(f'{loc}/{name}.gif', format='GIF', append_images=buf,
          save_all=True, duration=dur, loop=0)
