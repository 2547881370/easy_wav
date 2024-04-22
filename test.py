import datetime
import os
import shutil
import subprocess
import json
import threading
import time
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram

start_time = time.time()
import sys
print(f"sys.executable：{sys.executable}")
print('\rloading torch       ', end='')
import torch
print('\rloading numpy       ', end='')
import numpy as np
print('\rloading Image       ', end='')
from PIL import Image
print('\rloading argparse    ', end='')
import argparse
print('\rloading math        ', end='')
import math
print('\rloading os          ', end='')
import os
print('\rloading subprocess  ', end='')
import subprocess
print('\rloading pickle      ', end='')
import pickle
print('\rloading cv2         ', end='')
import cv2
print('\rloading audio       ', end='')
import audio
print('\rloading Wav2Lip     ', end='')
from models import Wav2Lip
print('\rRloading RetinaFace ', end='')
from batch_face import RetinaFace
print('\rloading re          ', end='')
import re
print('\rloading partial     ', end='')
from functools import partial
print('\rloading tqdm        ', end='')
from tqdm import tqdm
print('\rloading warnings    ', end='')
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.transforms.functional_tensor')
print('\rloading upscale     ', end='')
from enhance import upscale
print('\rloading load_sr     ', end='')
from enhance import load_sr
print('\rloading load_model  ', end='')
from easy_functions import load_model


args = argparse.Namespace(checkpoint_path='G:\\BaiduNetdiskDownload\\Easy-Wav2Lip-眠-0229\\checkpoints\\Wav2Lip.pth', segmentation_path='checkpoints/face_segmentation.pth', face='G:\\BaiduNetdiskDownload\\Easy-Wav2Lip-眠-0229\\temp\\demo.mp4', audio='G:\\BaiduNetdiskDownload\\Easy-Wav2Lip-眠-0229\\temp\\test1.wav', outfile='G:\\BaiduNetdiskDownload\\Easy-Wav2Lip-眠-0229\\temp\\demo.mp4', static=False, fps=25.0, pads=[0, 10, 20, 0], wav2lip_batch_size=1, out_height=1280, crop=[0, -1, 0, -1], box=[-1, -1, -1, -1], rotate=False, nosmooth='False', no_seg=False, no_sr=False, sr_model='gfpgan', fullres=1, debug_mask='False', preview_settings='False', mouth_tracking='True', mask_dilation=2.5, mask_feathering=5, quality='Fast')

device='cuda'
ffmpeg_path = 'ffmpeg.exe'

with open(os.path.join('checkpoints','predictor.pkl'), 'rb') as f:
    predictor = pickle.load(f)

with open(os.path.join('checkpoints','mouth_detector.pkl'), 'rb') as f:
    mouth_detector = pickle.load(f)
    
    
#creating variables to prevent failing when a face isn't detected
kernel = last_mask = x = y = w = h = None
def Experimental(img, original_img,run_params): 
  global kernel, last_mask, x, y, w, h  # Add last_mask to global variables
  
   # Convert color space from BGR to RGB if necessary 
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  original_img  = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

  if str(args.debug_mask) == 'True':
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

   # Detect face 
  faces = mouth_detector(img) 
  if len(faces) == 0: 
     if last_mask is not None: 
       last_mask = cv2.resize(last_mask, (img.shape[1], img.shape[0]))
       mask = last_mask  # use the last successful mask 
     else: 
       cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img) 
       return img, None 
  else: 
      face = faces[0] 
      shape = predictor(img, face) 
  
      # Get points for mouth 
      mouth_points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)]) 

      # Calculate bounding box dimensions
      x, y, w, h = cv2.boundingRect(mouth_points)

      # Set kernel size as a fraction of bounding box size
      kernel_size = int(max(w, h) * args.mask_dilation)
      upscale_kernel_size = int(max(w, h) * max(args.mask_dilation, 2.5))

      # Create kernels
      kernel = np.ones((kernel_size, kernel_size), np.uint8)
      upscale_kernel = np.ones((upscale_kernel_size, upscale_kernel_size), np.uint8)

      # Create binary mask for mouth 
      mask = np.zeros(img.shape[:2], dtype=np.uint8) 
      cv2.fillConvexPoly(mask, mouth_points, 255)

      last_mask = mask  # Update last_mask with the new mask
  
  # Dilate the mask for upscaling
  upscale_dilated_mask = cv2.dilate(mask, upscale_kernel)
  dilated_mask = cv2.dilate(mask, kernel)

  # Find contours in the dilated mask
  contours, _ = cv2.findContours(upscale_dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Find bounding box coordinates for each contour
  for contour in contours:
    x_dilated, y_dilated, w_dilated, h_dilated = cv2.boundingRect(contour)

    # Crop the image to the bounding box of the dilated mask
    cropped_img = img[y_dilated:y_dilated+h_dilated, x_dilated:x_dilated+w_dilated]

    # Save the cropped image here
    #cv2.imwrite('temp/cp.jpg', cropped_img)

    # Upscale the cropped image
    upscaled_img = upscale(cropped_img, run_params)

    #cv2.imwrite('temp/ucp.jpg', upscaled_img)

    if str(args.debug_mask) == 'True':
      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Paste the upscaled image back onto the original image
    img[y_dilated:y_dilated+h_dilated, x_dilated:x_dilated+w_dilated] = upscaled_img


  # Calculate distance transform of dilated mask
  dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)

  # Normalize distance transform
  cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

  # Convert normalized distance transform to binary mask and convert it to uint8
  _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
  masked_diff = masked_diff.astype(np.uint8)
  
  if not args.mask_feathering == 0:
    blur = args.mask_feathering
    # Set blur size as a fraction of bounding box size
    blur = int(max(w, h) * blur)  # 10% of bounding box size
    if blur % 2 == 0:  # Ensure blur size is odd
      blur += 1
    masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)

  # Convert numpy arrays to PIL Images
  input1 = Image.fromarray(img)
  input2 = Image.fromarray(original_img)

  # Convert mask to single channel where pixel values are from the alpha channel of the current mask
  mask = Image.fromarray(masked_diff)

  # Ensure images are the same size
  assert input1.size == input2.size == mask.size

  # Paste input1 onto input2 using the mask
  input2.paste(input1, (0,0), mask)

  # Convert the final PIL Image back to a numpy array
  input2 = np.array(input2)

  cv2.cvtColor(input2, cv2.COLOR_BGR2RGB, input2)
  
  return input2, mask

def create_tracked_mask(img, original_img): 
  global kernel, last_mask, x, y, w, h  # Add last_mask to global variables
  
   # Convert color space from BGR to RGB if necessary 
  cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img) 
  cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB, original_img) 
  
   # Detect face 
  faces = mouth_detector(img) 
  if len(faces) == 0: 
     if last_mask is not None: 
       last_mask = cv2.resize(last_mask, (img.shape[1], img.shape[0]))
       mask = last_mask  # use the last successful mask 
     else: 
       cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img) 
       return img, None 
  else: 
      face = faces[0] 
      shape = predictor(img, face) 
  
      # Get points for mouth 
      mouth_points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)]) 

      # Calculate bounding box dimensions
      x, y, w, h = cv2.boundingRect(mouth_points)

      # Set kernel size as a fraction of bounding box size
      kernel_size = int(max(w, h) * args.mask_dilation)
      #if kernel_size % 2 == 0:  # Ensure kernel size is odd
        #kernel_size += 1

      # Create kernel
      kernel = np.ones((kernel_size, kernel_size), np.uint8)

      # Create binary mask for mouth 
      mask = np.zeros(img.shape[:2], dtype=np.uint8) 
      cv2.fillConvexPoly(mask, mouth_points, 255)

      last_mask = mask  # Update last_mask with the new mask
  
  # Dilate the mask
  dilated_mask = cv2.dilate(mask, kernel)

  # Calculate distance transform of dilated mask
  dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)

  # Normalize distance transform
  cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

  # Convert normalized distance transform to binary mask and convert it to uint8
  _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
  masked_diff = masked_diff.astype(np.uint8)
  
  #make sure blur is an odd number
  blur = args.mask_feathering
  if blur % 2 == 0:
    blur += 1
  # Set blur size as a fraction of bounding box size
  blur = int(max(w, h) * blur)  # 10% of bounding box size
  if blur % 2 == 0:  # Ensure blur size is odd
    blur += 1
  masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)

  # Convert numpy arrays to PIL Images
  input1 = Image.fromarray(img)
  input2 = Image.fromarray(original_img)

  # Convert mask to single channel where pixel values are from the alpha channel of the current mask
  mask = Image.fromarray(masked_diff)

  # Ensure images are the same size
  assert input1.size == input2.size == mask.size

  # Paste input1 onto input2 using the mask
  input2.paste(input1, (0,0), mask)

  # Convert the final PIL Image back to a numpy array
  input2 = np.array(input2)

  #input2 = cv2.cvtColor(input2, cv2.COLOR_BGR2RGB)
  cv2.cvtColor(input2, cv2.COLOR_BGR2RGB, input2)
  
  return input2, mask

def create_mask(img, original_img): 
  global kernel, last_mask, x, y, w, h  # Add last_mask to global variables
  
   # Convert color space from BGR to RGB if necessary 
  cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img) 
  cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB, original_img)

  if last_mask is not None: 
      last_mask = np.array(last_mask)  # Convert PIL Image to numpy array
      last_mask = cv2.resize(last_mask, (img.shape[1], img.shape[0]))
      mask = last_mask  # use the last successful mask 
      mask = Image.fromarray(mask)

  else:
    # Detect face 
    faces = mouth_detector(img) 
    if len(faces) == 0: 
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img) 
        return img, None 
    else: 
        face = faces[0] 
        shape = predictor(img, face) 
    
        # Get points for mouth 
        mouth_points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)]) 

        # Calculate bounding box dimensions
        x, y, w, h = cv2.boundingRect(mouth_points)

        # Set kernel size as a fraction of bounding box size
        kernel_size = int(max(w, h) * args.mask_dilation)
        #if kernel_size % 2 == 0:  # Ensure kernel size is odd
          #kernel_size += 1

        # Create kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Create binary mask for mouth 
        mask = np.zeros(img.shape[:2], dtype=np.uint8) 
        cv2.fillConvexPoly(mask, mouth_points, 255)

        # Dilate the mask
        dilated_mask = cv2.dilate(mask, kernel)

        # Calculate distance transform of dilated mask
        dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)

        # Normalize distance transform
        cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

        # Convert normalized distance transform to binary mask and convert it to uint8
        _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
        masked_diff = masked_diff.astype(np.uint8)

        if not args.mask_feathering == 0:
          blur = args.mask_feathering
          # Set blur size as a fraction of bounding box size
          blur = int(max(w, h) * blur)  # 10% of bounding box size
          if blur % 2 == 0:  # Ensure blur size is odd
            blur += 1
          masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)

        # Convert mask to single channel where pixel values are from the alpha channel of the current mask
        mask = Image.fromarray(masked_diff)

        last_mask = mask  # Update last_mask with the final mask after dilation and feathering

  # Convert numpy arrays to PIL Images
  input1 = Image.fromarray(img)
  input2 = Image.fromarray(original_img)

  # Resize mask to match image size
  #mask = Image.fromarray(mask)
  mask = mask.resize(input1.size)

  # Ensure images are the same size
  assert input1.size == input2.size == mask.size

  # Paste input1 onto input2 using the mask
  input2.paste(input1, (0,0), mask)

  # Convert the final PIL Image back to a numpy array
  input2 = np.array(input2)

  #input2 = cv2.cvtColor(input2, cv2.COLOR_BGR2RGB)
  cv2.cvtColor(input2, cv2.COLOR_BGR2RGB, input2)
  
  return input2, mask

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images, results_file='last_detected_face.pkl'):
          
    if os.path.exists('results.npy'):
          # 从 .npy 文件中读取数据
          loaded_results_array = np.load('results.npy', allow_pickle=True)

          # 转换为列表
          loaded_results = loaded_results_array.tolist()
          return loaded_results

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    from tqdm import tqdm
    tqdm = partial(tqdm, position=0, leave=True)

    for image, rect in tqdm(zip(images, face_rect(images)), total=len(images), desc="detecting face in every frame", ncols=100):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if str(args.nosmooth) == 'False': boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    # 将 results 转换为 numpy 数组
    results_array = np.array(results, dtype=object)

    # 保存为 .npy 文件
    np.save('results.npy', results_array)

    return results_array

def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    print("\r" + " " * 100, end="\r")
    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda'

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint
model = detector = detector_model = None
def do_load(checkpoint_path):
    global model, detector, detector_model
    model = load_model(checkpoint_path)
    detector = RetinaFace(gpu_id=0, model_path="checkpoints/mobilenet.pth", network="mobilenet")
    detector_model = detector.model

def face_rect(images):
  face_batch_size = 8
  num_batches = math.ceil(len(images) / face_batch_size)
  prev_ret = None
  for i in range(num_batches):
      batch = images[i * face_batch_size: (i + 1) * face_batch_size]
      all_faces = detector(batch)  # return faces list of all images
      for faces in all_faces:
          if faces:
              box, landmarks, score = faces[0]
              prev_ret = tuple(map(int, box))
          yield prev_ret

def main():
    start_time = time.time()
    args.img_size = 96
    
    full_frames = []
    fps = 25
    
    if os.path.exists('full_frames.npy'):
          # 从 .npy 文件中读取数据
          loaded_results_array = np.load('full_frames.npy')

          # 转换为列表
          full_frames = loaded_results_array
    else :
          video_stream = cv2.VideoCapture(args.face)
          fps = video_stream.get(cv2.CAP_PROP_FPS)

          full_frames = []
          while 1:
              still_reading, frame = video_stream.read()
              if not still_reading:
                  video_stream.release()
                  break

              if args.fullres != 1:
                  aspect_ratio = frame.shape[1] / frame.shape[0]
                  frame = cv2.resize(frame, (int(args.out_height * aspect_ratio), args.out_height))
      
              if args.rotate:
                  frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

              y1, y2, x1, x2 = args.crop
              if x2 == -1: x2 = frame.shape[1]
              if y2 == -1: y2 = frame.shape[0]

              frame = frame[y1:y2, x1:x2]

              full_frames.append(frame) 
              
          # 保存为 .npy 文件
          np.save('full_frames.npy', full_frames)


    # if not args.audio.endswith('.wav'):
    #     print('Converting audio to .wav')
    #     subprocess.check_call([
    #           f"{ffmpeg_path}", "-y", "-loglevel", "error",
    #           "-i", args.audio,
    #           "temp/temp.wav",
    #       ])
    #     args.audio = 'temp/temp.wav'
        
    # print('analysing audio...')
    
    # if not torch.cuda.is_available():
    #     raise RuntimeError("CUDA is not available. Please check your installation.")

    # # 设置设备为第一个可用的GPU
    # device = torch.device("cuda:0")

    # # 加载音频文件
    # wav, sr = torchaudio.load(args.audio, normalize=True)
    # # 确保采样率为16000Hz
    # if sr != 16000:
    #     resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    #     wav = resampler(wav)
    # wav = wav.to(device)

    # # 创建梅尔频谱图变换器
    # mel_transformer = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=16000,
    #     n_fft=800,
    #     hop_length=200,
    #     f_min=55,
    #     f_max=7600,
    #     n_mels=80,
    # ).to(device)

    # # 计算梅尔频谱图
    # mel = mel_transformer(wav)

    # # 将结果从CUDA设备移动到CPU
    # # 调整输出形状
    # mel = mel.squeeze().cpu().numpy()
    
    
    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    
    print(mel.shape)
    
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
    
    mel_chunks = []

    mel_idx_multiplier = 80./fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1;
    
    full_frames = full_frames[:len(mel_chunks)]
    if str(args.preview_settings) == 'True':
      full_frames = [full_frames[0]]
      mel_chunks = [mel_chunks[0]]
    print (str(len(full_frames))+' frames to process')
    batch_size = args.wav2lip_batch_size
    
    print(f"代码块执行时间为: {time.time()  - start_time} 秒")
    
    
    start_time = time.time()
    
    if str(args.preview_settings) == 'True':
      gen = datagen(full_frames, mel_chunks)
    else:
      gen = datagen(full_frames.copy(), mel_chunks)
      
    print(f"代码块执行时间为: {time.time()  - start_time} 秒")
    start_time = time.time()

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(
    gen,
    total=int(np.ceil(float(len(mel_chunks))/batch_size)),
    desc="Processing Wav2Lip",ncols=100
)):
        if i == 0:

          if not args.quality=='Fast':
            print(f"mask size: {args.mask_dilation}, feathering: {args.mask_feathering}")  
            if not args.quality=='Improved':   
              print("Loading", args.sr_model)
              run_params = load_sr()

        #   print("Starting...")
          frame_h, frame_w = full_frames[0].shape[:-1]
          fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
          out = cv2.VideoWriter('temp/result.mp4', fourcc, fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to('cuda')
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to('cuda')

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            #cv2.imwrite('temp/f.jpg', f)
            
            y1, y2, x1, x2 = c

            if str(args.debug_mask) == 'True' and args.quality != "Experimental": #makes the background black & white so you can see the mask better
              f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
              f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            of=f
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            cf = f[y1:y2, x1:x2]
            
            f[y1:y2, x1:x2] = p
            # cv2.imwrite('temp/p.jpg', f)
            out.write(f)
              
            # 显示每一帧图像
            # cv2.imshow('Processed Frame', f)
            # cv2.waitKey(1)  # 等待1毫秒以确保图像显示在窗口中

    out.release()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"代码块执行时间为: {execution_time} 秒")
    # cv2.destroyAllWindows()  # 在结束时关闭OpenCV窗口
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    subprocess.check_call([
        "ffmpeg.exe", "-y", "-loglevel", "error",
        "-i", "temp/result.mp4",
        "-i", args.audio,
        "-c:v", "h264_nvenc",
        f'out/result_{timestamp}_.mp4' ,
      ])

if __name__ == '__main__':
    do_load(args.checkpoint_path)
    input("按下回车键执行 main() 方法：")
    thread = threading.Thread(target=main)
    
    # 启动线程
    thread.start()
    
    # main()
    while True:
        print('')
        time.sleep(10)


