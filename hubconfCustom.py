'''
A modified version of hubconf.py  

Modifications:
1. Added a function to detect PPE violation in a video file or video stream
2. Added a function to send email alert with attached image

Modifications made by Anubhav Patrick
Date: 04/02/2023
'''

import threading
import time
import cv2
import torch
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

#send mail alert
from send_mail import prepare_and_send_email

#Global Variables
is_email_allowed = False #when user checks the email checkbox, this variable will be set to True
send_next_email = True #We have to wait for 10 minutes before sending another email
# NEXT TWO STATEMENTS NEED TO BE CHANGED TO MATCH YOUR SETUP!!!
#set the default email sender and recipient 
email_sender = 'support.ai@giindia.com'
email_recipient = 'support.ai@giindia.com'
# detections_summary will be used to store the detections summary report
detections_summary = ''

classes_to_filter = None  #You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person' ]

# a dictionary to store options for inference
opt  = {
    "weights": "best.pt", # Path to weights file default weights are for nano model
    "yaml"   : "data/custom_data.yaml",
    "img-size": 640, # default image size
    "conf-thres": 0.25, # confidence threshold for inference.
    "iou-thres" : 0.45, # NMS IoU threshold for inference.
    "device" : 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : classes_to_filter  # list of classes to filter or None
}


def violation_alert_generator(im0, subject='PPE Violation Detected', message_text='A PPE violation is detected at ABESIT'):
    '''This function will send an email with attached alert image and then wait for 10 minutes before sending another email
    
    Parameters:
      im0 (numpy.ndarray): The image to be attached in the email
      subject (str): The subject of the email
      message_text (str): The message text of the email

    Returns:
      None
    '''
    global send_next_email, email_recipient
    send_next_email = False #set flag to False so that another email is not sent
    print('Sending email alert to ', email_recipient)
    prepare_and_send_email(email_sender, email_recipient, subject, message_text, im0)
    # wait for 10 minutes before sending another email
    time.sleep(600)
    send_next_email = True


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    '''Resize and pad image while meeting stride-multiple constraints

    Parameters:
        img (numpy.ndarray): Image to be padded
        new_shape (tuple): Desired output shape of (w, h) while meeting stride-multiple constraints
        color (tuple): Color
        auto (bool): Minimum rectangle
        scaleFill (bool): Stretch
        scaleup (bool): Scale up
        stride (int): Stride
    
    Returns:
        numpy.ndarray: Padded and resized image
    '''
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]

    # if new_shape is a single integer e.g. 640, convert to (640, 640)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def video_detection(conf_=0.25, frames_buffer=[]):
  '''This function will detect violations in a video file or a live stream 

  Parameters:
    conf_ (float): Confidence threshold for inference
    frames_buffer (list): A list of frames to be processed

  Returns:
    None
  '''    
  # Declare global variables to be used in this function
  global send_next_email
  global is_email_allowed
  global email_recipient

  violation_frames = 0 # Number of frames with violation
  
  import time
  #start_time = time.time()
  # total_detections = 0

  # create a list to store the detections
  global detections_summary

  #------ Customization  made by Anubhav Patrick------#
 
  #pop first frame from frames_buffer to get the first frame
  while True:
    if len(frames_buffer) > 0:
      _ = frames_buffer.pop(0)
      break
    #frame = frames_buffer.pop(0)

  #else path_x is a video file
  else:
    video_path = path_x
    video = cv2.VideoCapture(video_path)
    _, _ = video.read()

    #Video information
    fps = video.get(cv2.CAP_PROP_FPS)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Video information: ")
    print("FPS: ", fps)
    print("Width: ", w)
    print("Height: ", h)
    print("Number of frames: ", nframes)

  # Initialzing object for writing video output
  # output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'DIVX'),fps , (w,h))
  torch.cuda.empty_cache()
  # Initializing model and setting it for inference
  with torch.no_grad():
    weights, imgsz = opt['weights'], opt['img-size']
    set_logging()
    device = select_device(opt['device'])
    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
      model.half()

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    if device.type != 'cpu':
      model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    classes = None
    if opt['classes']:
      classes = []
      for class_name in opt['classes']:
        classes.append(opt['classes'].index(class_name))

    skip_frame = False
    #for j in range(nframes):
    while True:
    
        #------- Customization made by Anubhav Patrick --------#
        #if is_stream:
          # check if there are frames in the buffer

        if len(frames_buffer) > 0:
          #pop first frame from frames_buffer 
          img0 = frames_buffer.pop(0)
          if img0 is None:
            continue
          #print("Dimensions of frame: ", img0.shape)
          ret = True #we have successfully read one frame from stream
          if len(frames_buffer) >= 10:
            frames_buffer.clear() #clear the buffer if it has more than 10 frames to avoid memory overflow
        else:
          # buffer is empty, nothing to do
          continue
        
        '''else:
          # do predictions on alternate frames
          if skip_frame:
            ret, img0 = video.read()
            skip_frame = False
            continue
          else: 
            ret, img0 = video.read()
            skip_frame = True'''
        
        if ret:
          # perform predictions
          img = letterbox(img0, imgsz, stride=stride)[0]
          img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
          img = np.ascontiguousarray(img)
          img = torch.from_numpy(img).to(device)
          img = img.half() if half else img.float()  # uint8 to fp16/32
          img /= 255.0  # 0 - 255 to 0.0 - 1.0
          if img.ndimension() == 3:
            img = img.unsqueeze(0)

          # Inference
          t1 = time_synchronized()
          pred = model(img, augment= False)[0]

          # conf = 0.5
          total_detections = 0
          pred = non_max_suppression(pred, conf_, opt['iou-thres'], classes= classes, agnostic= False)
          
          t2 = time_synchronized()
          for i, det in enumerate(pred):
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
              det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

              unsafe = False # Flag to indicate if the frame is unsafe

              for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                total_detections += int(n)
                c = int(c)

                #we need to make sure at there is violation in atleast 5 continous frames
                # Check if the frame is unsafe
                if unsafe == False and (c == 0 or c == 1 or c == 3) and n > 0:
                  unsafe = True
                
                s += f"{n} {names[c]}{'s' * (n > 1)}, "  # add to string

              #code to send email on continous violations
              if unsafe == True and is_email_allowed == True:
                violation_frames += 1
                if violation_frames >= 5 and send_next_email == True:
                # reset the violation_frames since violation is detected
                  violation_frames = 0
                  # create a thread for sending email
                  t = threading.Thread(target=violation_alert_generator, args=(img0,))
                  t.start()
                elif unsafe == False:
                  # reset the number of violation_frames if current frame is safe
                  violation_frames = 0

              #get current time in hh:mm:ss format
              current_time = time.strftime("%H:%M:%S", time.localtime())
              detections_summary += f"\n {current_time}\n Total Detections: {total_detections}\n Detections per class: {s.split(maxsplit=1)[1]}\n###########\n"
              #print(detections_summary)
              
              for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                if label.startswith('safe'):
                  color = (0,255,0)
                else:
                  color = (0,0,255)

                plot_one_box(xyxy, img0, label=label, color=color, line_thickness=3)

          #fps_x = int((j+1)/(time.time() - start_time))
          fps_x = None
          # print(f"{j+1}/{nframes} frames processed")
          # print(conf)
          yield img0, fps_x, img0.shape, total_detections
          # cv2.imshow('hello',img0)
          # cv2.waitKey(1) & 0xFF == ord("q")

        else:
          # no more frames to read
          break
    
  '''if not is_stream:
    # output.release()
    video.release()'''
# cv2.imshow("image",img0)
# cv2.waitKey(0) & 0xFF == ord("q")
