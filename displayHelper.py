# This file contains a series of helper function used in display.py

import time
import cv2
import pygame
import numpy as np
from model import prediction
import threading

X = 1400
Y = 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 128, 0)
LIGHT_GREEN = (144, 238, 144)
RESULT_TIME = 7

ECO_TIPS = {
    "Recyclable Waste": "Always keep recyclables clean, dry, and free from food residue",
    "Compostable Waste": "Add to a compost bin. Avoid mixing with plastic or glass.",
    "Trash": "Dispose responsibly. Reduce single-use items to minimize landfill waste."
}

status_text = "Scanning for items..."
size = 54
cap_lock = threading.Lock()

class Box: #Object to represent label boxes
    def __init__(self, x, y, w, h, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.color = WHITE   

    def draw(self, screen, font):
        pygame.draw.rect(screen, self.color, self.rect)
        pygame.draw.rect(screen, GREEN, self.rect, 2)
       
        text_surface = font.render(self.label, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

def create_boxes():
    box_w = 350
    box_h = 250
    gap = 20
    total_w = box_w * 3 + gap * 2
    start_x = (X - total_w) // 2   
    y = Y // 2 + 50 

    labels = ["Wet Waste", "Dry Waste", "Hazardous Waste"]
    boxes = []
    for i, lbl in enumerate(labels):
        x = start_x + i * (box_w + gap)
        boxes.append(Box(x, y, box_w, box_h, lbl))
    return boxes

boxes = create_boxes() #generate the boxes

def load_images(screen): #renders images
    left_image = pygame.image.load("jpis.png")
    right_image = pygame.image.load("logo.png")

    left_image = left_image.convert_alpha()
    right_image = right_image.convert_alpha()

    left_image = pygame.transform.smoothscale(left_image, (300, 300))
    screen.blit(left_image, (75, 0))

    right_image = pygame.transform.smoothscale(right_image, (500, 500))
    screen.blit(right_image, (X - 490, -100))
    
    
def load_camera(screen, cap): #renders camera
    with cap_lock:
        ret, frame = cap.read()
    
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        
        frame_surface = pygame.surfarray.make_surface(frame)
        frame_surface = pygame.transform.scale(frame_surface, (X//3, Y//3))
        
        screen.blit(frame_surface, (X/3, 20))
        
def draw_boxes(screen): #display boxes
    font = pygame.font.SysFont(None, 48)
    for box in boxes:
        box.draw(screen, font)
        
    
def load(screen, cap): #loads everything
    global size
    font = pygame.font.SysFont(None, size)
    screen.fill(WHITE)
    load_images(screen)
    load_camera(screen, cap)
    text_surface = font.render(status_text, True, BLACK)
    text_rect = text_surface.get_rect(center=(X//2, Y//2 - 50))
    screen.blit(text_surface, text_rect)
    draw_boxes(screen)
    
def highlight(label): #gets prediction and highlights appropriate box
    global size
    global status_text
    size = 35
    status_text = f"Predicted: {label}"
    eco_tip = ECO_TIPS.get(label, "")
    status_text += f" | {eco_tip}"
    for box in boxes:
        if box.label == label:
            box.color = LIGHT_GREEN
            time.sleep(RESULT_TIME)
            box.color = WHITE
            status_text = "Scanning for items..."
            size = 54
        else:
            box.color = WHITE

#This wrapper function is used in order to utilize a cap_lock so that the 
#"prediction" and "load_camera" functions both don't access the camera at the same time.
def locked_prediction(cap): #gets the prediction
    return prediction(cap, cap_lock)
