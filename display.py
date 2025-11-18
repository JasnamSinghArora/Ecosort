# This file is responsible for rendering the pygame display and integrating all other files into one pipeline

import sys
import cv2
import pygame
from displayHelper import load, highlight, locked_prediction
import threading

WHITE = (255, 255, 255)
X = 1400
Y = 800
cap = cv2.VideoCapture(0) # turn camera on

pygame.init()

stop_event = threading.Event() # to exit interface when needed

screen = pygame.display.set_mode((X, Y))
pygame.display.set_caption("Ecosort")

def run_interface(): 
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # user closed window
                running = False
        
        load(screen, cap) #loads everything required for interface
        pygame.display.flip()

    stop_event.set()

def predict(): #gets and displays the prediction
    while not stop_event.is_set():
        label = None
        while label is None:
            label = locked_prediction(cap)
        highlight(label)

#multiprocessing for the model
model = threading.Thread(target=predict, daemon=True) 
model.start()

run_interface()

stop_event.set() #user exited
if model.is_alive():
    model.join(timeout=2.0)

cap.release()
pygame.quit()
sys.exit()
