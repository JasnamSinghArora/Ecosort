# This file loads the model and returns the predicted label of the waste.

from ultralytics import YOLO, SAM
from data_structures import classes
import numpy as np
import time
import os
from openai import OpenAI
import base64
import json, re
import cv2

CONF_THRESHOLD = 0.5 #Minimum amount of confidence required to make a prediction
MIN_CONF_SCORE = CONF_THRESHOLD * 4 #Minimum total confidence required to return a prediction. Used as a single object should be predicted multiple times to be an accurate prediction.

NEAR_AREA_RATIO = 0.15
SAM_IMG_SIZE = 1024 
YOLO_IMG_SIZE = 768 

os.environ["OPENAI_API_KEY"] = "YOUR_API_HERE"
client = OpenAI()

sam = YOLO("FastSAM-x.pt") # load segmentation model
model = YOLO("new.pt") # load prediction model


PROMPT = (
  "Return a JSON object with keys 'bg_fg' and 'category'. "
  "'bg_fg' must be 'FOREGROUND' or 'BACKGROUND'. "
  "If 'bg_fg' is 'BACKGROUND', 'category' must be null. "
  "'category' must be one of ['Wet Waste', 'Dry Waste', 'Hazardous Waste'] where trash is anything that is neither recyclable nor compostable"
  "Return JSON only, no extra text."
)

def prediction(cap, cap_lock):

    with cap_lock:
        ret, frame = cap.read()

    segmentation_results = sam.predict(source=frame, imgsz=SAM_IMG_SIZE) #segment image
    print("Segmented")

    for segmentation_result in segmentation_results:
        
        # get required info form results
        masks_bool = segmentation_result.masks.data.cpu().numpy().astype(bool)
        N = masks_bool.shape[0]
        areas = []
        
        for i in range(N): #loop through each segmentation
            area = int(masks_bool[i].sum()) #save area of each segmentaton
            areas.append(area)
            
        if areas: #find area of largest object 
            best_idx = int(np.argmax(areas))
            best_area = areas[best_idx]
            h, w = frame.shape[:2]
            frame_area = float(h * w)
            area_ratio = best_area / frame_area
            print("RATIO:", area_ratio)
            
            if area_ratio >= NEAR_AREA_RATIO: #if big enough
                        
                t1 = time.time()
                max_score = 0
                for key in classes: #reset confidence scores dictionary
                    classes[key] = 0
                    
                while max_score < MIN_CONF_SCORE and time.time()-t1 < 5:
                    results = model(frame, conf=CONF_THRESHOLD, imgsz=YOLO_IMG_SIZE) # get prediction form model
                    print("Detected")
                    
                    for r in results:  # Each detection result
                        class_ids = r.boxes.cls.cpu().numpy() if r.boxes is not None else []
                        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
                        names = [model.names[int(c)] for c in class_ids]
                        
                        for name, conf in zip(names, confs): #sum up confidence scores 
                            classes[name] += conf
                            
                    max_score = max(classes.values())
                    
                if max_score < MIN_CONF_SCORE:
                    return None
            
                ok, buf = cv2.imencode(".jpg", frame)  
                if ok: # can make API call
                    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

                    t1 = time.time()
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",     
                        temperature=0,
                        messages=[
                            {"role": "system", "content": PROMPT},
                            {"role": "user", "content": [
                                {"type": "text", "text": "Classify the waste item."},
                                {"type": "image_url", "image_url": {"url": data_url}}
                            ]}
                        ]
                    )
                    raw = resp.choices[0].message.content.strip()
                    print("RAW:", raw)
                    print("API took: ",time.time()-t1)

                    m = re.search(r"\{.*\}", raw, re.DOTALL)
                    output = json.loads(m.group(0) if m else raw)

                    if output.get("bg_fg") == "FOREGROUND":
                        return output.get("category")

    return None
