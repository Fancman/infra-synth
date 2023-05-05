import argparse
import sys
from pathlib import Path
import os
import torch
import numpy as np
import cv2
import math
from datetime import datetime
import pathlib
from ultralytics import YOLO


red   = (0,0,255)
green = (0,255,0)
blue  = (255,0,0)
black = (0,0,0)
white = (255, 255, 255)

# Geometric figures settings
thickness = 3
circle_radius = 6
fill = -1 # to fill the geometric figure

# Text settings
text_thickness = 1
text_size = 0.4
title_thickness = 2
title_size = 1
title = 'Threat detection'
font = cv2.FONT_HERSHEY_SIMPLEX # or cv2.FONT_HERSHEY_PLAIN

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


KNIFE_CONF_THRESH = 0.3
PERSON_CONF_THRESH = 0.5


def list_files_in_dir(dir, type = "video"):
	large_dir = pathlib.Path(dir)

	filenames = []

	for item in large_dir.rglob("*"):
		basename = os.path.basename(item)
		filename = os.path.splitext(basename)[0]
		knife_exists = False
	
		ext = os.path.splitext(basename)[1]

		ext = ext.lower()

		#print(item, ext)

		if type == "video" and ext not in [".m4v"]:
			continue

		if type == "image" and ext not in [".jpg", ".png", ".jpeg"]:
			continue

		path_parts = os.path.split( os.path.abspath(item) )
		parent_dir = path_parts[0]

		if os.path.exists(os.path.join(parent_dir, "knife.txt") ):
			knife_exists = True
			
		record = {
			"filepath": item,
			"knife": knife_exists
		}

		filenames.append(record)

	return filenames

def validate_video(opt, records):
	if opt.model_version == 'yolov8':
		knives_model = YOLO(opt.knife_weights)  # load a custom model
	else:
		knives_model = torch.hub.load('ultralytics/yolov5', 'custom', path=opt.knife_weights, force_reload=True)

	person_model = torch.hub.load('ultralytics/yolov5', 'custom', path=opt.person_weights, force_reload=True)

	analyzed_files = {}

	tp_knife = 0
	tn_knife = 0
	fp_knife = 0
	fn_knife = 0

	for idx, record in enumerate(records):

		filename = record["filepath"]
		knife_exists_in_video = record["knife"]
		knife_detected_in_video = False

		filename = str(filename)		

		cap = cv2.VideoCapture(filename)

		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")

		frames_processed = 0
		fps = int(cap.get(cv2.CAP_PROP_FPS)) # Frame rate
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Number of frames in the video file

		print("["+str(idx+1)+"/"+str(len(records))+"] Started video: " + filename + " with fps: " + str(fps) + " recognition at: ", current_time)

		knife_or_person_found_in_last_n_frames = 0

		knives_detect_cnt = 0
		people_detect_cnt = 0

		# Iterate number of seconds
		#for i in range( int( frame_count / fps ) ):
		# Loop every frame of video
		for i in range( frame_count ):

			if frames_processed > i:
				continue

			success, frame = cap.read()

			# Make detections 
			person_results = person_model(frame)

			if opt.model_version == 'yolov8':
				knife_results = knives_model(frame, verbose=False)
			else:
				knife_results = knives_model(frame)	

			seconds = round(i / fps, 2)

			frames_shift = fps / 1

			#print("Infering person and knife models at: " + str(seconds))

			person_found, knife_found = detect_knife(frame, seconds, person_results, knife_results, with_visual = False, model_version=opt.model_version)

			if knife_found:
				knives_detect_cnt += 1

			if person_found:
				people_detect_cnt += 1

			if person_found and knife_found:
				knife_or_person_found_in_last_n_frames = 5
				
				if knives_detect_cnt >= 3:
					knife_detected_in_video = True
					print("Stop because reached 4 knife detections")
					break
			
			if knife_or_person_found_in_last_n_frames > 0:
				knife_or_person_found_in_last_n_frames -= 1
				frames_shift = fps / 5	

			if frames_processed + frames_shift > frame_count:
				frames_processed = frame_count

			#print("Adding frame shift: " + str(frames_shift))

			frames_processed += frames_shift # fps

			cap.set(cv2.CAP_PROP_POS_FRAMES, frames_processed) #skip one second

		# Knife was found in video and realy is in video
		if knife_detected_in_video and knife_exists_in_video:
			tp_knife += 1

		# Knife was not found in video and realy is not in video
		if not knife_detected_in_video and not knife_exists_in_video:
			tn_knife += 1

		# Knife was not found in video but knife is in video
		if not knife_detected_in_video and knife_exists_in_video:
			fp_knife += 1

		# Knife was found in video but knife is not in video
		if knife_detected_in_video and not knife_exists_in_video:
			fn_knife += 1

		analyzed_files[filename] = {
			"knives_detect_cnt": knives_detect_cnt,
			"people_detect_cnt": people_detect_cnt,
		}

		now = datetime.now()

		current_time = now.strftime("%H:%M:%S")

		print("Ended video recognition at: ", current_time)

	print("--------------DETECTION COUNTS--------------")
	print(analyzed_files)
	print("---------------------True vs. False and Positive vs. Negative-----------------------")
	print(f"TP: {tp_knife}, TN: {tn_knife}, FP: {fp_knife}, FN: {fn_knife}")

def predict_video(opt, person_model, knives_model):
	cap = cv2.VideoCapture(opt.input)	

	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	print("Started video recognition at: ", current_time, cap.isOpened())	

	frames_processed = 0
	fps = int(cap.get(cv2.CAP_PROP_FPS)) # Frame rate
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Number of frames in the video file

	knife_or_person_found_in_last_n_frames = 0

	# Iterate number of seconds
	#for i in range( int( frame_count / fps ) ):
	for i in range( frame_count ):

		if frames_processed > i:
			continue

		success, frame = cap.read()

		# Make detections 
		person_results = person_model(frame)

		if opt.model_version == 'yolov8':
			knife_results = knives_model(frame, verbose=False)
		else:
			knife_results = knives_model(frame)

		seconds = round(i / fps, 2)

		frames_shift = fps / 4

		person_found, knife_found = detect_knife(frame, seconds, person_results, knife_results, with_visual = True, model_version=opt.model_version)

		if person_found and knife_found:
			knife_or_person_found_in_last_n_frames = 10
		
		if knife_or_person_found_in_last_n_frames > 0:
			knife_or_person_found_in_last_n_frames -= 1
			frames_shift = fps / 5			

		frames_processed += frames_shift # fps

		cap.set(cv2.CAP_PROP_POS_FRAMES, frames_processed) #skip one second

		cv2.imshow(title, frame)
		
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

	now = datetime.now()

	current_time = now.strftime("%H:%M:%S")

	print("Ended video recognition at: ", current_time)	

def predict_photo(opt, person_model, knives_model):

	img = cv2.imread(opt.input, cv2.IMREAD_COLOR)

	# Make detections 
	person_results = person_model(img)

	if opt.model_version == 'yolov8':
		knife_results = knives_model(img, verbose=False)
	else:
		knife_results = knives_model(img)

	person_found, knife_found = detect_knife(img, 0, person_results, knife_results, with_visual = True, model_version=opt.model_version)

	print(f"Person found: {person_found}", f"Knife found: {knife_found}")

	cv2.imshow(title, img)
	
	cv2.waitKey(0) 

	cv2.destroyAllWindows()

def predict(opt):
	person_model = torch.hub.load('ultralytics/yolov5', 'custom', path=opt.person_weights, force_reload=True)

	if opt.model_version == 'yolov8':
		knives_model = YOLO(opt.knife_weights)  # load a custom model
	else:
		knives_model = torch.hub.load('ultralytics/yolov5', 'custom', path=opt.knife_weights, force_reload=True)

	if os.path.exists(os.path.join(opt.input) ):
		print(f"Input file with path dont exist: {opt.input}")

	if opt.file_type == "video":
		predict_video(opt, person_model, knives_model)
	elif opt.file_type == "photo":
		predict_photo(opt, person_model, knives_model)
	

def detect_knife(frame, seconds, person_results, knife_results, with_visual = True, model_version = "yolov5"):
	person_found = knife_found = False

	# We extract the needed informations: xyxy, xywh
	person_model_predictions_xyxy = person_results.pandas().xyxy[0]
	person_model_predictions_xywh = person_results.pandas().xywh[0]

	# Let us consider only the 'person' label
	person_predictions_xyxy = person_model_predictions_xyxy[person_model_predictions_xyxy['name']=='person']
	person_predictions_xywh = person_model_predictions_xywh[person_model_predictions_xywh['name']=='person']
	
	# Let's adjust the indeces (they might be not good since we considered just the 'person' label)
	person_predictions_xyxy.index = range(len(person_predictions_xyxy))
	person_predictions_xywh.index = range(len(person_predictions_xywh))
	
	valid_persons = []
		
	best_person_detection = ()
	best_person_confidence = 0
	
	# For every person in the frame:
	for n in range(len(person_predictions_xyxy)):        

		confidence = round(float(person_predictions_xyxy['confidence'][n]), 2)
		
		# Person threshold
		if confidence < PERSON_CONF_THRESH:
			continue
			
		if confidence < best_person_confidence:
			continue
			
		# Save the coordinates of the box
		x_min = int(person_predictions_xyxy['xmin'][n])
		y_min = int(person_predictions_xyxy['ymin'][n])
		x_max = int(person_predictions_xyxy['xmax'][n])
		y_max = int(person_predictions_xyxy['ymax'][n])     

		valid_persons.append((x_min, y_min, x_max, y_max))

		best_person_detection = (x_min, y_min, x_max, y_max)
		best_person_confidence = confidence
		
		# and the coordinates of the center of each box
		x_center = int(person_predictions_xywh['xcenter'][n])
		y_center = int(person_predictions_xywh['ycenter'][n])  
		
	if best_person_detection:

		if with_visual == True:
			# Let's draw the bounding box  
			person_x_min = best_person_detection[0]
			person_y_min = best_person_detection[1]
			person_x_max = best_person_detection[2]
			person_y_max = best_person_detection[3]

			cv2.rectangle(frame, (person_x_min, person_y_min), (person_x_max, person_y_max), red, thickness);
			cv2.putText(frame, 'Person' + str(best_person_confidence), (person_x_min-3, person_y_min-5), font, text_size, red, text_thickness);
		
		person_found = True

		# Try to detect knife if person was found

		if model_version == "yolov8":
		
			for result in knife_results:
				conf = result.boxes.conf
				conf_list = conf.tolist()

				if len(conf_list) == 0:
					continue

				confidence = round(float(conf_list[0]), 2)

				#print("confidence: ", confidence)

				if confidence < KNIFE_CONF_THRESH:
					continue				

				for bbox in result.boxes.xyxy:
					x1, y1, x2, y2 = int(bbox[0].item()), int(bbox[1].item()), int(bbox[2].item()), int(bbox[3].item())
				
					if with_visual == True:

						cv2.rectangle(frame, (x1, y1), (x2, y2), red, thickness)
						
						cv2.putText(frame, 'Knife' + str(confidence), (x1-3, y1-5), font, text_size, red, text_thickness)
				
					print("Human and knife with conf("+str(confidence)+") appeared at " + str(seconds) + " second")

					knife_found = True

			return person_found, knife_found
	
		# We extract the needed informations: xyxy, xywh
		knife_model_predictions_xyxy = knife_results.pandas().xyxy[0]
		knife_model_predictions_xywh = knife_results.pandas().xywh[0]

		# Let us consider only the 'person' label
		knife_predictions_xyxy = knife_model_predictions_xyxy[knife_model_predictions_xyxy['name']=='knife']
		knife_predictions_xywh = knife_model_predictions_xywh[knife_model_predictions_xywh['name']=='knife']
		
		# Let's adjust the indeces (they might be not good since we considered just the 'person' label)
		knife_predictions_xyxy.index = range(len(knife_predictions_xyxy))
		knife_predictions_xywh.index = range(len(knife_predictions_xywh))
		
		valid_knives = []
			
		best_knife_detection = ()
		best_knife_confidence = 0
		
		# For every person in the frame:
		for n in range(len(knife_predictions_xyxy)):        

			confidence = round(float(knife_predictions_xyxy['confidence'][n]), 2)
			
			# Person threshold
			if confidence < KNIFE_CONF_THRESH:
				continue
				
			if confidence < best_knife_confidence:
				continue
				
			# Save the coordinates of the box
			x_min = int(knife_predictions_xyxy['xmin'][n])
			y_min = int(knife_predictions_xyxy['ymin'][n])
			x_max = int(knife_predictions_xyxy['xmax'][n])
			y_max = int(knife_predictions_xyxy['ymax'][n])     

			valid_knives.append((x_min, y_min, x_max, y_max))

			best_knife_detection = (x_min, y_min, x_max, y_max)
			best_knife_confidence = confidence
			
			# and the coordinates of the center of each box
			x_center = int(knife_predictions_xywh['xcenter'][n])
			y_center = int(knife_predictions_xywh['ycenter'][n])  
			
		if best_knife_detection:

			if with_visual == True:
				# Let's draw the bounding box  
				knife_x_min = best_knife_detection[0]
				knife_y_min = best_knife_detection[1]
				knife_x_max = best_knife_detection[2]
				knife_y_max = best_knife_detection[3]

				cv2.rectangle(frame, (knife_x_min, knife_y_min), (knife_x_max, knife_y_max), red, thickness);
				
				cv2.putText(frame, 'Knife' + str(best_knife_confidence), (knife_x_min-3, knife_y_min-5), font, text_size, red, text_thickness);
			
			print("Human and knife with conf("+str(best_knife_confidence)+") appeared at " + str(seconds) + " second")

			knife_found = True

	return person_found, knife_found

def infer_video(opt):
	if opt.model_version == 'yolov8':
		knives_model = YOLO(opt.knife_weights)  # load a custom model
	else:
		knives_model = torch.hub.load('ultralytics/yolov5', 'custom', path=opt.knife_weights, force_reload=True)

	person_model = torch.hub.load('ultralytics/yolov5', 'custom', path=opt.person_weights, force_reload=True)

	# Search for knife in whole image
	cap = cv2.VideoCapture(opt.input)
	#cap = cv2.VideoCapture("E:/Fotopasca/20.3.2023 - Rotunda/DSCF0001.AVI")

	fps = cap.get(cv2.CAP_PROP_FPS)
	timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
	calc_timestamps = [0.0]

	now = datetime.now()

	current_time = now.strftime("%H:%M:%S")

	print("Started video recognition at: ", current_time, cap.isOpened())	

	while cap.isOpened():
		ret, frame = cap.read()
		
		if ret == True:
			...
		else:
			print('Some error')
			break
			
		#crop timestamp from bottom part    
		w = frame.shape[1]
		h = frame.shape[0]
		
		#frame = frame[0:(h-60), 0:w]
		#frame = frame[0:h, 0:w]
			
		timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
		video_time_ms = calc_timestamps[-1] + 1000/fps
		calc_timestamps.append(video_time_ms)

		#continue
		
		# Make detections 
		person_results = person_model(frame)

		if opt.model_version == 'yolov8':
			knife_results = knives_model(frame, verbose=False)
		else:
			knife_results = knives_model(frame)
		
		# We extract the needed informations: xyxy, xywh
		person_model_predictions_xyxy = person_results.pandas().xyxy[0]
		person_model_predictions_xywh = person_results.pandas().xywh[0]

		# Let us consider only the 'person' label
		person_predictions_xyxy = person_model_predictions_xyxy[person_model_predictions_xyxy['name']=='person']
		person_predictions_xywh = person_model_predictions_xywh[person_model_predictions_xywh['name']=='person']
		
		# Let's adjust the indeces (they might be not good since we considered just the 'person' label)
		person_predictions_xyxy.index = range(len(person_predictions_xyxy))
		person_predictions_xywh.index = range(len(person_predictions_xywh))
		
		valid_persons = []
			
		best_person_detection = ()
		best_person_confidence = 0
		
		#print("person_predictions_xyxy", person_predictions_xyxy)
		
		# For every person in the frame:
		for n in range(len(person_predictions_xyxy)):        

			confidence = round(float(person_predictions_xyxy['confidence'][n]), 2)
			
			# Person threshold
			if confidence < 0.3:
				continue
				
			if confidence < best_person_confidence:
				continue
				
			# Save the coordinates of the box
			x_min = int(person_predictions_xyxy['xmin'][n])
			y_min = int(person_predictions_xyxy['ymin'][n])
			x_max = int(person_predictions_xyxy['xmax'][n])
			y_max = int(person_predictions_xyxy['ymax'][n])     

			valid_persons.append((x_min, y_min, x_max, y_max))

			best_person_detection = (x_min, y_min, x_max, y_max)
			best_person_confidence = confidence
			
			# and the coordinates of the center of each box
			x_center = int(person_predictions_xywh['xcenter'][n])
			y_center = int(person_predictions_xywh['ycenter'][n])  
			
		if best_person_detection:
			# Let's draw the bounding box  
			person_x_min = best_person_detection[0]
			person_y_min = best_person_detection[1]
			person_x_max = best_person_detection[2]
			person_y_max = best_person_detection[3]

			cv2.rectangle(frame, (person_x_min, person_y_min), (person_x_max, person_y_max), red, thickness);
			cv2.putText(frame, 'Person' + str(best_person_confidence), (person_x_min-3, person_y_min-5), font, text_size, red, text_thickness);
			
		
		# We extract the needed informations: xyxy, xywh
		knife_model_predictions_xyxy = knife_results.pandas().xyxy[0]
		knife_model_predictions_xywh = knife_results.pandas().xywh[0]

		# Let us consider only the 'person' label
		knife_predictions_xyxy = knife_model_predictions_xyxy[knife_model_predictions_xyxy['name']=='knife']
		knife_predictions_xywh = knife_model_predictions_xywh[knife_model_predictions_xywh['name']=='knife']
		
		# Let's adjust the indeces (they might be not good since we considered just the 'person' label)
		knife_predictions_xyxy.index = range(len(knife_predictions_xyxy))
		knife_predictions_xywh.index = range(len(knife_predictions_xywh))
		
		valid_knives = []
			
		best_knife_detection = ()
		best_knife_confidence = 0
		
		# For every person in the frame:
		for n in range(len(knife_predictions_xyxy)):        

			confidence = round(float(knife_predictions_xyxy['confidence'][n]), 2)
			
			# Person threshold
			if confidence < 0.3:
				continue
				
			if confidence < best_knife_confidence:
				continue
				
			# Save the coordinates of the box
			x_min = int(knife_predictions_xyxy['xmin'][n])
			y_min = int(knife_predictions_xyxy['ymin'][n])
			x_max = int(knife_predictions_xyxy['xmax'][n])
			y_max = int(knife_predictions_xyxy['ymax'][n])     

			valid_knives.append((x_min, y_min, x_max, y_max))

			best_knife_detection = (x_min, y_min, x_max, y_max)
			best_knife_confidence = confidence
			
			# and the coordinates of the center of each box
			x_center = int(knife_predictions_xywh['xcenter'][n])
			y_center = int(knife_predictions_xywh['ycenter'][n])  
			
		if best_knife_detection:
			# Let's draw the bounding box  
			knife_x_min = best_knife_detection[0]
			knife_y_min = best_knife_detection[1]
			knife_x_max = best_knife_detection[2]
			knife_y_max = best_knife_detection[3]

			cv2.rectangle(frame, (knife_x_min, knife_y_min), (knife_x_max, knife_y_max), red, thickness);
			
			cv2.putText(frame, 'Knife' + str(best_knife_confidence), (knife_x_min-3, knife_y_min-5), font, text_size, red, text_thickness);
			
			print("Human and knife appeared at " + str(math.floor(video_time_ms / 1000)) + " second")
		
		cv2.imshow(title, frame)
		
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break
			
	cap.release()
	cv2.destroyAllWindows()

	now = datetime.now()

	current_time = now.strftime("%H:%M:%S")

	print("Ended video recognition at: ", current_time)	

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=ROOT / 'images', help='input image, video or dir which should be processed')
    parser.add_argument('--task', type=str, default='predict', help='predict or validate')
    parser.add_argument('--file-type', type=str, default='video', help='video or photo')
    parser.add_argument('--person-weights', type=str, default=ROOT / 'weights', help='weights file')
    parser.add_argument('--knife-weights', type=str, default=ROOT / 'weights', help='weights file')
    parser.add_argument('--model-version', type=str, default='yolov5', help='use either yolov5 or yolov8 on knife detection')	
    
    return parser.parse_known_args()[0] if known else parser.parse_args()

def main(opt):
    	
	if opt.task == "predict":
		print("Started predicting")
		predict(opt)
		
	elif opt.task == "validate":
		print("Started validating")
		filenames = list_files_in_dir(opt.input, type = "video")
		validate_video(opt, filenames)

def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
