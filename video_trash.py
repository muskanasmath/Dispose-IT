import cv2
import numpy as np
import sys
import time
import copy

#add this to python path, since weights are there
sys.path.append('/trash/Project_Trash_Mask_RCNN/Mask-RCNN/Mask_RCNN-master/')
print(sys.path)

DIST_LIMIT = 100

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image

#Generate two text boxes a larger one that covers them
def merge_boxes(box1, box2):
    return [min(box1[0], box2[0]), 
         min(box1[1], box2[1]), 
         max(box1[2], box2[2]),
         max(box1[3], box2[3])]



#Computer a Matrix similarity of distances of the text and object
def calc_sim(text, obj):
    # text: ymin, xmin, ymax, xmax
    # obj: ymin, xmin, ymax, xmax
    text_ymin, text_xmin, text_ymax, text_xmax = text
    obj_ymin, obj_xmin, obj_ymax, obj_xmax = obj

    x_dist = min(abs(text_xmin-obj_xmin), abs(text_xmin-obj_xmax), abs(text_xmax-obj_xmin), abs(text_xmax-obj_xmax))
    y_dist = min(abs(text_ymin-obj_ymin), abs(text_ymin-obj_ymax), abs(text_ymax-obj_ymin), abs(text_ymax-obj_ymax))

    dist = x_dist + y_dist
    return dist

#Principal algorithm for merge text 
def merge_algo(ids_copied, boxes_copied, scores_copied):
    for i, (ids_1, box_1,score_1) in enumerate(zip(ids_copied, boxes_copied,scores_copied)):
        for j, (ids_2, box_2,score_2) in enumerate(zip(ids_copied, boxes_copied, scores_copied)):
            if j <= i:
                continue
            # Create a new box if a distances is less than disctance limit defined 
            if calc_sim(box_1, box_2) < DIST_LIMIT:
            # Create a new box  
                new_box = merge_boxes(box_1, box_2)            
                boxes_copied[i] = new_box
                #delete previous boxes
                boxes_copied = np.delete(boxes_copied, j, axis = 0)
                # Create a new text string 
                # new_mask = np.logical_or(mask_1,mask_2)
                # masks_copied[i] = new_mask
                # # delete previous mask 
                # masks_copied = np.delete(masks_copied, j, axis = 0)
                new_ids = ids_1 
                ids_copied[i] = new_ids
                #delete previous ids 
                ids_copied = np.delete(ids_copied, j,axis = 0)
                new_score = max(score_1,score_2)
                scores_copied[i] = new_score
                #delte previous score
                scores_copied = np.delete(scores_copied, j, axis = 0)

                return True, ids_copied, boxes_copied,scores_copied
    return False, ids_copied, boxes_copied, scores_copied


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    '''
    ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    '''
    empty = False
    # print(masks)
    # print(len(boxes), len(masks))
    # for i in range(len(boxes)):
    	# print(ids[i],boxes[i],scores[i],masks[i])
    ids_copied = copy.deepcopy(ids)
    boxes_copied = copy.deepcopy(boxes)
    scores_copied = copy.deepcopy(scores)
    # masks_copied = copy.deepcopy(masks)

    need_to_merge = True
    while need_to_merge:
    	need_to_merge, ids_copied, boxes_copied, scores_copied = merge_algo(ids_copied, boxes_copied, scores_copied)

    # ids = ids_copied
    # boxes = boxes_copied
    # scores = scores_copied
    # masks = masks_copied
    # print(boxes)
    # print(len(masks), len(boxes))
    # for i in range(len(boxes_copied)):
    # 	print(ids_copied[i],boxes_copied[i],scores_copied[i])
    '''
    
    '''

    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
        empty = True
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        # we want the colours to only ne in one color: SIFR orange ff5722
        # color = (255, 87, 34)
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]
        '''
        |||||||||||||||||||||||||||||
        '''
        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        ) 
        #cropping image according to bounding boxes in each frame ( taking additional 20 pixels around all edges )
        # y,x = org_image.shape[0],org_image.shape[1]
        # ymin = int(y1)-20
        # if ymin < 0:
        # 	ymin = 0
        # ymax = int(y2)+20
        # if ymax > y:
        # 	ymax = y
        # xmin = int(x1)-20
        # if xmin < 0 :
        # 	xmin = 0   
        # xmax = int(x2)+20
       	# if xmax > x:
       	# 	xmax = x
        # cropped_img = temp[ymin:ymax, xmin:xmax]
        # cv2.imshow('frame', temp)
        # time.sleep(5)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        # 	break
        # cropped_images.append(cropped_img)
        # for cropped_img in cropped_images:
        # 	cv2.imshow('frame', cropped_img)
        # 	time.sleep(5)
        # 	if cv2.waitKey(1) & 0xFF == ord('q'):
        # 		break

    return image, boxes_copied, empty


if __name__ == '__main__':
    """
        test everything
    """
    import os
    import sys

    # adding these to path 
    sys.path.append('/trash/Project_Trash_Mask_RCNN/Mask-RCNN/Mask_RCNN-master/')
    sys.path.append('/Project_Trash_Mask_RCNN/Mask-RCNN/Mask_RCNN-master')

    import trash
    from mrcnn import model as modellib, utils
    #import utils
    #import model as modellib
    

    batch_size = 1

    ROOT_MODEL_DIR = os.path.abspath("../../")
    MODEL_DIR = os.path.join(ROOT_MODEL_DIR, "logs")
    ROOT_DIR = os.getcwd()
    VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
    VIDEO_SAVE_DIR = os.path.join(VIDEO_DIR, "save_short")
    VIDEO_CROPPED_SAVE_DIR = os.path.join(VIDEO_SAVE_DIR,"save_cropped_short")
    # varem: "weights/mask_rcnn_trash_0050_030419.h5"
    MODEL_PATH = "Trash_Detection\\weights\\mask_rcnn_trash_0200_030519_large.h5"


    config = trash.TrashConfig()

    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(
        mode="inference", model_dir=MODEL_DIR, config=config
    )
    weights_path = os.path.join(ROOT_MODEL_DIR, MODEL_PATH)
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    class_names = [
        'BG', 'trash'
    ]
    '''
	
	'''
    capture = cv2.VideoCapture(os.path.join(VIDEO_DIR, 'Trash_short.mp4'))
    # capture = cv2.VideoCapture(0)

    try:
        if not os.path.exists(VIDEO_SAVE_DIR):
            os.makedirs(VIDEO_SAVE_DIR)
        if not os.path.exists(VIDEO_CROPPED_SAVE_DIR):
        	os.makedirs(VIDEO_CROPPED_SAVE_DIR)
    except OSError:
        print ('Error: Creating directory of data')
    frames = []
    frame_count = 0
    # these 2 lines can be removed if you dont have a 1080p camera.
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    avg_area = []
    count = 0
    while True:
        ret, frame = capture.read()
        # Bail out when the video file ends
        if not ret:
            break
        
        # Save each frame of the video to a list
        '''
        
        n = len(frame)
		frame = frame[0:3:n]
        '''
        frame_count += 1
        frames.append(frame)

        
        temp = copy.deepcopy(frame)
        if len(frames) == batch_size and count%4 == 0:
        	print('frame_count :{0}'.format(frame_count))
        	results = model.detect(frames, verbose=0)
        	print('Predicted')
        	for i, item in enumerate(zip(frames, results)):
        		cropped_images = []
        		frame = item[0]
        		r = item[1]
        		# cv2.imshow('frame 1', temp)
        		# # time.sleep(5)
        		# if cv2.waitKey(1) & 0xFF == ord('q'):
        		# 	break
        		frame, boxes, empty = display_instances(
        			frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        		# cv2.imshow('frame 2', temp)
        		# # time.sleep(5)
        		# if cv2.waitKey(1) & 0xFF == ord('q'):
        		# 	break
        		if empty : continue
        		name = '{0}.jpg'.format(frame_count + i - batch_size+1)
        		name = os.path.join(VIDEO_SAVE_DIR, name)
        		cv2.imwrite(name, frame)
        		c = 1
        		for i in range(len(boxes)):
	        		y1, x1, y2, x2 = boxes[i]
	        		y,x = temp.shape[0],temp.shape[1]
	        		ymin = int(y1)-20
	        		if ymin < 0:
	        			ymin = 0
	        		ymax = int(y2)+20
	        		if ymax > y:
	        			ymax = y
			        xmin = int(x1)-20
			        if xmin < 0 :
			        	xmin = 0   
			        xmax = int(x2)+20
			       	if xmax > x:
			       		xmax = x
			        cropped_img = temp[ymin:ymax, xmin:xmax]
			        # cv2.imshow('frame 3', cropped_img)
			        # # time.sleep(5)
			        # if cv2.waitKey(1) & 0xFF == ord('q'):
			        # 	break
			        cropped_images.append(cropped_img)
        		for cropped_img in cropped_images:
        			# cv2.imshow('frame', cropped_img)
        			# time.sleep(5)
        			# if cv2.waitKey(1) & 0xFF == ord('q'):
        			# 	break
        			cropped_img_name = '{0}_{1}.jpg'.format((frame_count + i - batch_size+1),c)
        			cropped_img_path = os.path.join(VIDEO_CROPPED_SAVE_DIR, cropped_img_name)
        			cv2.imwrite(cropped_img_path, cropped_img)
        			print('Boundary box {1} saved :{0}'.format(cropped_img_name,c))
        			c+=1
        		'''
        		idhar |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        		'''
        		print('writing to file:{0}'.format(name))
        		'''
        		|||||||||||||||||||||||||||||||||||||||||||
        		'''
        		#area estimation of masked pixel 
        		img_a = copy.deepcopy(temp)
        		img_a = img_a[ymin:ymax, xmin:xmax]
        		gray = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        		thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        		pixels = cv2.countNonZero(thresh)
        		pixels = (img_a.shape[0] * img_a.shape[1]) - pixels
        		area = pixels/1000
        		avg_area.append(area)
				#res_avg_area= avg(avg_area)
        			# total += pixels
        		# cv2.putText(img_a, '{}'.format(pixels), (xx,yy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        		print("Area of the garbage pile = {0}".format(pixels))
        		# cv2.imshow('thresh', thresh)
        		# cv2.imshow('image', img_a)
        		time.sleep(5)
        		if cv2.waitKey(1) & 0xFF == ord('q'):
        			break
            # Clear the frames array to start the next batch
        frames = []
        if count == 5:
        	count = 1
        else :
        	count+=1
        # if switch =:
        # 	switch = False
        # else:
        # 	switch = True
        	'''

                idhar |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
            '''
            # temp = np.array(temp)
            # cv2.imshow('frame', temp)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            # 	break
    capture.release()



video = cv2.VideoCapture(os.path.join(VIDEO_DIR, 'Trash_short.mp4'))
# video = cv2.VideoCapture(0)

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if int(major_ver)  < 3 :
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = video.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

video.release();

def make_video(images=None, fps=30, size=None,
               is_color=True, format="FMP4"):
    """
    Create a video from a list of images.
 
    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
 
    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter('project1.mp4', fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    
import glob
import os

# Directory of images to run detection on
# ROOT_DIR = os.getcwd()
# VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
# VIDEO_SAVE_DIR = os.path.join(VIDEO_DIR, "save_short")
images = list(glob.iglob(os.path.join(VIDEO_SAVE_DIR, '*.*')))
print(len(images))
# Sort the images by integer index
images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))
make_video(images, fps=30)
