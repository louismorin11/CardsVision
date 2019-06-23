import numpy as np
import argparse
import os
import random
import cv2
import imutils
import sys


IMG_HEIGHT = 500
CARD_THRESH = 25
CORNER_WIDTH = 32
CORNER_HEIGHT = 54
PERI_MIN = 200
CARD_HEIGHT = 300
CARD_WIDTH = 200



suit_path = '/home/louis/projects/vision/suits'
rank_path = '/home/louis/projects/vision/ranks'


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be analysed")
ap.add_argument("-s", "--saving", required = False,
	help = "Saving result (for initial setup)")
args = vars(ap.parse_args())
saving = args["saving"]


def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_center(pts):
	pts = order_points(pts)
	(top_left, top_right, bottom_right, bottom_left) = pts
	center_x = int((top_left[0] + top_right[0]) * 0.5)
	center_y = int((top_left[1] + bottom_left[1]) * 0.5)
	return (center_x,center_y)




#from the 4 points defining the card, returns an image of the card from the front view
def get_front_view(image, pts):
	rect = order_points(pts)
	(top_left, top_right, bottom_right, bottom_left) = rect
	widthA = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
	widthB = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
	heightA = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
	heightB = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
	dst = np.array([[0, 0],[CARD_WIDTH - 1, 0],[CARD_WIDTH - 1,CARD_HEIGHT - 1],[0,CARD_HEIGHT - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (CARD_WIDTH, CARD_HEIGHT))
	return warped


#load the sample ranks
ranks_imgs = []
ranks_names = []
valid_images = [".jpg",".gif",".png",".tga"]
for f in os.listdir(rank_path):
	ext = os.path.splitext(f)[1]
	if ext.lower() not in valid_images:
		continue

	ranks_names.append(os.path.join(rank_path, f).split('/')[-1].split('.')[0])
	ranks_imgs.append(cv2.cvtColor(cv2.imread(os.path.join(rank_path,f)), cv2.COLOR_BGR2GRAY))

#load the sample suits
suits_imgs = []
suits_names = []
for f in os.listdir(suit_path):
	ext = os.path.splitext(f)[1]
	if ext.lower() not in valid_images:
		continue
	suits_names.append(os.path.join(suit_path, f).split('/')[-1].split('.')[0])
	suits_imgs.append(cv2.cvtColor(cv2.imread(os.path.join(suit_path,f)), cv2.COLOR_BGR2GRAY))

image = cv2.imread(args["image"])
if image is None:
	sys.exit("Not an image")
ratio = image.shape[0] / IMG_HEIGHT
orig = image.copy()
image = imutils.resize(image, height = IMG_HEIGHT)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #we convert the image to a gray image
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edge = cv2.Canny(gray, 75, 200) #we get the edges out of the photo

#we find the contours in the edge photo
cnts = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

rectangles_found = []
approx_found = []

#we choose in the set of contours the ones that are cards
for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	if len(approx) == 4 and peri>400:
		new = True
		for old_approx in approx_found:
			if(abs(old_approx[0][0][0]-approx[0][0][0])<2 or abs(old_approx[0][0][1]-approx[0][0][1])<2):
				new = False
		if(new):
			rectangles_found.append(c)
			approx_found.append(approx)


imgs = []
ranks = []
suits = []

#from the image of the rank, return the name of the rank
def compute_rank(rank):
	best_match_count = 0
	best_match = ranks_imgs[0]
	cv2.imshow('rank'+str(ranks_imgs[0]),rank)
	for i in range(len(ranks_imgs)):
		#lets compute the difference between rank and ranks_imgs[î]
		diff = np.subtract(ranks_imgs[i], rank)
		res = np.count_nonzero(diff==0)
		if (res > best_match_count):
			best_match_count = res
			best_match = ranks_names[i]
	return best_match

#from the image of the suit, return  the name of the suit
def compute_suit(suit):
	best_match_count = 0
	best_match = suits_names[0]
	cv2.imshow('suit'+suits_names[0],suit)
	cv2.waitKey(0)
	for i in range(len(suits_imgs)):
		#lets compute the difference between suit and suits_imgs[i]
		diff = np.subtract(suits_imgs[i], suit)
		res = np.count_nonzero(diff==0)
		if (res > best_match_count):
			best_match_count = res
			best_match = suits_names[i]
	return best_match

#from the top corner of the card, returns the name of the rank
def find_rank(rank, image, path):
	#we find the biggest contour on the image, it's the rank  (problem with 10, we only see the 0)
	conts = cv2.findContours(rank.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	conts = conts[0] if imutils.is_cv2() else conts[1]
	conts = sorted(conts, key = cv2.contourArea, reverse = True)
	if len(conts)!=0:
		x,y,w,h = cv2.boundingRect(conts[0])
		t = cv2.rectangle(rank,(x,y),(x+w,y+h),(0,255,0),2)
		f_rank = rank[y:y+h,x:x+w]
		#we resize the image so that it is a perfect zoom on the rank
		f_rank = cv2.resize(f_rank, (200,400))
		#for init
		if (saving):
			path = rank_path + '/' +path.split('/')[-1]
			print(path)
			cv2.imwrite(path, f_rank)
		return compute_rank(f_rank)

#from the top corner of the card, returns the name of the suit
def find_suit(suit, image, path):
	#we find the biggest contour on the image, it's the suit
	conts = cv2.findContours(suit.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	conts = conts[0] if imutils.is_cv2() else conts[1]
	conts = sorted(conts, key = cv2.contourArea, reverse = True)
	if len(conts)!=0:
		x,y,w,h = cv2.boundingRect(conts[0])
		t = cv2.rectangle(suit,(x,y),(x+w,y+h),(0,255,0),2)
		f_suit = suit[y:y+h,x:x+w]
		f_suit = cv2.resize(f_suit, (400,400))
		if saving:
			path = suit_path + '/' + path.split('/')[-1]
			print(path)
			cv2.imwrite(path, f_suit)
		return compute_suit(f_suit)

#from a card, return the rank and the suit on the card
def find_rank_suit(approx, image, card, path):
	wraped = get_front_view(image, approx.reshape(4,2)) #we transform the image to get a front view of the card
	corner=wraped[0:CORNER_HEIGHT, 0:CORNER_WIDTH] #we select its top left corner
	corner_zoom = cv2.resize(corner, (0,0), fx=4, fy=4)
	corner_zoom = cv2.cvtColor(corner_zoom,cv2.COLOR_BGR2GRAY)
	white_level = corner_zoom[15,int(CORNER_WIDTH*2)] #this value should be a white spot, we use it to tresh
	thresh_level = white_level - CARD_THRESH
	if (thresh_level <= 0):
		thresh_level = 1
	res, threshed = cv2.threshold(corner_zoom, thresh_level, 255, cv2. THRESH_BINARY_INV)

	rank = threshed[10:150, 0:115]
	suit = threshed[115:336, 0:115]
	imgs.append(corner_zoom)
	ranks.append(rank)
	suits.append(suit)
	r = (find_rank(rank, image, path))
	s = (find_suit(suit, image, path))
	if r is None or s is None:
		return 'No match'
	return r+' de '+s


#print('Found '+str(len(rectangles_found))+' cards')
#for each card, find the rank on the suit
for i in range(len(rectangles_found)):
	s=(find_rank_suit(approx_found[i], image, rectangles_found[i], args["image"]))
	if s!= "No match":
		print(s)
		c = find_center(approx_found[i].reshape(4,2))
		font = cv2.FONT_HERSHEY_SIMPLEX
		textsize = cv2.getTextSize(s, font, 1, 2)[0]
		textX = (c[0] - (textsize[0]//2))
		textY = (c[1] - (textsize[1]//2))
		cv2.putText(image,s,(textX,textY), font, 1,(255,255,0),2,cv2.LINE_AA)
		cv2.imshow("res", image)










cv2.waitKey(0)
cv2.destroyAllWindows()
