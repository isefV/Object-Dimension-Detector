import cv2 as cv
import numpy as np

#for number in range(count):

number = str(215)
image_num = "0000"
image_num = image_num[:4-len(number)] + number
input_filename = f"img/NGR_{image_num}.JPG"

# Read Image
img = cv.imread(input_filename)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Get height And width
base_height,base_width = img.shape

# Create New Blank Image
blank_img = np.zeros((610,370,3),np.uint8)

# Rotate And Fit Image
rot_mat = cv.getRotationMatrix2D((base_width//2,base_height//2),0.8,1)
img = cv.warpAffine(img,rot_mat,(base_width,base_height),flags=cv.INTER_LINEAR)
img = cv.resize(img,(0,0),fx=0.2,fy=0.2)

# Crop The Center Of Image
img = img[140:750 ,400:770]

# Set Two Threshold Filter To Detect Object
ret , thresh1 = cv.threshold(img,120,255,cv.THRESH_TOZERO)
ret , thresh2 = cv.threshold(thresh1,180,255,cv.THRESH_BINARY)

# This Filter For Remove Outter Noises In Image
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh2,cv.MORPH_OPEN,kernel)

# This Filter For Filled Object 
kernel = np.ones((7,7),np.uint8)
dilated = cv.dilate(opening,kernel,iterations = 2)

# This Filter For Remove Inner Noises
kernel = np.ones((5,5),np.uint8)
closing = cv.morphologyEx(dilated,cv.MORPH_CLOSE,kernel)

# Get The Objects Coordinates
contours , hierarchy = cv.findContours(closing,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

# Detect The Main Object
cnt = max(contours,key=cv.contourArea)

# Highlighting The Main Object
x,y,w,h = cv.boundingRect(cnt)
cv.drawContours(blank_img,contours,-1,(0,255,0),2)
cv.rectangle(blank_img,(x,y),(x+w,y+h),(0,0,255),2)

# Detect Coordinates Of Two Edge Side Object
left , right = [],[]
offset = 40
for item in cnt:
    if ((y + offset) < item[0,1]) and ((y + h - offset) > item[0,1]):
        if item[0,0] < (x + (w // 2)):
            left.append(item)
        else:
            right.append(item)

left = np.array(left)
right = np.array(right)

# Draw Two Edge Side Of Object
for point in range(1,len(left)):
    x1,y1 = left[point-1].ravel()
    x2,y2 = left[point].ravel()
    cv.line(blank_img,(x1,y1) ,(x2,y2),(255,0,0),3)

for point in range(1,len(right)):
    x1,y1 = right[point-1].ravel()
    x2,y2 = right[point].ravel()
    cv.line(blank_img,(x1,y1) ,(x2,y2),(255,0,0),3)


# Calculate Width Of Two Edge Side Object
width_list = []
count_point = min(len(left),len(right))
#for point in range(count_point):
#    x_left,y_left = left[point].ravel()
#    x_right,y_right = right[count_point - 1 - point].ravel()
#    width_list.append( abs( x_right - x_left) )
#    if abs(y_right - y_left) <= 2 :
#        cv.line(blank_img,(x_left,y_left) ,(x_right,y_right),(255,255,255),1)

len_right = len(right)
len_left = len(left)

for point_left in range(len_left):
    x_left , y_left = left[point_left].ravel()

    for point_right in range(len_right-1,0,-1):
        x_right , y_right = right[point_right].ravel()
        
        if y_left == y_right: # or abs(y_right - y_left) < 3 :
            cv.line(blank_img,(x_left,y_left) ,(x_right,y_right),(255,255,255),1)
            width_list.append(abs(x_right - x_left))
            break
        elif abs(y_right - y_left) > 5 :
            len_right = len_right - 3
            break
        continue



with open(f"list-of-widths-{image_num}.txt",'w') as file:
    file.write(str(width_list))

# Display Images For Debug
#cv.imshow('Image',img)
#cv.imshow('Threshold 1',thresh1)
#cv.imshow('Threshold 2',thresh2)
#cv.imshow('Dilated',dilated)
#cv.imshow('Opening',opening)
#cv.imshow('Closing',closing)
cv.imshow('Blank',blank_img)
cv.imwrite(f"export/image-{image_num}.jpg",blank_img)
cv.waitKey(0)
