import cv2 as cv
import numpy as np

#for number in range(count):

number = str(11)
image_num = "0000"
image_num = image_num[:4-len(number)] + number
input_filename = f"img/NGR_{image_num} copy.jpg"

# Read Image
img = cv.imread(input_filename)
img = img[885:,:]
img_real = img.copy()
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#img = cv.resize(img,(0,0),fx=0.2,fy=0.2)

# Get height And width
base_height,base_width = img.shape

# Create New Blank Image
blank_img = np.zeros((base_height,base_width),np.uint8)

# Rotate And Fit Image
#rot_mat = cv.getRotationMatrix2D((base_width//2,base_height//2),0.8,1)
#img = cv.warpAffine(img,rot_mat,(base_width,base_height),flags=cv.INTER_LINEAR)
#img = cv.resize(img,(0,0),fx=0.2,fy=0.2)

# Crop The Center Of Image
#img = img[140:750 ,400:770]

# Set Two Threshold Filter To Detect Object
#ret , thresh1 = cv.threshold(img,120,255,cv.THRESH_TOZERO)
ret , thresh2 = cv.threshold(img,180,255,cv.THRESH_BINARY_INV)

# This Filter For Remove Outter Noises In Image
kernel = np.ones((15,15),np.uint8)
opening = cv.morphologyEx(thresh2,cv.MORPH_OPEN,kernel)

# This Filter For Filled Object 
kernel = np.ones((23,23),np.uint8)
dilated = cv.dilate(opening,kernel,iterations = 1)

# This Filter For Remove Inner Noises
kernel = np.ones((19,19),np.uint8)
closing = cv.morphologyEx(dilated,cv.MORPH_CLOSE,kernel)

# Get The Objects Coordinates
contours , hierarchy = cv.findContours(closing,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

# Detect The Main Object
cnt = max(contours,key=cv.contourArea)

# Highlighting The Main Object
#x,y,w,h = cv.boundingRect(cnt)
#cv.rectangle(blank_img,(x,y),(x+w,y+h),(0,0,255),2)
cv.drawContours(blank_img,[cnt],0,255,2)

# Detect The Vertical Lines
sobelx = cv.Sobel(blank_img,cv.CV_8U,1,0,ksize = 5)
kernel = np.ones((5,5),np.uint8)
lines = cv.dilate(sobelx,kernel,iterations = 3)

# Get The Vertical Lines Coordinates And Sorted Them
contours ,hierarchy = cv.findContours(lines,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
cnt = sorted(contours,key=cv.contourArea,reverse=True)
left , right = (cnt[0],cnt[1]) if cnt[0][0,0,0] < cnt[1][0,0,0] else (cnt[1],cnt[0])

# Draw The Two Vertical Edge Lines
blank_img = np.zeros((base_height,base_width,3),np.uint8)
alpha = 0.6
blank_img = np.uint8(blank_img*alpha + img_real*(1-alpha))

cv.drawContours(blank_img,[left],0,(0,255,0),cv.FILLED)
cv.drawContours(blank_img,[right],0,(0,255,0),cv.FILLED)


min_ly = min(left,key=lambda x: x[0,1])[0,1]
max_ly = max(left,key=lambda x: x[0,1])[0,1]
min_ry = min(right,key=lambda x: x[0,1])[0,1]
max_ry = max(right,key=lambda x: x[0,1])[0,1]

min_y = max(min_ly,min_ry)
max_y = min(max_ly,max_ry)

l = []
r = []

for item in left:
    if min_y <= item[0,1] <= max_y:
        l.append(item)

for item in right:
    if min_y <= item[0,1] <= max_y:
        r.append(item)

left = np.array(l)
right = np.array(r)

cv.drawContours(blank_img,[left],0,(255,0,0),cv.FILLED)
cv.drawContours(blank_img,[right],0,(255,0,0),cv.FILLED)

#print(blank_img.shape)

#minmum = left if len(left) < len(right) else right


# Detect Coordinates Of Two Edge Side Object
#left , right = [],[]
#offset = 40
#for item in cnt:
#    if ((y + offset) < item[0,1]) and ((y + h - offset) > item[0,1]):
#        if item[0,0] < (x + (w // 2)):
#            left.append(item)
#        else:
#            right.append(item)


# Draw Two Edge Side Of Object
#for point in range(1,len(left)):
#    x1,y1 = left[point-1].ravel()
#    x2,y2 = left[point].ravel()
#    cv.line(blank_img,(x1,y1) ,(x2,y2),(255,0,0),3)

#for point in range(1,len(right)):
#    x1,y1 = right[point-1].ravel()
#    x2,y2 = right[point].ravel()
#    cv.line(blank_img,(x1,y1) ,(x2,y2),(255,0,0),3)


# Calculate Width Of Two Edge Side Object
#width_list = []
#count_point = min(len(left),len(right))
#for point in range(count_point):
#    x_left,y_left = left[point].ravel()
#    x_right,y_right = right[count_point - 1 - point].ravel()
#    width_list.append( abs( x_right - x_left) )
#    if abs(y_right - y_left) <= 2 :
#        cv.line(blank_img,(x_left,y_left) ,(x_right,y_right),(255,255,255),1)


points = []
widths = []
len_right = len(right)
len_left = len(left)
current_level = 0
#y_left_prev = None
#steps = 20

for point_left in range(len_left):
    x_left , y_left = left[point_left].ravel()
    
#    if y_left_prev != None and ( (min_y > y_left or y_left > max_y) or abs(y_left - y_left_prev) < steps):
#        continue
    
    for point_right in range(current_level,len_right):
        x_right , y_right = right[point_right].ravel()
        
        if y_left == y_right: # or abs(y_right - y_left) < 3 :
            widths.append(abs(x_right - x_left))
            points.append( [(x_left,y_left),(x_right,y_right)] )
            current_level = point_right
            break

#    y_left_prev = y_left


length = len(widths)
steps = 50
pixel = ((max_y - min_y) - ((max_y - min_y) % steps ) ) / steps
print(pixel)

#for vline in range(0,length - (length%100),100):
#pixel = min_y
#while pixel <= max_y:    
final_widths = []
distance = None
counter = 1
for index,item in enumerate(points):
    if distance == None or abs(item[0][1] - distance) == pixel:
        cv.line(blank_img,item[0],item[1],(0,0,255),2)
        cv.putText(blank_img,f'{counter}',(500 , item[0][1]),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv.LINE_AA)
        final_widths.append(widths[index])
        distance = item[0][1]
        if counter == steps:
            break
        counter += 1



cv.putText(blank_img,f'Time : N/A ns' ,(20 , 100),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv.LINE_AA)
cv.putText(blank_img,f'Name : {image_num}',(20 , 150),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv.LINE_AA)
cv.putText(blank_img,f'min VLine: {max_y - min_y} px',(20 , 200),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv.LINE_AA)
cv.putText(blank_img,f'Step : {steps}',(20 , 250),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv.LINE_AA)
cv.putText(blank_img,f'Gap : {pixel} px',(20 , 300),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv.LINE_AA)


with open(f"data-{image_num}.txt",'w') as file:
    file.write(str(final_widths))
    file.write(f'\nLength of Vertical Line : {max_y - min_y} px')
    file.write(f'\nGap : {pixel} px')

# Display Images For Debug
#cv.imshow('Image',img)
#cv.imshow('Threshold 2',thresh2)
#cv.imshow('Dilated',dilated)
#cv.imshow('Opening',opening)
#cv.imshow('Closing',closing)
cv.imshow('Blank',blank_img)
#cv.imshow('Sobel X',sobelx)
#cv.imshow('Lines',lines)
cv.imwrite(f"export/image-{image_num}.jpg",blank_img)
cv.waitKey(0)
