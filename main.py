import os
import sys
import cv2 as cv
import numpy as np
import pandas as pd

# Folder setting initilizing
foldername = "101D5600/"
widths_list = []
list_dir = list(os.listdir(foldername))
total_image = len(list_dir)
list_dir.sort()

# Run code for every photo in folder
for image_index,image_name in enumerate(list_dir):
    image_num = image_name.split('_')[1][:4]
    input_filename = f"{foldername}/{image_name}"
   
    # Read image
    img = cv.imread(input_filename)
    img = img[885:,:]

    # Save the backup of image
    img_real = img.copy()
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Get height and width
    base_height,base_width = img.shape

    # Create new blank image
    blank_img = np.zeros((base_height,base_width),np.uint8)

    # Set threshold filter to detect object
    ret , thresh2 = cv.threshold(img,180,255,cv.THRESH_BINARY_INV)

    # Set the morphologyEx.MORPH_OPEN filter for remove outter noises in image
    kernel = np.ones((15,15),np.uint8)
    opening = cv.morphologyEx(thresh2,cv.MORPH_OPEN,kernel)

    # Set the dilate filter for filled object 
    kernel = np.ones((23,23),np.uint8)
    dilated = cv.dilate(opening,kernel,iterations = 1)

    # Set the morphologyEx.MORPH_CLOSE filter for remove inner noises
    kernel = np.ones((19,19),np.uint8)
    closing = cv.morphologyEx(dilated,cv.MORPH_CLOSE,kernel)

    # Get the objects coordinates
    contours , hierarchy = cv.findContours(closing,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

    # Detect the main object (the bigger one)
    cnt = max(contours,key=cv.contourArea)

    # Highlighting the main object
    cv.drawContours(blank_img,[cnt],0,255,2)

    # Detect the vertical lines
    sobelx = cv.Sobel(blank_img,cv.CV_8U,1,0,ksize = 5)
    kernel = np.ones((5,5),np.uint8)
    lines = cv.dilate(sobelx,kernel,iterations = 3)

    # Get the vertical lines coordinates and sorted them
    contours ,hierarchy = cv.findContours(lines,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    cnt = sorted(contours,key=cv.contourArea,reverse=True)
    left , right = (cnt[0],cnt[1]) if cnt[0][0,0,0] < cnt[1][0,0,0] else (cnt[1],cnt[0])

    # Draw the real image in lower opacity in background
    blank_img = np.zeros((base_height,base_width,3),np.uint8)
    alpha = 0.6
    blank_img = np.uint8(blank_img*alpha + img_real*(1-alpha))

    # Draw the two side vertical edge lines
    cv.drawContours(blank_img,[left],0,(0,255,0),cv.FILLED)
    cv.drawContours(blank_img,[right],0,(0,255,0),cv.FILLED)

    # Calculate the two blue edge lines
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
    # Free memory allocations
    l = None
    r = None

    # Draw the two blue edge lines
    cv.drawContours(blank_img,[left],0,(255,0,0),cv.FILLED)
    cv.drawContours(blank_img,[right],0,(255,0,0),cv.FILLED)

    # Calculate the distance between two blue edge lines
    points = []
    widths = []
    len_right = len(right)
    len_left = len(left)
    current_level = 0
    for point_left in range(len_left):
        x_left , y_left = left[point_left].ravel()
        for point_right in range(current_level,len_right):
            x_right , y_right = right[point_right].ravel()
            if y_left == y_right:   # or abs(y_right - y_left) < 3 :
                widths.append(abs(x_right - x_left))
                points.append( [(x_left,y_left),(x_right,y_right)] )
                current_level = point_right
                break

    # Free memory allocations
    left = None
    right = None

    # Draw the horizontal red lines and print the number of lines in image
    length = len(widths)
    steps = 50
    pixel = ((max_y - min_y) - ((max_y - min_y) % steps ) ) / steps
    final_widths = [f'{image_name}']
    distance = None
    counter = 1
    for index,item in enumerate(points):
        if distance == None or abs(item[0][1] - distance) == pixel:
            cv.line(blank_img,item[0],item[1],(0,0,255),2)
            cv.putText(blank_img,f'{counter}',(500 , item[0][1]),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv.LINE_AA)
            final_widths.append(widths[index])
            distance = item[0][1]
            if counter == steps:
                break
            counter += 1

    # Free memory allocations
    widths_list.append(final_widths)
    widths = None
    points = None
    
    # Print the data of image
    cv.putText(blank_img,f'Time : N/A ns' ,(20 , 100),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv.LINE_AA)
    cv.putText(blank_img,f'Name : {image_num}',(20 , 150),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv.LINE_AA)
    cv.putText(blank_img,f'min VLine: {max_y - min_y} px',(20 , 200),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv.LINE_AA)
    cv.putText(blank_img,f'Step : {steps}',(20 , 250),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv.LINE_AA)
    cv.putText(blank_img,f'Gap : {pixel} px',(20 , 300),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv.LINE_AA)

    # Save the result of image
    with open(f"exported_data/data-{image_num}.txt",'w') as file:
        file.write(str(final_widths))
        file.write(f'\nLength of Vertical Line : {max_y - min_y} px')
        file.write(f'\nGap : {pixel} px')

    final_widths = None
    
    # Save the image
    # Display Images For Debug
    #cv.imshow('Image',img)
    #cv.imshow('Threshold 2',thresh2)
    #cv.imshow('Dilated',dilated)
    #cv.imshow('Opening',opening)
    #cv.imshow('Closing',closing)
    #cv.imshow(f'{image_name}',blank_img)
    #cv.imshow('Sobel X',sobelx)
    #cv.imshow('Lines',lines)
    cv.imwrite(f"exported_image/image-{image_num}.jpg",blank_img)
    #cv.waitKey(0)
    
    # Show the process of detector
    os.system('clear')
    percent = int(round(image_index / total_image,2) * 100) // 10
    print(f"Image Name : {image_name}\n")
    print(percent*'#',(10 - percent)*' ',f'\t %{percent}')

# Export the csv result data
df = pd.DataFrame(widths_list)
print("\n\n- Done !")
df.to_csv('The_Results.csv',index=False)


