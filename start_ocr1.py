import cv2
import numpy as np
import time
import csv
import pytesseract
import os
import threading
from PIL import Image, ImageOps, ImageFilter 
import imutils
from cnn_ocr1 import predict
## Init
jpegopt={
    "quality": 100,
    "progressive": True,
    "optimize": True
}

global thread_kill_flags
input_folder = "pdf_img/"
output_folder = "output_img/"
out_csv = "table_1.csv"
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def uuid(filename):
    time_str = str(int(round(time.time() * 1000)))
    file = filename.split(".pdf")[0]
    id = file + "&&&" + time_str
    return id


def img_preprocessing(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((1,1),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=4)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    sure_fg = 255 - sure_fg

    return sure_fg


def caculate_time_difference(start_milliseconds, end_milliseconds, filename):
    if filename == 'total':
        diff_milliseconds = int(end_milliseconds) - int(start_milliseconds)
        seconds=(diff_milliseconds / 1000) % 60
        minutes=(diff_milliseconds/(1000*60))%60
        hours=(diff_milliseconds/(1000*60*60))%24
        print("Total run time", hours,":",minutes,":",seconds)
    else:
        diff_milliseconds = int(end_milliseconds) - int(start_milliseconds)
        seconds=(diff_milliseconds / 1000) % 60
        print(seconds, "s", filename)


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def draw_green_line(img):
    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_cny = cv2.Canny(img_gry, 50, 200)
    lns = cv2.ximgproc.createFastLineDetector().detect(img_cny)
    img_cpy = img.copy()
    if lns is not None:
        for ln in lns:
            x1 = int(ln[0][0])
            y1 = int(ln[0][1])
            x2 = int(ln[0][2])
            y2 = int(ln[0][3])
            cv2.line(img_cpy, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=5)
        return (1, img_cpy)
    else:
        return (0, img_cpy)


def v_remove_cnts(image):
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        h_arr = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            h_arr.append(h)
        mode_h = (max(set(h_arr), key = h_arr.count))
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            h_arr.append(h)
            if h < mode_h - 50:
                cv2.drawContours(mask, [cnt], -1, 0, -1)
        image = cv2.bitwise_and(image, image, mask=mask)
        return (1, image)
    else:
        return (0, image)


def h_remove_cnts(image):
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box_info = [x, y, w, h]
        if w < 2640:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    image = cv2.bitwise_and(image, image, mask=mask)
    return image


def find_0(s):
    ch = "0"
    return [i for i, ltr in enumerate(s) if ltr == ch]


def recognition_num(img, i, data):
    dim = (32, 32)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    value = predict(resized)
    if value != "n" and value != "m":
        if value == None:
            time_str = str(int(round(time.time() * 1000)))
            name = "numbers/" + str(time_str) + str(i) + ".jpg"
            cv2.imwrite(name, resized)
            value = "_"
        data += str(value)
    # else:
    #     time_str = str(int(round(time.time() * 1000)))
    #     name = "n_m/" + str(time_str) + str(i) + ".jpg"
    #     cv2.imwrite(name, resized) 
    
    if value == None:
        time_str = str(int(round(time.time() * 1000)))
        name = "numbers/" + str(time_str) + str(i) + ".jpg"
        cv2.imwrite(name, resized)
    return data


def split_number(img):
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    data = ''
    if len(cnts) != 0:
        cnts, boundingBoxes = sort_contours(cnts, method="left-to-right")
        
        if len(cnts) == 1:
            return 0
        else:
            i = 1
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if w > 3 and h > 15 and w < 80 and h < 60:
                    
                    if w > 3 and w < 24:
                        new_img = img[y:y+h, x:x+w]
                        data = recognition_num(new_img, i, data)

                    elif w > 23 and w < 43:
                        new_img1 = img[y:y+h, x:x+int(w/2)+2]
                        data = recognition_num(new_img1, i, data)
                        new_img2 = img[y:y+h, x+int(w/2)-1:x+w]
                        data = recognition_num(new_img2, i + 1000, data)

                    elif w > 42 and w < 65:
                        new_img1 = img[y:y+h, x:x+int(w/3)+2]
                        data = recognition_num(new_img1, i, data)
                        new_img2 = img[y:y+h, x+int(w/3)-1:x+int(w*2/3)+2]
                        data = recognition_num(new_img2, i + 1000, data)
                        new_img3 = img[y:y+h, x+int(w*2/3)-1:x+w]
                        data = recognition_num(new_img3, i + 2000, data)
                    
                    elif w > 64 and w < 80:
                        new_img1 = img[y:y+h, x:x+int(w/4)+2]
                        data = recognition_num(new_img1, i, data)
                        new_img2 = img[y:y+h, x+int(w/4)-1:x+int(w*2/4)+2]
                        data = recognition_num(new_img2, i + 1000, data)
                        new_img3 = img[y:y+h, x+int(w*2/4)-1:x+int(w*3/4)]
                        data = recognition_num(new_img3, i + 2000, data)
                        new_img4 = img[y:y+h, x+int(w*3/4)-1:x+w]
                        data = recognition_num(new_img4, i + 2000, data)
                    
                    i += 1
    return data

#Functon for extracting the box
def box_extraction(img_for_box_extraction_path, cropped_dir_path):
    filename_no_xtensn = img_for_box_extraction_path.split(".")[0]
    xtensn = img_for_box_extraction_path.split(".")[1]
    if xtensn == "jpg" or xtensn == "jpeg" or xtensn == "JPG" or xtensn == "JPEG":
        filename_no_folder = filename_no_xtensn.split(input_folder)[1]
        img = cv2.imread(img_for_box_extraction_path)  # Read the image
        img1 = cv2.imread(img_for_box_extraction_path)  # Read the image
        (green_flag, img1) = draw_green_line(img1)

        if green_flag == 1:
            Cimg_gray_para = [3, 3, 0]
            Cimg_blur_para = [150, 255]

            gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            blurred_img = cv2.GaussianBlur(gray_img, (Cimg_gray_para[0], Cimg_gray_para[1]), Cimg_gray_para[2])
            (thresh, img_bin) = cv2.threshold(blurred_img, 200, 255,
                                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
            img_bin = 255-img_bin  # Invert the image

            # Defining a kernel length
            kernel_length = np.array(img).shape[1]//40
            
            # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
            verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
            # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
            hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
            # A kernel of (3 X 3) ones.
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

            # Morphological operation to detect verticle lines from an image
            img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=2)
            verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=20)
            verticle_lines_img = cv2.erode(verticle_lines_img, verticle_kernel, iterations=2)

            # Find valid cnts
            (cnts_flag, v_cnts_img) = v_remove_cnts(verticle_lines_img)

            if cnts_flag == 1:
                # Morphological operation to detect horizontal lines from an image
                img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=2)
                horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=10)
                horizontal_lines_img = cv2.erode(horizontal_lines_img, hori_kernel, iterations=2)
                
                # Find valid cnts
                h_cnts_img = h_remove_cnts(horizontal_lines_img)

                # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
                alpha = 0.5
                beta = 1.0 - alpha

                # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
                img_final_bin = cv2.addWeighted(v_cnts_img, alpha, h_cnts_img, beta, 0.0)
                img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
                (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                # Find contours for image, which will detect all the boxes
                contours, hierarchy = cv2.findContours(
                    img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # Sort all the contours by top to bottom.
                (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

                ## Find suitable boxes
                boxes = []
                xx = []
                yy = []
                ww = []
                hh = []
                areaa = []
                for c in contours:
                    # Returns the location and width,height for every contour
                    x, y, w, h = cv2.boundingRect(c)
                    area = w * h
                    box_info = [x, y, w, h, area]
                    # print(box_info)
                    image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 255, 0), 5)
                    if x > 50 and x < 2500 and (x + w) < 2640 and y > 50 and w > 70 and h > 55 and w < 1000 and h < 200:
                        image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 0, 255), 25)
                        boxes.append(box_info)
                    elif x > 3 and w > 800 and (x + w) < 2640 and y > 50 and h > 55 and w < 1000 and h < 200:
                        image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 0, 255), 25)
                        boxes.append(box_info)
                boxes_sorted_y = sorted(boxes, key=lambda x: x[1])

                ## Sort boxes by x and make rows
                i = 1
                columns = []
                row_columns = []
                for box in boxes_sorted_y:
                    columns.append(box)
                    if i % 7 == 0:
                        boxes_sorted_x = sorted(columns, key=lambda x: x[0])
                        row_columns.append(boxes_sorted_x)
                        columns = []
                    i += 1

                idx = 0
                csv_row_col = []
                col = 0
                for columns in row_columns:
                    csv_cols = []
                    if col == 0:
                        row = 0
                        for box in columns:
                            idx += 1
                            new_img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                            time_str = str(int(round(time.time() * 1000)))
                            # w_filename = cropped_dir_path+filename_no_folder+ '_' +time_str+ '_' +str(idx) + '.png'
                            if row == 0:
                                w_filename = cropped_dir_path+filename_no_folder+ '_' +str(idx) +'_Address.png'
                            if row == 3:
                                w_filename = cropped_dir_path+filename_no_folder+ '_' +str(idx) +'_Guardian.png'
                            if row == 4:
                                w_filename = cropped_dir_path+filename_no_folder+ '_' +str(idx) +'_Name.png'
                            cv2.imwrite(w_filename, new_img)
                            # csv_cols.append(filename_no_xtensn+ '_' +time_str+ '_' +str(idx) + '.png')
                            row += 1
                    else:
                        row = 0
                        for box in columns:
                            if row  == 0 or row == 3 or row == 4:
                                idx += 1
                                new_img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                                if row == 0:
                                    w_filename = cropped_dir_path+filename_no_folder+ '_' +str(idx) +'_Address.png'
                                if row == 3:
                                    w_filename = cropped_dir_path+filename_no_folder+ '_' +str(idx) +'_Guardian.png'
                                if row == 4:
                                    w_filename = cropped_dir_path+filename_no_folder+ '_' +str(idx) +'_Name.png'
                                cv2.imwrite(w_filename, new_img)
                                csv_cols.append(w_filename.split(cropped_dir_path)[1])
                            else:
                                idx += 1
                                new_img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                                new_img = img_preprocessing(new_img)
                                data = "&&  " + str(split_number(new_img))
                                csv_cols.append(data)
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                cv2.putText(img, data, (box[0],box[1]), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
                            row += 1
                        # Add page number to last column
                        csv_cols.append(filename_no_folder.split("-")[1].split(".")[0])
                    csv_row_col.append(csv_cols)
                    col += 1
                with open(out_csv, 'a', newline='') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerows(csv_row_col)  #considering my_list is a list of lists.

            else:
                print("no table in ", cropped_dir_path+filename_no_folder)
        else:
            print("no table in ", cropped_dir_path+filename_no_folder)


def main(start_time):
    start_total = str(int(round(time.time() * 1000)))
    for filename in os.listdir(input_folder):
        print(filename)
        start_milliseconds = str(int(round(time.time() * 1000)))
        file_path = input_folder + filename
        box_extraction(file_path, output_folder)
        end_milliseconds = str(int(round(time.time() * 1000)))
    end_total = str(int(round(time.time() * 1000)))
    caculate_time_difference(start_total, end_total, "total")    
    

if __name__ == '__main__':
    print("Reading image..")
    start_time = str(int(round(time.time() * 1000)))
    main(start_time)