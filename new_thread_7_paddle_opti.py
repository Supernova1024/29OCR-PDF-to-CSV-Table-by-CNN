import cv2
import numpy as np
import time
import csv
import os
from multiprocessing import Pool
import statistics
from paddleocr import PaddleOCR,draw_ocr
from scipy import ndimage
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

input_folder = "pdf_img/"
output_folder = "output"
out_csv = "table_1.csv"
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False)


def uuid(filename):
    time_str = str(int(round(time.time() * 1000)))
    file = filename.split(".pdf")[0]
    id = file + "&&&" + time_str
    return id


def box_processing(img, lang):
    if lang == 'arabic':
        img = img[0:img.shape[0], int(img.shape[1]/2):img.shape[1]]

    # kernel = np.ones((1, 1), np.uint8)
    # dilateimg = cv2.dilate(img, kernel, iterations=30)

    # kernel1 = np.ones((3, 3), np.uint8)
    # erodeimg = cv2.erode(dilateimg, kernel1, iterations=1)

    # kernel = np.ones((3, 3), np.uint8)
    # dilateimg = cv2.dilate(erodeimg, kernel, iterations=1)
    return img

def sort_contours(cnts, method="left-to-right"):
    try:
        # initialize the reverse flag and sort index
        reverse = False
        i = 0

        # handle if we need to sort in reverse
        if method in ["right-to-left", "bottom-to-top"]:
            reverse = True

        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method in ["top-to-bottom", "bottom-to-top"]:
            i = 1

        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        cnts, boundingBoxes = zip(*sorted(zip(cnts, boundingBoxes),
                                          key=lambda b: b[1][i], reverse=reverse))

        # return the list of sorted contours and bounding boxes
        return cnts, boundingBoxes
    except Exception as e:
        print(f"Error: {e}")


def draw_blue_line(img, rect_barcode):
    try:
        img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_cny = cv2.Canny(img_gry, 150, 200)
        lns = cv2.ximgproc.createFastLineDetector().detect(img_cny)
        img_cpy = img.copy()

        if len(rect_barcode) > 0:
            if lns is not None:
                for ln in lns:
                    x1 = int(ln[0][0])
                    y1 = int(ln[0][1])
                    x2 = int(ln[0][2])
                    y2 = int(ln[0][3])
                    cv2.line(img_cpy, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=5)
                return (1, img_cpy)

        elif lns is not None:
            for ln in lns:
                x1 = int(ln[0][0])
                y1 = int(ln[0][1])
                x2 = int(ln[0][2])
                y2 = int(ln[0][3])
                cv2.line(img_cpy, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=5)
            return (1, img_cpy)

        return (0, img_cpy)

    except Exception as e:
        print(f"Error: {e}")


def v_remove_cnts(image, rect_barcode):
    try:
        # Initialize variables
        limit_distance = 200
        mask = np.ones(image.shape[:2], dtype="uint8") * 255

        # Find contours and sort by height
        contours, hierarchy = cv2.findContours(
            image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        h_arr = [cv2.boundingRect(cnt)[3] for cnt in contours]
        h_arr.sort()

        # Cluster contours based on height
        current_cluster = []
        clusters = []
        for i in range(len(h_arr)):
            if not current_cluster or h_arr[i] - current_cluster[-1] <= limit_distance:
                current_cluster.append(h_arr[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [h_arr[i]]
        clusters.append(current_cluster)

        # Find the largest cluster and average height
        largest_cluster = max(clusters, key=len)
        avg_value = sum(largest_cluster) / len(largest_cluster)

        # Remove contours outside of the largest cluster
        for cnt in contours:
            h = cv2.boundingRect(cnt)[3]
            if h < avg_value - limit_distance or h > avg_value + limit_distance:
                cv2.drawContours(mask, [cnt], -1, 0, -1)
        image = cv2.bitwise_and(image, image, mask=mask)

        # Draw first vertical line if barcode exists
        if len(rect_barcode) > 0:
            x_barcode_center = rect_barcode[0][0]
            y_barcode_center = int(rect_barcode[0][1])
            w_barcode = rect_barcode[1][0]
            x_first_vertical = int(x_barcode_center - 350/2 - 8)
            cv2.line(image, pt1=(x_first_vertical, y_barcode_center), pt2=(x_first_vertical, image.shape[0]), color=(255, 0, 0), thickness=10)

        return (1, image)

    except Exception as e:
        print(f"Error in v_remove_cnts: {str(e)}")
        return (0, image)


def h_remove_cnts(image):
    try:
        mask = np.ones(image.shape[:2], dtype="uint8") * 255
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < image.shape[1]:
                cv2.drawContours(mask, [cnt], -1, 0, -1)
        image = cv2.bitwise_and(image, image, mask=mask)
        return image
    except Exception as e:
        print(f"Error occurred in h_remove_cnts: {e}")
        return None


def find_0(s):
    ch = "0"
    return [i for i, ltr in enumerate(s) if ltr == ch]

def img_rotate(img, rect_barcode):
    try:
        angle = rect_barcode[2]
    except IndexError:
        return img
    
    # Rotate the original image based on the angle of the polygon
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    if angle > 45:
        angle1 = angle - 90 - 0.8
    else:
        angle1 = angle
    M = cv2.getRotationMatrix2D(center, angle1, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


def find_barcode(img, file_path):
    try:
        time_str = str(int(round(time.time() * 1000)))
        resizeimg = img[100:600, 20:800]
        kernel = np.ones((1, 1), np.uint8)
        dilateimg = cv2.dilate(resizeimg, kernel, iterations=3)

        kernel1 = np.ones((1, 7), np.uint8)
        erodeimg = cv2.erode(dilateimg, kernel1, iterations=5)

        kernel2 = np.ones((7, 7), np.uint8)
        dilateimg = cv2.dilate(erodeimg, kernel2, iterations=7)

        kernel1 = np.ones((7, 7), np.uint8)
        erodeimg = cv2.erode(dilateimg, kernel1, iterations=7)

        gray = cv2.cvtColor(erodeimg, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to the grayscale image to get a binary image
        _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)

        # Find the contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect_barcode = []
        if len(contours) > 0:
            # Filter the contours to find the box contour based on its shape and size
            for contour in contours:
                # Approximate the contour to a polygon
                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

                # Check if the polygon has 4 vertices (i.e., is a quadrilateral) and has a certain minimum area
                if len(approx) == 4 and cv2.contourArea(approx) < 25000 and cv2.contourArea(approx) > 8000:
                    # Draw the contour on the original image
                    cv2.drawContours(erodeimg, [approx], 0, (0, 0, 255), 2)
                    # Calculate the angle of the polygon
                    rect_barcode = cv2.minAreaRect(approx)
                    break
            if not rect_barcode:
                print("Barcode not found. ", file_path)
        else:
            print("Barcode not found. ", file_path)
        return rect_barcode
    except Exception as e:
        print(f"Error: {str(e)}")
        return rect_barcode


def reorganize_array(arr):
    # Making 3 dimensional array based on big Y difference
    # Making the length of sub array less than 7
    group = []
    subarray = [arr[0]]

    try:
        for i in range(1, len(arr)):
            if arr[i][1] - arr[i-1][1] > 40:
                subarray.sort()
                if len(subarray) >= 7:
                    subarray = subarray[:7]
                group.append(subarray)
                subarray = []
            subarray.append(arr[i])

        subarray.sort()
        if len(subarray) >= 7:
            subarray = subarray[:7]
        group.append(subarray)
    except Exception as e:
        print(f"Error in reorganize_array: {e}")
        return []

    return group


# Table rotation
def correct_skew(image, delta=2, limit=7):
    def determine_score(arr, angle):
        data = ndimage.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

        scores = []
        angles = np.arange(-limit, limit + delta, delta)
        for angle in angles:
            histogram, score = determine_score(thresh, angle)
            scores.append(score)

        best_angle = angles[scores.index(max(scores))]

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
                borderMode=cv2.BORDER_REPLICATE)

        return best_angle, corrected
    except Exception as e:
        print("Error: ", e)
        return None, None



#Functon for extracting the box
def box_extraction(img_for_box_extraction_path):
    print("==filename=====", img_for_box_extraction_path)
    try:
        # Split filename and extension
        filename_no_xtensn, xtensn = os.path.splitext(img_for_box_extraction_path)
        
        # Check if extension is valid
        if xtensn.lower() not in ['.jpg', '.jpeg']:
            raise ValueError("Invalid image file extension")
            
        # Get PDF name from filename
        filename_no_folder = filename_no_xtensn.split(input_folder)[1]
        pdf_name = filename_no_folder.split("&&&")[0]
        
        # Create output directory if it doesn't exist
        output_dir = output_folder + "_" + pdf_name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load image and find barcode
        output_img_path = output_folder + "_" + pdf_name + "/"
        img_origin = cv2.imread(img_for_box_extraction_path)
        rect_barcode = find_barcode(img_origin, img_for_box_extraction_path)
        
        # Rotate image if barcode is found
        if len(rect_barcode) > 0:
            print("=========found barcode=====", img_for_box_extraction_path)
            img = img_rotate(img_origin, rect_barcode)
            rect_barcode = find_barcode(img, img_for_box_extraction_path)
            
            # Draw blue line on barcode
            green_flag, img1 = draw_blue_line(img, rect_barcode)
            
            if green_flag == 1:
                print("===1======green_flag == 1=====", img_for_box_extraction_path)
                # Threshold and binarize image
                Cimg_gray_para = [3, 3, 0]
                Cimg_blur_para = [150, 255]
                gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                blurred_img = cv2.GaussianBlur(gray_img, (Cimg_gray_para[0], Cimg_gray_para[1]), Cimg_gray_para[2])
                thresh, img_bin = cv2.threshold(blurred_img, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                img_bin = 255 - img_bin
                
                # Detect vertical and horizontal lines
                kernel_length = np.array(img).shape[1] // 40
                verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
                hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                
                # Morphological operations to detect lines
                img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=2)
                verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=20)
                verticle_lines_img = cv2.erode(verticle_lines_img, verticle_kernel, iterations=2)
                cnts_flag, v_cnts_img = v_remove_cnts(verticle_lines_img, rect_barcode)
                
                if cnts_flag == 1:
                    print("====cnts_flag == 1====", img_for_box_extraction_path)
                    # Morphological operation to detect horizontal lines from an image
                    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=2)
                    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=17)
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
                    cv2.waitKey(0)

                    # Find contours for image, which will detect all the boxes
                    contours, hierarchy = cv2.findContours(
                        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # Sort all the contours by top to bottom.
                    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

                    boxes = []
                    xx = []
                    yy = []
                    ww = []
                    hh = []
                    areaa = []

                    for c in contours:
                        # Returns the location and width,height for every contour
                        x, y, w, h = cv2.boundingRect(c)
                        if x > 3 and w > 800 and (x + w) < 2000 and y > 50 and y+h < 3650 and h > 55 and w < 1000 and h < 100:
                            xx.append(x)

                    first_column_x = 131
                    if len(xx) > 0:
                        first_column_x = statistics.mode(xx)
                    if first_column_x > 130:
                        for c in contours:
                            # Returns the location and width,height for every contour
                            x, y, w, h = cv2.boundingRect(c)
                            area = w * h
                            box_info = [x, y, w, h, area]
                            # print(box_info)
                            image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 255, 0), 5)
                            if x > 100 and x < (img1.shape[0] - 70) and (x + w) < (img1.shape[1] - 10) and y > 50 and y+h < 3650 and w > 100 and h > 55 and w < 1000 and h < 200:
                                # image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 0, 255), 25)
                                boxes.append(box_info)
                            elif x > 3 and w > 800 and (x + w) < 2050 and y > 40 and y+h < 3650 and h > 40 and w < 1000 and h < 200:
                                # image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 0, 255), 25)
                                boxes.append(box_info)
                            elif x > 500 and x < 1500 and w > 40 and y > 50 and y+h < 3650 and h > 40 and w < 1000 and h < 200:
                                # image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 0, 255), 25)
                                boxes.append(box_info)
                    else:
                        for c in contours:
                            # Returns the location and width,height for every contour
                            x, y, w, h = cv2.boundingRect(c)
                            area = w * h
                            box_info = [x, y, w, h, area]
                            # print(box_info)
                            image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 255, 0), 5)
                            if x > 100 and x < (img1.shape[0] - 70) and (x + w) < (img1.shape[1] - 100) and y > 50 and y+h < 3650 and w > 100 and h > 55 and w < 1000 and h < 200:
                                # image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 0, 255), 25)
                                boxes.append(box_info)
                            elif x > 3 and w > 800 and (x + w) < 2050 and y > 50 and y+h < 3650 and h > 40 and w < 1000 and h < 200:
                                # image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 0, 255), 25)
                                boxes.append(box_info)
                            elif x > 500 and x < 1500 and w > 40 and y > 50 and y+h < 3650 and h > 40 and w < 1000 and h < 200:
                                # image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 0, 255), 25)
                                boxes.append(box_info)

                    boxes_sorted_y = sorted(boxes, key=lambda x: x[1])
                    box_array = boxes_sorted_y

                    if len(box_array) > 0:
                        row_columns = reorganize_array(box_array)
                    else:
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

                    for sub_boxes  in row_columns:
                        for box  in sub_boxes:
                            x = box[0]
                            y = box[1]
                            w = box[2]
                            h = box[3]
                            image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 0, 255), 25)

                    ## Write red rect image
                    time_str = str(int(round(time.time() * 1000)))
                    w_filename = 'cnts/' +filename_no_folder+ '_' + '.png'
                    cv2.imwrite(w_filename, image)
                    
                    idx = 0
                    csv_row_col = []
                    col = 0
                    w_filename_base = output_img_path + filename_no_folder+ '_'
                    for columns in row_columns:
                        csv_cols = []
                        if col == 0:
                            row = 0
                            for box in columns:

                                idx += 1
                                new_img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                                time_str = str(int(round(time.time() * 1000)))
                                if row == 0:
                                    w_filename = w_filename_base +str(idx) +'_Address.png'
                                if row == 3:
                                    w_filename = w_filename_base +str(idx) +'_Guardian.png'
                                if row == 4:
                                    w_filename = w_filename_base +str(idx) +'_Name.png'
                                cv2.imwrite(w_filename, new_img)
                                csv_cols.append(filename_no_xtensn+ '_' +time_str+ '_' +str(idx) + '.png')
                                row += 1
                        else:
                            row = 0
                            for box in columns:
                                if row  == 0 or row == 3 or row == 4:
                                    idx += 1
                                    new_img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                                    if row == 0:
                                        w_filename = w_filename_base +str(idx) +'_Address.png'
                                    if row == 3:
                                        w_filename = w_filename_base +str(idx) +'_Guardian.png'
                                    if row == 4:
                                        w_filename = w_filename_base +str(idx) +'_Name.png'
                                    cv2.imwrite(w_filename, new_img)
                                    csv_cols.append(w_filename.split(output_img_path)[1])
                                else:
                                    idx += 1
                                    new_img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]

                                    if row  == 1:
                                        processed_box = box_processing(new_img, "arabic")
                                    else:
                                        processed_box = box_processing(new_img, "number")
                                    
                                    result = ocr.ocr(processed_box, cls=False)
                                    txts = [line[1][0] for line in result[0]]
                                    if len(txts) == 0:
                                        data = ''
                                    else:
                                        data = txts[0]
                                    csv_cols.append(data)
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    cv2.putText(img, data, (box[0],box[1]), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                                row += 1
                            # Add page number to last column
                            csv_cols.append(filename_no_folder.split("-")[1].split(".")[0])
                        csv_row_col.append(csv_cols)
                        col += 1
                    with open(pdf_name + ".csv", 'a', newline='') as f:
                        writer = csv.writer(f, delimiter=',')
                        writer.writerows(csv_row_col)  #considering my_list is a list of lists.
                else:
                    print("no table in ", output_img_path+filename_no_folder)

                ## Write ocr result to the image
                w_filename = 'text/' +filename_no_folder+ '_' + '.png'
                cv2.imwrite(w_filename, img)
            
        else:
            print("=========no found barcode=====")
            angle, img = correct_skew(img_origin)
            img1 = img  # Read the image
            rect_barcode = []
            (green_flag, img1) = draw_blue_line(img1, rect_barcode)

            if green_flag == 1:
                print("=========green_flag == 1=====")
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
                (cnts_flag, v_cnts_img) = v_remove_cnts(verticle_lines_img, rect_barcode)

                if cnts_flag == 1:
                    # Morphological operation to detect horizontal lines from an image
                    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=2)
                    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=17)
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

                    boxes = []
                    xx = []
                    yy = []
                    ww = []
                    hh = []
                    areaa = []

                    for c in contours:
                        # Returns the location and width,height for every contour
                        x, y, w, h = cv2.boundingRect(c)
                        if x > 3 and w > 800 and (x + w) < 2000 and y > 50 and y+h < 3650 and h > 55 and w < 1000 and h < 100:
                            xx.append(x)

                    first_column_x = 131
                    if len(xx) > 0:
                        first_column_x = statistics.mode(xx)
                    if first_column_x > 130:
                        for c in contours:
                            # Returns the location and width,height for every contour
                            x, y, w, h = cv2.boundingRect(c)
                            area = w * h
                            box_info = [x, y, w, h, area]
                            # print(box_info)
                            image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 255, 0), 5)
                            if x > 100 and x < (img1.shape[0] - 70) and (x + w) < (img1.shape[1] - 10) and y > 50 and y+h < 3650 and w > 100 and h > 55 and w < 1000 and h < 200:
                                # image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 0, 255), 25)
                                boxes.append(box_info)
                            elif x > 3 and w > 800 and (x + w) < 2050 and y > 40 and y+h < 3650 and h > 40 and w < 1000 and h < 200:
                                # image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 0, 255), 25)
                                boxes.append(box_info)
                            elif x > 500 and x < 1500 and w > 40 and y > 50 and y+h < 3650 and h > 40 and w < 1000 and h < 200:
                                # image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 0, 255), 25)
                                boxes.append(box_info)
                    else:
                        for c in contours:
                            # Returns the location and width,height for every contour
                            x, y, w, h = cv2.boundingRect(c)
                            area = w * h
                            box_info = [x, y, w, h, area]
                            # print(box_info)
                            image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 255, 0), 5)
                            if x > 100 and x < (img1.shape[0] - 70) and (x + w) < (img1.shape[1] - 100) and y > 50 and y+h < 3650 and w > 100 and h > 55 and w < 1000 and h < 200:
                                # image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 0, 255), 25)
                                boxes.append(box_info)
                            elif x > 3 and w > 800 and (x + w) < 2050 and y > 50 and y+h < 3650 and h > 40 and w < 1000 and h < 200:
                                # image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 0, 255), 25)
                                boxes.append(box_info)
                            elif x > 500 and x < 1500 and w > 40 and y > 50 and y+h < 3650 and h > 40 and w < 1000 and h < 200:
                                # image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 0, 255), 25)
                                boxes.append(box_info)

                    boxes_sorted_y = sorted(boxes, key=lambda x: x[1])
                    box_array = boxes_sorted_y

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

                    for sub_boxes  in row_columns:
                        for box  in sub_boxes:
                            x = box[0]
                            y = box[1]
                            w = box[2]
                            h = box[3]
                            image = cv2.rectangle(img1, (x, y), (x+w, y+h),(0, 0, 255), 25)

                    ## Write red rect image
                    time_str = str(int(round(time.time() * 1000)))
                    w_filename = 'cnts/' +filename_no_folder+ '_' + '.png'
                    # cv2.imwrite(w_filename, image)

                    idx = 0
                    csv_row_col = []
                    col = 0
                    w_filename_base = output_img_path + filename_no_folder+ '_'
                    for columns in row_columns:
                        csv_cols = []
                        if col == 0:
                            row = 0
                            for box in columns:

                                idx += 1
                                new_img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                                time_str = str(int(round(time.time() * 1000)))
                                # w_filename = w_filename_base +time_str+ '_' +str(idx) + '.png'
                                if row == 0:
                                    w_filename = w_filename_base +str(idx) +'_Address.png'
                                if row == 3:
                                    w_filename = w_filename_base +str(idx) +'_Guardian.png'
                                if row == 4:
                                    w_filename = w_filename_base +str(idx) +'_Name.png'
                                cv2.imwrite(w_filename, new_img)
                                csv_cols.append(filename_no_xtensn+ '_' +time_str+ '_' +str(idx) + '.png')
                                row += 1
                        else:
                            row = 0
                            for box in columns:
                                if row  == 0 or row == 3 or row == 4:
                                    idx += 1
                                    new_img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                                    if row == 0:
                                        w_filename = w_filename_base +str(idx) +'_Address.png'
                                    if row == 3:
                                        w_filename = w_filename_base +str(idx) +'_Guardian.png'
                                    if row == 4:
                                        w_filename = w_filename_base +str(idx) +'_Name.png'
                                    cv2.imwrite(w_filename, new_img)
                                    csv_cols.append(w_filename.split(output_img_path)[1])
                                else:
                                    idx += 1
                                    new_img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]

                                    if row  == 1:
                                        processed_box = box_processing(new_img, "arabic")
                                    else:
                                        processed_box = box_processing(new_img, "number")
                                    
                                    result = ocr.ocr(processed_box, cls=False)
                                    txts = [line[1][0] for line in result[0]]
                                    if len(txts) == 0:
                                        data = ''
                                    else:
                                        data = txts[0]
                                    csv_cols.append(data)
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    cv2.putText(img, data, (box[0],box[1]), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                                row += 1
                            # Add page number to last column
                            csv_cols.append(filename_no_folder.split("-")[1].split(".")[0])
                        csv_row_col.append(csv_cols)
                        col += 1
                    with open(pdf_name + ".csv", 'a', newline='') as f:
                        writer = csv.writer(f, delimiter=',')
                        writer.writerows(csv_row_col)  #considering my_list is a list of lists.
                else:
                    print("no table in ", output_img_path+filename_no_folder)
            else:
                print("no table in ", output_img_path+filename_no_folder)

            ## Write ocr result to the image
            w_filename = 'text/' +filename_no_folder+ '_' + '.png'
            cv2.imwrite(w_filename, img)

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print("An unexpected error occurred:", e)
    finally:
        cv2.destroyAllWindows()


def process_files(files):
    for filename in files:
        file_path = os.path.join(input_folder, filename)
        try:
            box_extraction(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


def main_threading():
    filenames = os.listdir(input_folder)
    if len(filenames) < 5:
        for filename in filenames:
            file_path = os.path.join(input_folder, filename)
            try:
                box_extraction(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    else:
        count_processes = 5
        step = len(filenames) // count_processes
        file_groups = [filenames[i*step:(i+1)*step] for i in range(count_processes)]
        if len(filenames) % count_processes != 0:
            file_groups[-1].extend(filenames[count_processes*step:])
        with Pool(count_processes) as p:
            p.map(process_files, file_groups)


def main_single():
    stop_threads = False 
    for filename in os.listdir(input_folder):
        file_path = input_folder + filename
        box_extraction(file_path)


if __name__ == '__main__':
    print("Reading image..")
    main_threading()