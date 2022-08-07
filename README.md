# Ocr_pdf_table_csv
This is a project that extracts data from pdf snapshot and enters data into csv
This project is using 2 methods to detect boxes from table.
First method depends on the border of table.
Custom CNN model is used in this project.
You can find the project that is used pretrained Tesseract model on https://github.com/Supernova1024/OCR-Pdf-to-CSV-Table  
Please give me star if this project was helpful to your startup project. :)

The result is stored in "focus_border_Images" folder
![](https://github.com/Supernova1024/OCR-PDF-to-CSV-Table-by-CNN/blob/main/focus_border_img.jpg)
![](https://github.com/Supernova1024/OCR-PDF-to-CSV-Table-by-CNN/blob/main/focus_border_images.png)
  
# Installing
- Download this repository
- Install requirements.txt in project root directory

# How to Run
- Convert pdf to images
  Open the pdf_image.py and define the parameters.
  You can check the parameters here.
  https://pdf2image.readthedocs.io/en/latest/reference.html
- Preparing Datasets and Training the Model.
  Please use this project.
  https://github.com/Supernova1024/train-CNN-model-for-number-classification-OCR
- After getting model by using above project, please copy the model into "models" folder.
- Start
  ```
  python start_ocr1.py
  ```
 
# Result Description
  ![](https://github.com/Supernova1024/OCR-PDF-to-CSV-Table-by-CNN/blob/main/table_1.jpg)
  In this project, I used the table that has 7 columns
  1, 4, and 5 columns can't be recognized by the model.
  The boxes of these columns are stored in "output_img" folder as JPG and added their file name to csv file.
  You can check example of "output_img" folder here.
  https://drive.google.com/drive/folders/1nrns5zZkzfVP9o8aCyjkj9O4O-TwJAvP?usp=sharing
  Other columns can be recognized by the model and the results are stored in csv directly.
  The csv keeps table structure of original pdf
  I attached example "output_img" folder and "table_1.csv"

Please give me star if this project was helpful to your startup project. :)



  
