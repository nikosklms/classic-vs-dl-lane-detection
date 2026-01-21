Project: Lane Detection & Drivable Area Segmentation using traditional CV vs YOLOP
Authors: Vasileios Grapsopoulos, Nikolaos Kalamaris
Date: January 2026

--- OVERVIEW ---
This submission contains the source code used to reproduce the experiments described in our paper. 
**YOLOP**: The code is provided as a Jupyter Notebook (.ipynb) designed to run on Google Colab, 
which handles all dependencies and hardware acceleration (GPU).
**Our implementations**: We provide a python script with a **hard coded** input folder path with images, it does the processing
as described and produces an output folder with the final images 

--- HOW TO RUN ---
**YOLOP**
Option 1: Using the provided .ipynb file (Recommended)
1. Go to https://colab.research.google.com/
2. Click "Upload" and select the file "yolop.ipynb" from this folder.
3. Important: Before running, go to "Runtime" -> "Change runtime type" and select "T4 GPU".
4. Follow the instructions inside the notebook to mount Google Drive and download the necessary data/weights.

Option 2: Direct Colab Link
You can access the notebook directly via the following link (ensure you are logged in to Google):
https://colab.research.google.com/drive/1mSE0hwnCSZ7L8kiukyOUtdBlGBHWuUJW?usp=sharing
*Note: Please make sure to copy the notebook to your drive if you wish to make changes.*

**Our implementations**
libraries used: cv2 , numpy , os
simply run python3 ld_das.py
to change the input images changes to following line:
img_dir = "images_hybrid/" -> img_dir = "my_path/"
were my_path is the image folder to be tested

--- REQUIREMENTS ---
- Google Account (for Colab)
- Google Drive (for dataset storage, as explained in the notebook cells)
