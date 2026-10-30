# Import basic libraries
import numpy as np                # NumPy for numerical operations and arrays
import matplotlib.pyplot as plt   # For visualizing images and data
from PIL import Image             # For opening image files
import cv2                        # OpenCV library (used at the end: BGR vs RGB)

from common import data_path

# --- NumPy Basics: Arrays and Matrices ---

myList = [1,2,4,5,8]
arr = np.array(myList)            # Convert Python list into a NumPy array

arr = np.arange(0,10)             # Create array of numbers 0–9
print(arr)                        # Output: [0 1 2 3 4 5 6 7 8 9]

arr = np.arange(0,10,2)           # Create array with a step size of 2 → [0 2 4 6 8]

arr = np.zeros((5,5))             # Create a 5x5 matrix filled with zeros
arr = np.ones((2,4))              # Create a 2x4 matrix filled with ones

np.random.seed(101)               # Set random seed for reproducibility
arr = np.random.randint(0,100,10) # Create 10 random integers between 0 and 99

arr2 = np.random.randint(0,100,10)# Another random array (same range, different values)

# --- Basic Array Operations ---

arr.max()     # Returns maximum value in array
arr.min()     # Returns minimum value
arr.mean()    # Returns average value
arr.argmin()  # Returns index of smallest value
arr.argmax()  # Returns index of largest value

arr.reshape(2,5) # Reshape 1D array (10 values) into 2 rows x 5 columns

# --- Matrix Example (2D Array) ---

mat = np.arange(0,100).reshape(10,10)  # Create 10x10 matrix with values 0–99
print(mat)                             # Print the matrix (like an image grid)

row = 0
col = 1
mat[row,col]          # Access single value at row 0, column 1 → value = 1

# --- Slicing (selecting parts of the matrix) ---

mat[:,col]            # All rows, column 1 → returns the 2nd column
mat[row,:]            # Row 0, all columns → returns the 1st row
mat[0:3,0:3]          # Top-left 3x3 corner of the matrix
mat.max()             # Maximum value in matrix → 99

print(mat)            # Prints matrix again (that’s the double output you saw)

# --- Image as an Array (Bridge to Computer Vision) ---

pic = Image.open(data_path('red_panda.jpg'))  # Open an image file
pic                                     # Show image metadata (not the picture itself)

type(pic)   # Check the object type → <class 'PIL.JpegImagePlugin.JpegImageFile'>

pic_arr = np.asarray(pic)   # Convert image into a NumPy array (pixels → numbers)
pic_arr.shape               # Output: (height, width, 3) → 3 = RGB channels
print(pic_arr.shape)

# Create a copy of the image to modify
pic_red = pic_arr.copy()


# PIL loads images as RGB → channel 0 = Red, 1 = Green, 2 = Blue
# Keep only the red channel:
pic_red[:, :, 1] = 0   # Set green channel to 0
pic_red[:, :, 2] = 0   # Set blue channel to 0

# Show the modified image
plt.imshow(pic_red)
plt.title("Only the red channel (RGB index 0)")
plt.show()

# --- The classic trap: OpenCV loads images as BGR, not RGB ---

img_bgr = cv2.imread(str(data_path('red_panda.jpg')))   # channel 0 = Blue!
plt.imshow(img_bgr)                                      # Matplotlib assumes RGB
plt.title("OpenCV image shown as-is: red and blue swapped")
plt.show()

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)      # fix: convert before showing
plt.imshow(img_rgb)
plt.title("After cv2.cvtColor(..., cv2.COLOR_BGR2RGB)")
plt.show()
