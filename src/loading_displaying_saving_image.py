import cv2 
import os


# Bild laden 
def loadImage(filename):
    img = cv2.imread(filename)
    return img

# Bild anzeigen 
def displayImage(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Bild speichern
def saveImage(img, filename):
    # Split pathname 
    name, extension = os.path.splitext(filename)
    # neuen Dateinamen generieren
    new_filename = f"{name}_copy{extension}"

    #Bild speichern 
    cv2.imwrite(new_filename, img)






