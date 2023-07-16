# import cv2
# import numpy as np
# from translator.utils import debug_image,inpaint,do_mask,cv2_to_pil,pil_to_cv2,mask_charactetrs


# frame = cv2.imread("contours_test_2.png")


# debug_image(frame,"Frame")
# mask = mask_charactetrs(frame)
# debug_image(mask,"Mask")
# debug_image(pil_to_cv2(inpaint(cv2_to_pil(frame),cv2_to_pil(mask))),"Inpainted")

import cv2
import numpy as np

def generate_character_mask(image_path):
    # Load the manga page image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to the grayscale image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)

    # Perform morphological operations to improve the text extraction
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours of the characters
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask image
    mask = np.zeros_like(image)

    # Draw contours on the mask
    for contour in contours:
        # Filter out small contours and contours with a large aspect ratio
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            

    # Find contours of the characters
    contours, heiriachy = cv2.findContours(cv2.threshold(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY),125,255,0)[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask image
    mask = np.zeros_like(image)

    # Draw contours on the mask
    for contour,info in zip(contours,heiriachy[0]):
        # Filter out small contours and contours with a large aspect ratio
        (x, y, w, h) = cv2.boundingRect(contour)
        if info[2] == -1:
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=1)#cv2.FILLED)

    return mask

# Example usage
image_path = 'contours_test_2.png'
character_mask = generate_character_mask(image_path)

# Display the character mask
cv2.imshow('Character Mask', character_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()