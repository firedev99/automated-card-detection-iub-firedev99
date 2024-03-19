import cv2 
import pytesseract
import re


# define Tesseract-OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class InitializeDetector():
    def __init__(self):
        self.input_image = None
        self.rois_offsets = {
            "name": 0.585,
            "student_id": 0.67
        }
        self.labels = {}


    # pre process the image before finding the contours 
    def preprocess_image(self, img, target_size=(1000, 1000)):
        self.input_image = img.copy()
        height, width = img.shape[:2]
        
        # calculate aspect ratio
        aspect_ratio = width / height

        # resize to target size while preserving aspect ratio
        if aspect_ratio > 1:
            new_width = target_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_size[1]
            new_width = int(new_height * aspect_ratio)

        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)


        self.input_image = img
        
        

    # detect id card bounding box based on the largest contours  
    def detect_id(self):
        img = self.input_image
 
        # covert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # apply gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # apply thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
        # find the contours in the 
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # filter out small contours
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
        filtered_contours.sort(key=cv2.contourArea, reverse=True)
        
        if len(filtered_contours) == 0:
            ValueError('something went wrong, try again!')

        id_card_contours = filtered_contours[0]


        # get the bounding box of the id card 
        x, y, w, h = cv2.boundingRect(id_card_contours)


        # Crop the document from the original image
        cropped_image = img[y:y+h, x:x+w]  

        
        # x, y, w, h = cv2.boundingRect(filtered_contours[0])
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        

        self.input_image = cropped_image
        


    # capitalize text 
    def capitalize(self, text):
        words = text.split()
        camel_case = ' '.join(word.capitalize() for word in words[:])
        return camel_case
        
    
    # classify text extraction
    def classify_text(self):
        img = self.input_image
        details = []
        height, width = img.shape[:2]
        space = int(0.11 * height)

        # extract the text from specific coordinates based on region of interest 
        for cat, offset in self.rois_offsets.items():
            # get the region of interest (y-axis)
            roi_y = int(offset * height)
            # crop the detected labeled text 
            roi_cropped = img[roi_y:roi_y + space, 0:width]

            # extract the labeled text 
            text = pytesseract.image_to_string(roi_cropped)
            
            if cat == "name":
                text = self.capitalize(text.replace('Student', ''))
                details.append(text)
            else:
                # extract all digits
                text = max(re.findall(r'\d+', text))
                details.append(text)
                

        if len(details) != 2:
            ValueError('something went wrong')
            return None
        

        self.labels = {
            "name": details[0],
            "student_id": details[1]
        }
        
        print(self.labels)
    
    



if __name__ == '__main__':
    img = cv2.imread('samples/random_lost_iub_id_card_3.jpg')

    # initialize detector
    detector = InitializeDetector()

    # pre process the image
    preprocessed_img = detector.preprocess_image(img)

    # detect id card bounding box edges based on contours 
    contours = detector.detect_id()

    # extract data based on specified label
    detector.classify_text()


    cv2.waitKey(0)


