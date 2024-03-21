import cv2
import easyocr
import pandas as pd


class Detector():
    def __init__(self):
        self.img = None
        self.roi_offsets = {
            "name": 0.572,
            "student_id": 0.67
        }
        
    
    # detect the corner
    def detect_corner(self, img):
        self.img = img

        # covert the image to grayscale 
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

        # apply gaussian blur 
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)


        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # filter out the small contours
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
        filtered_contours.sort(key=cv2.contourArea, reverse=True)


        # contours sorounding the id card
        id_card_contours = filtered_contours[0]


        return id_card_contours



    # get the id card 
    def get_id_card(self, contours):
        img = self.img 

        # detected bounding rectangle
        x, y, w, h = cv2.boundingRect(contours)

        # cropped detected id card
        cropped_img = img.copy()[y:y+h, x:x+w] 
        new_img = cv2.resize(cropped_img, (600, 600), interpolation=cv2.INTER_CUBIC)

        return new_img
    


    # recognize text and their label based on their coordinates 
    def recognize_text(self, img):
        details = []

        # generate an empty data frame
        df = pd.DataFrame()

        # initialize easyocr
        reader = easyocr.Reader(['en'], gpu=False)

        # extract the height and width from the image 
        height, width = img.shape[:2]
        # make some gap around the region of interest 
        space = int(0.12 * height)

        # assign label and extract text based on region of interest     
        for _, (label, offset) in enumerate(self.roi_offsets.items()): 
            roi_y = int(offset * height)
            
            # crop the area around the detected label
            roi_cropped = img[roi_y:roi_y + space, 0:width]
    
            
            # extract the text using easy-ocr   
            results = reader.readtext(roi_cropped)
            # generate a data frame based on the result       
            img_df = pd.DataFrame(results, columns=['bbox','text','conf'])
  

            # modify data frame values         
            if label == 'name':
                # get the text id that has max len            
                long_text_id = img_df['text'].str.len().idxmax()   
                text = ''.join(img_df['text'][long_text_id]).replace(':', '.')
            else:
                # get the text id that has max len 
                long_text_id = img_df['text'].str.len().idxmax()
                text = ''.join(img_df['text'][long_text_id]).replace('ID: ', '')
                
                            
            # append student details and data frame      
            details.append(text)
            df = df._append(img_df)

    
        return df, details
    





if __name__ == '__main__':
    test = cv2.imread(r"E:\ML\projects\automated-card-detection-iub-firedev99\samples\random_lost_iub_id_card_4.jpg")

    # initialize the detector
    d = Detector()

    # detect corner of the image 
    detected_id_contours = d.detect_corner(test)

    # get the detected id card 
    result = d.get_id_card(detected_id_contours)

    # recognize the text based on region of interest 
    df, details = d.recognize_text(result)

    print(details)

