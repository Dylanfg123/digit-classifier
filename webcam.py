import cv2
from joblib import load


def show_webcam(mirror=False):
    clf = load("clf.joblib")
    clf3 = load("clf3.joblib")
    i = 0
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    newImage = cam.read()

    while True:
        ret_val, img = cam.read()
        # img, contonours, thresh = get_img_contour_thresh(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #gray = cv2.bilateralFilter(gray, 11, 17, 17)

        img1 = cv2.GaussianBlur(gray, (35, 35), 0)
        img12 = cv2.GaussianBlur(gray, (11, 11), 0)

        img13 = cv2.GaussianBlur(gray, (51, 51), 0)
        img2 = cv2.blur(gray, (35, 35))
        ret, thresh1 = cv2.threshold(img12, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh2 = thresh1[60:60 + 400, 0:0 + 400]
        #thresh2 = thresh1[0:0 + 300, 0:0 + 300]
        contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:
                print("Test " + str(i))
                i += 1
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img, (x-20, y+20), (x+w+10, y+h+70), (200,255,200), 2)
                #cv2.rectangle(img, (x, y), (x + w, y + h), (200, 255, 200), 2)
                newImage = thresh1[y:y + h + 60, x:x + w + 60]
                #newImage = thresh1[y:y + h, x:x + w]
                newImage = cv2.resize(newImage, (20, 20))
                newImage = newImage / 255.0
                color = [0, 0, 0]
                newImage = cv2.copyMakeBorder(newImage, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=color)
                #print(newImage)
                prediction = clf.predict(newImage.reshape(1, -1))
                print(prediction)
                cv2.putText(img, "Prediction CLF: " + str(prediction), (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

                prediction2 = clf3.predict(newImage.reshape(1, -1))
                cv2.putText(img, "Prediction CLF3: " + str(prediction2), (40, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 100, 255), 2)

        # cv2.drawContours(thresh2, contours, -1, (255, 255, 0), 8)
        # if mirror: 
        #     img1 = cv2.flip(img, 1)
        cv2.imshow('original', img)
        # cv2.imshow('gray', gray)
        # cv2.imshow('gaussian mid', img1)
        # cv2.imshow('gaus low', img12)
        # cv2.imshow('gaus high', img13)
        # cv2.imshow('blur', img2)
        cv2.imshow('threshold', thresh1)
        cv2.imshow('threhold2', thresh2)
        # print(type(newImage))
        # print("New Image Test")
        cv2.imshow('test images', newImage)
        # cv2.imshow('test', test)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()

# def get_img_contour_thresh(img):
#     x, y, w, h = 0, 0, 300, 300
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (35, 35), 0)
#     ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     thresh1 = thresh1[y:y + h, x:x + w]
#     contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
#     return img, contours, thresh1


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()