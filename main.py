import cv2
import pyautogui

video = cv2.VideoCapture(0)
subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    _, frame = video.read()
    flipFrame = cv2.flip(frame, 1)

    grayscale_img = cv2.cvtColor(flipFrame, cv2.COLOR_BGR2GRAY)

    threshold = cv2.threshold(grayscale_img, 245, 255, cv2.THRESH_BINARY)[1]
    mask = subtractor.apply(threshold)

    erosion_size = 2
    erosion_shape = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))
    erosion = cv2.erode(mask, element)

    res = cv2.bitwise_and(erosion, threshold)

    contours, _ = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        pyautogui.moveTo(x, y)

    cv2.imshow('gray', grayscale_img)
    cv2.imshow('mask', mask)
    cv2.imshow("webcam", res)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
