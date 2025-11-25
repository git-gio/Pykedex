# import the necessary packages
from skimage import exposure
from utils import *
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def find_screen(query_path, save_path, verbose=False):
    # load the query image, compute the ratio of the old height to the new height, clone it, and resize it
    img = cv2.imread(query_path)
    ratio = img.shape[0] / 300.0
    orig = img.copy()
    img = resize(img, height=300)

    # convert the image to grayscale, blur it, and find edges in the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    # gray = cv2.GaussianBlur(gray, (15, 15), 3.5)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 5)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # edged = auto_canny(thresh)

    # find contours in the edged image, keep only the largest ones, and initialize our screen contour
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    # loop over our contours
    for c in cnts:
        bbox = cv2.boundingRect(c)
        # approximate the contour
        peri = 0.1 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            print(cv2.contourArea(c))
            screenCnt = approx
            img_bbox = cv2.rectangle(img.copy(), bbox, (0, 0, 255), 4)
            img_cnt = cv2.drawContours(img.copy(), [screenCnt], -1, (0, 255, 0), 4)
            break
    # check if a contour has been found
    if screenCnt is None:
        cv2.imshow("image", resize(orig, height=300))
        cv2.imshow("img", resize(img, height=300))
        cv2.imshow("thresh", thresh)
        # cv2.imshow("edge", edged)
        # cv2.imshow("bbox", resize(img_bbox, height=300))
        # cv2.imshow("contour", resize(img_cnt, height=300))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return False

    pts = screenCnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    rect *= ratio

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

    warp_scaled = exposure.rescale_intensity(warp, (0, 255))
    # warp_scaled in realtà non cambia niente impostando out_range a (0, 255), threshold fa peggio
    # ret, warp_scaled = cv2.threshold(warp, 200, 255, cv2.THRESH_BINARY)
    # the Pokémon we want to identify will be in the top-right corner of the warped image -- let's crop this region out
    (h, w) = warp_scaled.shape[:2]
    (dX, dY) = (int(w * 0.4), int(h * 0.45))  # in origine per gb erano 0.4, 0.45
    crop = warp_scaled[5:dY - 23,
           w - dX - 10:w - 20]  # parametri originali = 10, no sottrazione a dY (-25 per togliere nome)
    # save the cropped image to file
    print(save_path)
    cv2.imwrite(save_path, crop)
    # cv2.imwrite(save_path + "warped_scaled.png", warp_scaled)
    # cv2.imwrite(save_path + "warped.png", warp)
    if verbose:
        # show our processed_images
        cv2.imshow("image", resize(orig, height=300))
        cv2.imshow("img", resize(img, height=300))
        cv2.imshow("thresh", thresh)
        #cv2.imshow("edge", edged)
        cv2.imshow("bbox", resize(img_bbox, height=300))
        cv2.imshow("contour", resize(img_cnt, height=300))
        cv2.imshow("warp", resize(warp, height=300))
        cv2.imshow("warp_scaled", resize(warp_scaled, height=300))
        cv2.imshow("crop", resize(crop, height=300))
        # cv2.imwrite("C:/Users/giova/Desktop/Uni/sistemi_intelligenti_avanzati/progetto/utils/immagini/segmentazione/image.png", orig)
        # cv2.imwrite("C:/Users/giova/Desktop/Uni/sistemi_intelligenti_avanzati/progetto/utils/immagini/segmentazione/img.png", img)
        # cv2.imwrite("C:/Users/giova/Desktop/Uni/sistemi_intelligenti_avanzati/progetto/utils/immagini/segmentazione/thresh.png", thresh)
        # # cv2.imshow("edge", edged)
        # cv2.imwrite("C:/Users/giova/Desktop/Uni/sistemi_intelligenti_avanzati/progetto/utils/immagini/segmentazione/bbox.png", img_bbox)
        # cv2.imwrite("C:/Users/giova/Desktop/Uni/sistemi_intelligenti_avanzati/progetto/utils/immagini/segmentazione/contour.png", img_cnt)
        # cv2.imwrite("C:/Users/giova/Desktop/Uni/sistemi_intelligenti_avanzati/progetto/utils/immagini/segmentazione/warp.png", warp)
        # cv2.imwrite("C:/Users/giova/Desktop/Uni/sistemi_intelligenti_avanzati/progetto/utils/immagini/segmentazione/warp_scaled.png", warp_scaled)
        # cv2.imwrite("C:/Users/giova/Desktop/Uni/sistemi_intelligenti_avanzati/progetto/utils/immagini/segmentazione/crop.png", crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return True


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)

    query_path = os.path.join(dirname, 'dataset/psyduck/noName/psyduck (2).jpg')
    save_path = os.path.join(dirname, 'processed_images/cropped.png')

    find_screen(query_path=query_path, save_path=save_path, verbose=True)
    quit()



