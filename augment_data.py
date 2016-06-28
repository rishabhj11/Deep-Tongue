import csv
import cv2


def shrink(image, image_name):
    to_shrink = image.copy()
    shrunk = cv2.resize(to_shrink, (200, 200)).copy()
    cv2.imshow('original', image)
    cv2.imshow('shrunk', shrunk)
    cv2.waitKey()
    cv2.imwrite(image_name + '(5)' + '.jpg', shrunk)


def enlarge(image, image_name):
    to_enlarge = image.copy()
    enlarged = cv2.resize(to_enlarge, (512, 512))
    '''cv2.imshow('original', image)
    cv2.imshow('enlarged', enlarged)
    cv2.waitKey()'''
    cv2.imwrite(image_name + '(6)' + '.jpg', enlarged)


def rotate_ninety(image, image_name):
    to_rotate = image.copy()
    (h, w) = to_rotate.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, 90, 1.0)  # Calculates an affine matrix of 2D rotation
    rotated = cv2.warpAffine(to_rotate, matrix, (w, h))  # Applies an affine transformation to an image.
    '''cv2.imshow('original', image)
       cv2.imshow('enlarged', rotated)
       cv2.waitKey()'''
    cv2.imwrite(image_name + '(1)' + '.jpg', rotated)


def rotate_180(image, image_name):
    to_rotate = image.copy()
    (h, w) = to_rotate.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, 180, 1.0)  # Calculates an affine matrix of 2D rotation
    rotated = cv2.warpAffine(to_rotate, matrix, (w, h))  # Applies an affine transformation to an image.
    '''cv2.imshow('original', image)
    cv2.imshow('enlarged', rotated)
    cv2.waitKey()'''
    cv2.imwrite(image_name + '(2)' + '.jpg', rotated)


def horizontal_flip(image, image_name):
    to_flip = image.copy()
    cv2.flip(to_flip, 0, to_flip)
    cv2.imwrite(image_name + '(3)' + '.jpg', to_flip)


def vertical_flip(image, image_name):
        to_flip = image.copy()
        cv2.flip(to_flip, 1, to_flip)
        cv2.imwrite(image_name + '(4)' + '.jpg', to_flip)


file_name = '1-20.csv'
data_set = csv.reader(open(file_name, "rb"))
positive_list = []
for row in data_set:
    if float(row[1]) == 1:
        img = cv2.imread(row[0]+'.jpg')
        positive_list.append(row[0])
        print (row[0])
        rotate_ninety(img, row[0])
        rotate_180(img, row[0])
        horizontal_flip(img, row[0])
        vertical_flip(img, row[0])
        # shrink(img, row[0])
        # enlarge(img, row[0])

with open(file_name, 'ab') as fp:
    for item in positive_list:
        for i in range(1, 5):
            a = csv.writer(fp, delimiter=',')
            data = [[item + '(' + str(i) + ')', 1]]
            a.writerows(data)



