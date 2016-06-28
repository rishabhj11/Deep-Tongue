import csv

file_name = '1-20.csv'
partition_ratio = 0.80
data_set = csv.reader(open(file_name, "rb"))
image_set, label_set = [], []
i, j = 0, 0
for row in data_set:
    '''
    print row[0]
    img = cv2.imread(row[0]+'.jpg')
    print (img.shape)
    '''
    if float(row[1]) == 1:
        i += 1
    if float(row[1]) == 0:
        j += 1

print (i, j)