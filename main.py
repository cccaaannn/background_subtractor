import os
import cv2
import numpy as np
import random as rnd



def path_creator(path, img_count,staring_string,extension):
    img_count_str = str(img_count)

    while True:
        if(len(img_count_str)<6):
            img_count_str = "0" + img_count_str
        else:
            break
    img_path = "{0}{1}{2}{3}".format(path,staring_string,img_count_str, extension)
    img_name = "{0}{1}{2}".format(staring_string,img_count_str, extension)
    return img_path, img_name

def get_images(*,img_path, staring_image, img_count, random=False):
    images = []
    for i in range(img_count):
        if(random):
            image_count = rnd.randint(1,1000)
        else:
            image_count = i + staring_image

        path, _ = path_creator(img_path, image_count, staring_string = "in",extension = ".jpg")
        img = cv2.imread(path)
        images.append(img)
    # show_images(images)
    return images

def save_image(image_path, image):
    cv2.imwrite(image_path, image)

def median_method(frames):

    medianFrame = np.zeros(frames[0].shape)
    print(frames[0].shape)
    # satır sutun  240 320  3

    R = []
    G = []
    B = []
    for row_count in range(frames[0].shape[0]):
        for colum_count in range(frames[0].shape[1]):
            for frame in frames:
                R.append(frame[row_count][colum_count][0])
                G.append(frame[row_count][colum_count][1])
                B.append(frame[row_count][colum_count][2])

            medianFrame[row_count][colum_count][0] = np.median(R)
            medianFrame[row_count][colum_count][1] = np.median(G)
            medianFrame[row_count][colum_count][2] = np.median(B)
            R = []
            G = []
            B = []
    medianFrame = medianFrame.astype(dtype=np.uint8)  
    return medianFrame

def evaluate(img1, img2):
    mean_error = abs((img1.astype("float") - img2.astype("float"))).mean()
    sum_error = abs(np.sum(img1.astype("float") - img2.astype("float")))/255
    return sum_error, mean_error

def create_log_file(logs, file_name):
    with open(file_name + ".log", "w") as file:
        for log in logs:
            file.write(log)
            file.write("\n")




# my implementation (slower than opencv)
def abs_diff(img1,img2):
    diffrance = np.zeros(img1.shape)
    for row_count in range(img1.shape[0]):
        for colum_count in range(img1.shape[1]):
            for color in range(3):
                diffrance[row_count][colum_count][color] = abs(int(img1[row_count][colum_count][color]) - int(img2[row_count][colum_count][color]))
    diffrance = diffrance.astype(dtype=np.uint8) 
    return diffrance

def abs_diff_gray(img1,img2):
    diffrance = np.zeros(img1.shape)
    for row_count in range(img1.shape[0]):
        for colum_count in range(img1.shape[1]):
            diffrance[row_count][colum_count] = abs(int(img1[row_count][colum_count]) - int(img2[row_count][colum_count]))
    diffrance = diffrance.astype(dtype=np.uint8) 
    return diffrance


# dataset
# http://www.changedetection.net  Dataset>2012>Baseline

test_img_root_path = "C:\\Users\\can\\ProjectDependencies\\datasets\\computer_vision\\highway\\input\\"
validation_img_root_path = "C:\\Users\\can\\ProjectDependencies\\datasets\\computer_vision\\highway\\groundtruth\\"
save_path = "C:\\Users\\can\\Desktop\\highway"

logfile_name = "highway"


# get images
images = get_images(img_path = test_img_root_path, staring_image=1, img_count=25, random=True)


# medianFrame = median_method(images)
medianFrame = np.median(images, axis=0).astype(dtype=np.uint8)    
medianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

results = []
img_count = 0
while(True):

    # get paths
    img_count += 1
    test_img_path, test_img_name = path_creator(test_img_root_path, img_count, staring_string = "in",extension = ".jpg")
    validation_img_path, _ = path_creator(validation_img_root_path, img_count, staring_string = "gt",extension = ".png")

    # read images
    test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    val_img = cv2.imread(validation_img_path, cv2.IMREAD_GRAYSCALE)

    if(test_img is None):
        break


    # find difference and treshold
    # difference = abs_diff_gray(medianFrame, frame)
    difference = cv2.absdiff(medianFrame, test_img)
    _, thresholded_img = cv2.threshold(difference, 35, 255, cv2.THRESH_BINARY)

    # save_image(os.path.join(save_path, test_img_name), thresholded_img)

    # evaluate
    sum_error, mean_error =  evaluate(thresholded_img,val_img)

    # show results
    result = "img name:{0}  ->  sum_error:{1:.0f}  mean_error:{2:.3f} ".format(test_img_name, sum_error, mean_error)
    print(result)
    results.append(result)


    # show images
    cv2.imshow("median img", medianFrame)
    cv2.imshow("test img", test_img)
    cv2.imshow("validation img", val_img)
    cv2.imshow("difference", thresholded_img)

    cv2.waitKey(30)
    

# save results
create_log_file(results, logfile_name)












