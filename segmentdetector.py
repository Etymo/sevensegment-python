
# coding: utf-8

# In[27]:


import numpy as np
import cv2

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

cv2.__version__


# In[28]:


def ImageBinarization(im1, threshold = 150):
    
    for i in range(len(im1)):
        for j in range(len(im1[i])):
            if im1[i][j] > threshold: 
                im1[i][j] = 255
            else:
                im1[i][j] = 0
    
    return im1


def  RemoveNoise(im1, threshold_px = 50):
    ## 水平一列を走査して閾値より小さければ除去
    #WARN: ドットも消える可能性あり
    
    
    #TODO: ネストが深くて見にくい、要修正
    for i in range(len(im1)):
        count_horizon = 0
        for j in range(len(im1[i])):
            if sum(im1[i][j]) > 10:
                count_horizon +=1
                
        if count_horizon < threshold_px:
            for j in range(len(im1[i])):
                im1[i][j] = 0
                
    return im1


def CropImage(im1, target_width, target_height,  wstart, hstart):    
    hnew_min = hstart
    hnew_max = hstart + target_height
    wnew_min = wstart
    wnew_max = wstart + target_width

    
    im2 = np.zeros((target_height, target_width, 3), np.uint8)
#    plt.imsave("webcam_out/out01.jpg", blank)
#    im2 = cv2.imread("webcam_out/out01.jpg", 0)

    for h in range(target_height):
        for w in range(target_width):
                im2[h][w] = im1[hnew_min+h][wnew_min+w]
   
    return im2

def padding_position(x, y, w, h, p):
    return x - p, y - p, w + p * 2, h + p * 2


# In[29]:


def DetectVerticle(img, numarea, segments, starth, startw, len_detector, thresh_px):
    ##垂直方向の検出
    
    num_px = 0
    for h in range(starth, starth + len_detector):
        w = startw
        if sum(img[h][w])  > 30:
            num_px += 1
            if num_px > thresh_px:
                segments[numarea] = 1
                return segments

    return segments


def DetectHorizon(img, numarea, segments, starth, startw, len_detector, thresh_px):
    ##水平方向の検出
    num_px = 0
    for w in range(startw, startw + len_detector):
        h = starth
        if sum(img[h][w])  > 30:
            num_px += 1
            if num_px > thresh_px:
                segments[numarea] = 1
                return segments

    return segments



def RecognizeNumber(digit_img, len_detector = 9):
    
    thresh_px = 1
    segments = [0, 0, 0, 0, 0, 0, 0]

    #多めに減らしておく(detectorの初期位置になるため)
    w_AGD = len(digit_img[0]) // 2 
    w_BC    = len(digit_img[0]) - (len_detector + 2)
    w_FE     = len(digit_img[0]) // 4 - 5

    h_BF   = len(digit_img) // 3
    h_CE   = len(digit_img) // 3 * 2
    h_A      = 2
    h_G      = len(digit_img) //2 - 5
    h_D      = len(digit_img) - 10
    
    #TODO: もっとスマートに書けるか？ 見栄えが悪い
    segments = DetectVerticle(digit_img, 0, segments,  h_A,   w_AGD, len_detector, thresh_px)  #A
    segments = DetectHorizon(digit_img, 1, segments, h_BF,     w_BC, len_detector, thresh_px)  #B
    segments = DetectHorizon(digit_img, 2, segments,  h_CE,     w_BC, len_detector, thresh_px)  #C
    segments = DetectVerticle(digit_img, 3, segments,  h_D,     w_AGD, len_detector, thresh_px)  #D
    segments = DetectHorizon(digit_img, 4, segments, h_CE,     w_FE, len_detector, thresh_px)  #E
    segments = DetectHorizon(digit_img, 5, segments, h_BF,     w_FE, len_detector, thresh_px)  #F
    segments = DetectVerticle(digit_img, 6, segments, h_G,     w_AGD, len_detector, thresh_px)  #G
        
    #WARN: 7の条件に注意(fを含むか?)
    if segments == [1, 1, 1, 1, 1, 1, 0]: return 0
    if segments == [0, 1, 1, 0, 0, 0, 0]: return 1
    if segments == [1, 1, 0, 1, 1, 0, 1]: return 2
    if segments == [1, 1, 1, 1, 0, 0, 1]: return 3
    if segments == [0, 1, 1, 0, 0, 1, 1]: return 4
    if segments == [1, 0, 1, 1, 0, 1, 1]: return 5
    if segments == [1, 0, 1, 1, 1, 1, 1]: return 6
    if segments == [1, 1, 1, 0, 0, 1, 0]: return 7
    if segments == [1, 1, 1, 1, 1, 1, 1]: return 8
    if segments == [1, 1, 1, 1, 0, 1, 1]: return 9
    
    return "Error"


def ReturnNumbers(digits_img):
    nums = []
    for i in range(len(digits_img)):
        num = RecognizeNumber(digits_img[i])
        nums.append(num)
    
    return nums


def PlotMultiImages(images, row = 1):
    #TODO: サイズの調整(y軸に合わせる等)
    col = len(images)
    images = images[::-1]

    plt.figure(figsize = (20,20))

    for i in range(1, len(images)+1):
        plt.subplot(row, col, i)

        plt.imshow(images[len(images)-i], cmap = "gray")
        plt.axis("off")

    plt.tight_layout()


# In[30]:


def RecognizeDP(image, h = 37,  thresh_px = 2):
    detector_px = 5
    white_px = 0
    isDP = False

    for w in range(len(image[h]) - 5, len(image[h])):
        if sum(image[h][w]) > 100:
            white_px += 1
            if white_px > thresh_px:
                isDP = True
                break
                
    return isDP

def Numbers2Float(numbers):
    
    ##FIXIT: 小数点がない場合機能しない
    
    value_float = 0
    pos_DP = numbers.index(".")

    value_int = 0
    for i in range(pos_DP):
        value_int += numbers[i] * pow(10, pos_DP - i - 1)


    value_dicimal = 0
    for i in range(pos_DP + 1, len(numbers)):
        value_dicimal += numbers[i] * pow(10, pos_DP - i)


    value_float = value_int + value_dicimal
    return value_float

#TEST: Number
#numbers = [2, 5, ".", 5, 3]
#value = Numbers2Float(numbers)
#print(value)


# In[31]:


def InsertDP(images, numbers):
    isDP = False


    for i in range(len(images)):
        isDP = RecognizeDP(images[i])
        if(isDP == True):
            numbers.insert(i+1,".")
            break
    
    return numbers


def RecognizeExponentSign(image, threshold_px = 40):

#    plt.imshow(image)


    threshold_px = 40

    px = 0
    isMinus = True

    for h in range(len(image)):
        for w in range(len(image[h])):
            if(sum(image[h][w]) > 100):
                px+= 1


    if px > threshold_px:
        isMinus = False
        return isMinus

    return isMinus


# In[32]:


#TODO: 各変換前後をプロットする(matplotlibのsubplot?)
#TODO: 部分テストの実装

path_read = "webcam_img/webcam_01.png"
path_out = "webcam_out/out_01.png"

#TODO: 見つからなかった場合の例外処理
im1 = cv2.imread(path_read, 0)

plt.imshow(im1, cmap = "gray")


# In[33]:


im1_bin = ImageBinarization(im1,200)

#左上を指定
area_width = 130
area_startw = 70

area_height = 80
area_starth = 185

im1_area = CropImage(im1_bin, area_width, area_height, area_startw, area_starth)
im1_area = RemoveNoise(im1_area, 1)
im1_area = cv2.cvtColor(im1_area, cv2.COLOR_BGR2RGB)
plt.imsave(path_out, im1_area)

plt.imshow(im1_area)


# In[34]:


#基数部分
number_width =30
number_height = 45
number_startw = 8
number_starth = 30

images = []
numbers = []

im2 = cv2.imread(path_out, 0)

for i in range(3):
    w = number_startw + i * number_width
    h = number_starth
    image = CropImage(im2, number_width, number_height, w, h)
    images.append(image)

PlotMultiImages(images)

    
numbers = ReturnNumbers(images)
numbers_dp = InsertDP(images, numbers)

value = Numbers2Float(numbers_dp)
print(numbers_dp)
print(value)


# In[35]:


#Exponent:指数部分
exponent_width =18
exponent_height = 25
exponent_startw = 90
exponent_starth = 4

images_exponent = []

for i in range(2):
    w = exponent_startw + i * exponent_width
    h = exponent_starth
    image = CropImage(im2, exponent_width, exponent_height, w, h)
    images_exponent.append(image)


PlotMultiImages(images_exponent)
exponent_val = RecognizeNumber(images_exponent[1])

isExponentMinus = RecognizeExponentSign(images_exponent[0])
if isExponentMinus == True: exponent_val *= -1

print("1e{} ".format(exponent_val))


# In[ ]:





# In[ ]:


#|====A====|
#| F                   | B
#|                      |
#|====G====|
#|                      | 
#| E                   | C
#|====D====|    


# In[ ]:


import time
start_time = time.time()

elapsed_time = time.time() - start_time
print("処理時間: {:.4}[s]".format(elapsed_time))

