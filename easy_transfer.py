import albumentations as A
import os
import cv2
import numpy as np
import shutil

trans = A.Compose([
    A.RGBShift(90, 90, 90),
    A.HueSaturationValue(),
    A.ChannelShuffle(),
    A.CLAHE(),
    A.RandomGamma(),
    A.RandomScale(),
    A.ChannelDropout(),
])

base_color = dict(
    sensible=(240, 300),
    warm=(0, 90),
    sweet1=(315, 345),
    sweet2=(195, 240),
    energetic=(90, 180),
)


def get_black_style(img):
    bg = np.array(img[0][0], dtype=np.int32)
    m = cv2.inRange(img, bg - 20, bg + 20)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray[m > 100] = 255
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img


def get_random_style(img):
    img2 = trans(image=img)['image']
    return img2


trans2 = A.Compose([
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=50, val_shift_limit=50),
])


def get_specific_style(img, style, need_bg=False):
    bg = np.array(img[0][0], dtype=np.int32)
    m = cv2.inRange(img, bg - 5, bg + 5)
    img = trans2(image=img)['image']
    low, high = base_color[style]
    low, high = low // 2, high // 2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.asarray(hsv, dtype=np.int32)
    hsv[:, :, 0] = hsv[:, :, 0] + np.random.randint(0, 10000)
    hsv[:, :, 0] = hsv[:, :, 0] % (high - low) + low
    hsv = np.asarray(hsv, dtype=np.uint8)
    img2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if need_bg:
        img2[m > 100] = bg
    else:
        img2[m > 100] = (255, 255, 255)
    return img2


def convert(img, dir):
    img1 = cv2.imread(img)
    name = os.path.split(img)[-1].split('.')[0]
    this_dir = os.path.join(dir, name.split('.')[0])
    if os.path.exists(this_dir):
        shutil.rmtree(this_dir)
    os.mkdir(this_dir)
    cv2.imwrite(os.path.join(this_dir, name + '.png'), img1)
    # free style
    for i in range(50):
        img2 = trans(image=img1)['image']
        cv2.imwrite(os.path.join(this_dir, 'free' + str(i) + '.png'), img2)
    cv2.imwrite(os.path.join(this_dir ,'black' + '.png'), get_black_style(img1))
    # specific
    for style in base_color.keys():
        for i in range(10):
            img2 = get_specific_style(img1, style)
            cv2.imwrite(os.path.join(this_dir, style + str(i) + '.png'), img2)


def one_aug(img_path):
    img = cv2.imread(img_path)
    augs = dict(shift=A.RGBShift(90, 90, 90,p=1.0),
                hue=A.HueSaturationValue(p=1.0),
                cs=A.ChannelShuffle(p=1.0),
                rg=A.RandomGamma(p=1.0),
                rs=A.RandomScale(p=1.0),
                cd=A.ChannelDropout(p=1.0),
                )
    for aug in augs.keys():
        for i in range(5):
            img2 = augs[aug](image=img)['image']
            cv2.imwrite('base/'+aug+str(i)+'.png',img2)



if __name__ == '__main__':
    for root,dir,files in os.walk('data/test'):
        for f in files:
            if f.endswith('.png'):
                try:
                    convert(os.path.join(root,f),'data/tmp2')
                except Exception as e:
                    print(e)
    # for i in range(100):
    #     try:
    #         convert('data/logos_all/logo_' + str(i * 100).zfill(6) + '.png', 'data/tmp')
    #     except Exception as e:
    #         print(e)
    #         continue
