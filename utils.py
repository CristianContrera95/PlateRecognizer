import re
import os
import ftplib

from random import randint
from io import BytesIO

from skimage.metrics import structural_similarity
from PIL import Image

import cv2
import requests
import imagehash
import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    from imageai.Detection import ObjectDetection


class FTP:

    def __init__(self, url, port, folder='.', user='anonymous', password='', connect_tries=10):
    
        self.ftp = ftplib.FTP()
        self.url = url
        self.port = port
        self.folder = folder
        self.user = user
        self.password = password
        self.connect_tries = connect_tries
        self.img_buff = BytesIO()

    def __check_addr(self):
        if self.url[len(self.url)-1] == '.':
                self.url = self.url[:-1]

        if re.fullmatch('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', self.url) is None:
            print('Error in server address')
            return 1
        if not isinstance(self.port, int):
            try:
                self.port = int(self.port)
            except:
                print('Error in server address')
                return 1
        return 0            

    def connect(self, login=False):
        if self.__check_addr():
            return 1
        count = 0

        while count < self.connect_tries:
            try:
                count += 1
                self.ftp.connect(self.url, self.port)
                print(self.ftp.welcome, '\nConnected to ftp server in {}:{}'.format(self.url, self.port))
            except Exception as ex:
                if count == self.connect_tries:
                    print('Couldn\'t connect to {}:{} ftp-server.\nPlease check address'.format(self.url, self.port))
                    return 1
            else:
                break
        if login:
            self.login()
        return 0

    def login(self):
        try:
            self.ftp.login(self.user, self.password)
            print('Logged in server with {} user'.format(self.user))
        except Exception as ex:
            print('Couldn\'t login with {} user.\nPlease check user and password'.format(self.user))

    def change_folder(self):
        try:
            self.ftp.cwd(self.folder)
        except Exception as ex:
            print('Couldn\'t find {} dir.\nYou do not have permissions or the directory does not exist'.format(self.folder))
            return 1
        return 0

    def close(self):
        pass

    def handle_binary(self, data):
        self.img_buff.write(data)

    def get_file(self, filename):
        try:
            #filename = os.path.join(self.folder, filename)
            print('getting: {}'.format(filename))
            result = self.ftp.retrbinary('RETR {}'.format(os.path.join(self.folder, filename)),
                                         self.handle_binary,
                                         blocksize=1024*1024*1024)
            self.img_buff.seek(0)
            if not result.startswith('226'):
                return False
            return True
        except Exception as ex:
            print(ex)
            return False

    def get_files_from_folder(self, foldername):
        try:
            return self.ftp.nlst(foldername)
        except Exception as ex:
            return []

    def remove_file(self, filename, folder=False):
        pass

    def create_file(self, filename, folder=False):
        pass


def get_probably_char(candidates, index, func):
    for candidate in candidates:
        n_plate = candidate['plate']
        if getattr(n_plate[index], func)(): 
            return n_plate[index]


class API:
    """docstring for API"""
    def __init__(self, api_url, token, regions=None):
        self.API_URL = api_url
        self.API_TOKEN = token
        self.regions = regions

    def request(self, img):
        with open(img, 'rb') as fp:
            response = requests.post(
                    self.API_URL,
                    files=dict(upload=fp),
                    headers={'Authorization': 'Token {}'.format(self.API_TOKEN)}
                    )
        return response.json()

    def improve_plate(self, result):
        result = result[0] if isinstance(result, list) and (len(result) > 0) else result

        plate_box = list(result['box'].values())  # ymin, xmin, ymax, xmax
        plate_box = [plate_box[1]-10, plate_box[0]-10, plate_box[3]+10, plate_box[2]+10]
        plate = result['plate']
        is_new_plate = 1 if len(plate) == 7 else 0

        for i in range(len(plate)):
            if (is_new_plate and (i in [0,1,5,6])) or ((not is_new_plate) and (i in [3,4,5])):
                if not plate[i].isnumeric():
                    c = get_probably_char(result['candidates'], i, 'isnumeric')
                    l = list(plate)
                    l[i] = c
                    plate = ''.join(l)
            else:
                if not plate[i].isalpha():
                    c = get_probably_char(result['candidates'], i, 'isalpha')
                    l = list(plate)
                    l[i] = c
                    plate = ''.join(l)

        return plate.upper(), plate_box


def get_center_square(squares_list):
    # order squares_list by best centered squares
    results = []
    for square in squares_list:
        height = square[5][0]
        width = square[5][0]
        x1,y1,x2,y2 = square[3]
        percen_centered_width = abs((((x1+x2)/2) * 100 / width) - 50)
        percen_centered_height = abs((((y1+y2)/2) * 100 / height) - 50)
        results.append((percen_centered_width, percen_centered_height, square))

    results = list(sorted(results, key=lambda x: (x[0], x[1]), reverse=True))
    return list(map(lambda x: x[2], results))


class CarDetector:

    def __init__(self, model, threshold):
        super(CarDetector, self).__init__()
        self.model = model
        self.threshold = threshold
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsTinyYOLOv3()
        self.detector.setModelPath(os.path.join(os.curdir, self.model))
        self.detector.loadModel(detection_speed='normal')  # "normal, "fast", "faster", "fastest","flash".
        self.custom = self.detector.CustomObjects(car=True, motorcycle=True, bus=True, truck=True, suitcase=True)

    def remove_equalscars(self, images_list):

        images_list = sorted(images_list, key=lambda x: len(x), reverse=True)
        results = images_list.copy()

        first_image = images_list.pop(0)
        col_num = 0
        row_num = 0
        first_num = 0
        for car1, img_size1, _ in first_image:
            flag = True
            gray1 = cv2.cvtColor(car1, cv2.COLOR_BGR2GRAY)
            for image in images_list:
                row_num += 1

                scores = []
                for car2, img_size2, _ in image:

                    gray2 = cv2.cvtColor(car2, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]), interpolation=cv2.INTER_AREA)

                    scores.append((col_num, img_size2, structural_similarity(gray1, gray2)))
                    col_num += 1

                if scores:
                    col, img_size2, max_score = max(scores, key=lambda x: x[2])

                    if max_score > 0.20:
                        if img_size1 >= img_size2:
                            if results:
                                if len(results[row_num]) > 0:
                                    results[row_num].pop(col)
                        else:
                            if results:
                                if len(results[first_num]) > 0:
                                    results[first_num].pop(col)
                                    first_num = row_num
                                    car1 = results[row_num][col][0]
                                    gray1 = cv2.cvtColor(car1, cv2.COLOR_BGR2GRAY)
                                    img_size1 = img_size2
                                    break
        return results

    def detect(self, images):
        results = []  # [(vehicle, confiablity, boxsize, coordinates, img_detected),] 
        image_num = 0
        images_list = []
        for img in images:
            #img = cv2.imread(image_file)  # Read images in ftp_images/ dir
            _, detections = self.detector.detectCustomObjectsFromImage(custom_objects=self.custom,
                                                                       input_type="array",
                                                                       input_image=img,
                                                                       output_type='array',
                                                                       minimum_percentage_probability=self.threshold)
            cars = []
            count = 0
            for item in detections:
                if item["name"] in ('car', 'truck', 'suitcase', 'motorcycle', 'bus'):
                    x1,y1,x2,y2 = item["box_points"]
                    box_size = (x2-x1)*(y2-y1)

                    cars.append((img[y1:y2, x1:x2],
                                 box_size, 
                                 (item["name"], item["percentage_probability"], box_size, item["box_points"],
                                  img[y1:y2, x1:x2], img.shape, image_num)
                                ))
            if cars:        
                images_list.append(cars)

        #results = self.remove_equalscars(images_list)  # get index of uniques cars with max sizes
        results = images_list
        results = list(filter(lambda x: len(x) > 0, results))
        results = list(map(lambda x: x[0][2], results))
        #results = get_center_square(results)
        #results = list(sorted(results, key=lambda x: (x[2], x[1]), reverse=True))
        return results
