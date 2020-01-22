import re
import os
import ftplib

from random import randint
from io import BytesIO

import numpy as np
from skimage.metrics import structural_similarity
from PIL import Image
import dlib
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
            except Exception as ex:
                print('Error in server address\n', ex)
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
                print(self.ftp.welcome, f'\nConnected to ftp server in {self.url}:{self.port}')
            except Exception as ex:
                if count == self.connect_tries:
                    print(f'Couldn\'t connect to {self.url}:{self.port} ftp-server.\nPlease check address')
                    return 1
            else:
                break
        if login:
            self.login()
        return 0

    def login(self):
        try:
            self.ftp.login(self.user, self.password)
            print(f'Logged in server with {self.user} user')
        except Exception as ex:
            print(f'Couldn\'t login with {self.user} user.\nPlease check user and password')

    def change_folder(self):
        try:
            self.ftp.cwd(self.folder)
        except Exception as ex:
            print(f'Couldn\'t find {self.folder} dir.\nYou do not have permissions or the directory does not exist')
            return 1
        return 0

    def close(self):
        pass

    def handle_binary(self, data):
        self.img_buff.write(data)

    def get_file(self, filename):
        try:
            result = self.ftp.retrbinary(f'RETR {os.path.join(self.folder, filename)}',
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
                    headers={'Authorization': f'Token {self.API_TOKEN}'}
                    )
        return response.json()

    def improve_plate(self, result):
        result = result[0] if isinstance(result, list) and (len(result) > 0) else result

        plate_box = list(result['box'].values())  # ymin, xmin, ymax, xmax
        plate_box = [plate_box[1]-10, plate_box[0]-10, plate_box[3]+10, plate_box[2]+10]
        plate = result['plate']
        is_new_plate = True if len(plate) == 7 else False

        for i in range(len(plate)):
            if (is_new_plate and (i in [2, 3, 4])) or ((not is_new_plate) and (i in [3, 4, 5])):
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
        x1, y1, x2, y2 = square[3]
        percen_centered_width = abs((((x1+x2)/2) * 100 / width) - 50)
        percen_centered_height = abs((((y1+y2)/2) * 100 / height) - 50)
        results.append((percen_centered_width, percen_centered_height, square))

    results = list(sorted(results, key=lambda x: (x[0], x[1]), reverse=True))
    return list(map(lambda x: x[2], results))


def box_size(box):
    return (box[2]-box[0])*(box[3]-box[1])


def nms(boxes, overlapThresh):
    """
    :param boxes: array of tuples like (x,y,w,h)
    :param overlapThresh: float used to represent overlap percentage
    """
    if len(boxes) <= 0:
        return []

    # get the coordinates of the all bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the boxes and sort the boxes by the bottom-right y-coordinate
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    pick = []
    while len(idxs) > 0:

        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in range(last):
            j = idxs[pos]

            # find the largest (x, y) coordinates for the top-left of the boxes and
            # the smallest (x, y) coordinates for the bottom-right of the boxes
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed box and the box in the area list
            overlap = float(w * h) / area[j]

            if overlap > overlapThresh:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    return boxes[pick]


class CarDetector:

    def __init__(self, model, threshold, car_percent):
        super(CarDetector, self).__init__()
        self.model = model
        self.threshold = threshold
        self.car_percent = car_percent
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsTinyYOLOv3()
        self.detector.setModelPath(os.path.join(os.curdir, self.model))
        self.detector.loadModel(detection_speed='normal')  # "normal, "fast", "faster", "fastest","flash".
        self.custom = self.detector.CustomObjects(car=True, motorcycle=True, bus=True, truck=True, suitcase=True)
        self.trackers = []
        self.count_cars = 0

    def detect(self, images):
        detected_cars = {}
        image_num = 0
        count = -1
        remove_images = []

        for img in images:  # TODO: check rgb in imagescar
            # iteration over images and detect vehicles in there
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            _, detections = self.detector.detectCustomObjectsFromImage(custom_objects=self.custom,
                                                                       input_type="array",
                                                                       input_image=img,
                                                                       output_type='array',
                                                                       minimum_percentage_probability=self.threshold)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            count += 1
            if not detections:
                remove_images.append(count)
                continue

            img_area = img.shape[0] * img.shape[1]
            image_num += 1
            detected_cars[f'image_{image_num}'] = []
            for item in detections:
                # Guardamos para cada vehiculo el box donde se encuenta en la imagen
                print('Area: ', (box_size(item['box_points'])*100 / img_area))
                if (box_size(item['box_points'])*100 / img_area) > self.car_percent:
                    # Solo Guardamos si el box es hasta 5 veces mas chica que la imagen
                    detected_cars[f'image_{image_num}'].append(tuple(item['box_points']))
                    self.count_cars += 1

            if len(detected_cars[f'image_{image_num}']) == 0:
                del detected_cars[f'image_{image_num}']
                image_num -= 1
                remove_images.append(count)
                continue
            else:
                # eliminamos los box sobrepuestos que puedan existir
                detected_cars[f'image_{image_num}'] = nms(np.array(detected_cars[f'image_{image_num}']), 0.3).tolist()

        # Remove without cars insisde
        offset = 0
        for index in remove_images:
            images.pop(index-offset)
            offset += 1

        dict_unique_cars = {}
        car_count = 0

        image_num = 0
        for img in images:  # iteramos sobre cada imagen para trackear cada auto en ella en las demas imagenes

            image_num += 1
            for box in detected_cars[f'image_{image_num}']:  # trackeamos cada auto en la imagen
                detected_cars[f'image_{image_num}'].remove(box)
                tracker = dlib.correlation_tracker()
                x1, y1, x2, y2 = box
                tracker.start_track(img, dlib.rectangle(x1, y1, x2, y2))

                car_count += 1
                dict_unique_cars[f'car_{car_count}'] = [img[y1:y2, x1:x2]]

                image_num2 = 0
                for img2 in images:  # iteramos sobre las demas imagenes buscando el auto
                    image_num2 += 1
                    if image_num != image_num2:
                        tracker.update(img2)
                        car = tracker.get_position()
                        car = (int(car.left()), int(car.top()), int(car.right()), int(car.bottom()))

                        # obtenemos los autos detectados en la imagen para comparar con el tracking del auto actual
                        boxs_of_cars = detected_cars[f'image_{image_num2}']

                        if len(boxs_of_cars) == len(nms(np.array(boxs_of_cars + [car]), 0.15)):
                            # True se el auto esta en la segunda imagen tambien

                            for box2 in boxs_of_cars:  # buscamos nuevamente el auto dentro de la segunda imagen
                                if len(nms(np.array([box2, car]), 0.15)) != 2:  # True si box2 y car son el mismo auto
                                    try:
                                        x1, y1, x2, y2 = box2
                                        detected_cars[f'image_{image_num2}'].remove(box2)

                                        dict_unique_cars[f'car_{car_count}'].append(img2[y1:y2, x1:x2])
                                    except Exception as ex:
                                        print(ex)
                                        pass

                                    break

        for key in dict_unique_cars.keys():
            dict_unique_cars[key] = list(sorted(dict_unique_cars[key], key=lambda x: x.size, reverse=True))

        return dict_unique_cars



"""
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
        amount_cars = []
        cars = []
        l = []
        li = []
        for img in images:
            
            _, detections = self.detector.detectCustomObjectsFromImage(custom_objects=self.custom,
                                                                       input_type="array",
                                                                       input_image=img,
                                                                       output_type='array',
                                                                       minimum_percentage_probability=self.threshold)
            if len(detections) == 0:
                images.remove(img)
            for item in detections:
                if item["name"] in ('car', 'truck', 'suitcase', 'motorcycle', 'bus'):
                    x1, y1, x2, y2 = item["box_points"]
                    box_size = (x2-x1)*(y2-y1)
                    cars.append((img[y1:y2, x1:x2],
                                 box_size,
                                 (item["name"], item["percentage_probability"], box_size, item["box_points"],
                                  img[y1:y2, x1:x2], img.shape, image_num),
                                 img))
            if cars:
                images_list.append(cars)
                cars = []

        index = images_list.index(max(images_list))
        for car in images_list.pop(index):
            self.trackers.append((car[-1], dlib.correlation_tracker().start_track(car[-1], dlib.rectangle(car[1]))))

        for cur_car, tracker in self.trackers:
            cur_car = cv2.cvtColor(cur_car, cv2.COLOR_BGR2GRAY)
            for img in images:
                if np.isnan(tracker.update(img)):
                    continue
                box = tracker.get_position()
                x1, y1, x2, y2 = box.left(), box.top(), box.right(), box.bottom()
                img = img[y1:y2, x1:x2]
                img = cv2.resize(img, (cur_car.shape[1], cur_car.shape[0]), interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if structural_similarity(cur_car, img) > 0.4:
                    cur_car = img if img.size > cur_car.size else cur_car

            results.append(cur_car)

        return results

"""