import os
import time
import configparser
from multiprocessing import Process, Queue
from datetime import datetime as dt
from base64 import b64encode
import numpy as np
import cv2
from PIL import Image

from utils import FTP, API, CarDetector


CONFIG_FILE = 'config.ini'
VERSION = '0.1'

sep = '/'

PRE_DATE_FILENAME = 'Patente_698582022_'  # 'LPR_C87925560_'  #
POST_DATE_FILENAME = '_LINE_CROSSING_DETECTION.jpg'  # '_VEHICLE_DETECTION.jpg'  #

queue = Queue()
proc_put_file = None


def check_folders(folders_list):
    for folder in folders_list:
        if not os.path.exists(folder):
            try:
                os.mkdir(folder)
            except Exception as ex:
                print('Could\'t create {} folder\n\nExiting..'.format(folder))


def put_file_in_queue(ftp):  # TODO: remove image from ftp server and check string path for ftp_files
    global queue
    images_processed = []
    ftp.create_file(ftp.trash_folder, folder=True)

    try:
        while True:
            if ftp.connect(login=True):
                exit(0)

            ftp_files = ftp.get_files_from_folder(ftp.folder)
            ftp_files = list(map(lambda x: x.split(sep)[-1] if sep in x else x, ftp_files))
            ftp_files = list(filter(lambda x: (x.startswith(PRE_DATE_FILENAME)) and (x not in images_processed), ftp_files))
            ftp_files = list(sorted(map(lambda x: dt.strptime(x, f'{PRE_DATE_FILENAME}%Y%m%d%H%M%S%f{POST_DATE_FILENAME}'),
                                        ftp_files), reverse=True))

            if not ftp_files:
                continue
            files = [ftp_files[0]]
            for file in ftp_files[1:]:
                if (files[-1] - file).seconds < 2:
                    files.append(file)
            print(f'Getting {len(files)} files')
            files = list(map(lambda x: f'{PRE_DATE_FILENAME}{x.strftime("%Y%m%d%H%M%S%f")[:-3]}{POST_DATE_FILENAME}', files))
            images = []
            for file in files:
                if file in images_processed:
                    continue
                if not ftp.get_file(ftp.folder + sep + file):
                    print(f'Could get {file}')
                    continue
                imagen = ftp.img_buff.getvalue()
                ftp.move_file(ftp.folder + sep + file, ftp.trash_folder + sep + file)
                images_processed.append(file)
                images.append((imagen, file))
            if images:
                queue.put(images)
            time.sleep(5)
    except KeyboardInterrupt:
        pass


def get_file_in_queue():
    global queue
    if not queue.empty():
        return queue.get()
    return []


def cut_and_save(car_image_path, plate_box, plate, plates_folder):
    # img_plate = cv2.imread(car_image_path)
    # y1, x1, y2, x2 = plate_box  # ymin, xmin, ymax, xmax
    # img_plate = img_plate[y1:y2, x1:x2]
    # cv2.imwrite(os.path.join(plates_folder, plate+'.jpg'), img_plate)
    img_plate = Image.open(car_image_path)
    img_plate = img_plate.crop(tuple(plate_box))
    img_plate.save(plates_folder + sep + plate+'.jpg')
    return img_plate


def check_folder_day(folder_path, prefix=''):
    folder = folder_path.split(os.path.sep)[-1]
    if folder != prefix+dt.now().strftime('%Y-%m-%d'):
        return f'{os.path.sep}'.join(folder_path.split(os.path.sep)[:-1]) + sep + prefix+dt.now().strftime('%Y-%m-%d')
    return folder_path


def main():
    global queue, sep, proc_put_file

    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    if (not config) or (not config.sections()):
        print('No se encontro el archivo de configuracion.')
        exit(0)

    ftp_images_folder = config['folders'].get('ftp_images', '')
    car_images_folder = config['folders'].get('car_images', '.')
    plates_folder = config['folders'].get('plates', '.')
    sep = config['folders'].get('folder_sep', '.')

    car_images_folder = car_images_folder + sep + 'cars_'+dt.now().strftime('%Y-%m-%d')
    if not os.path.exists(car_images_folder):
        os.mkdir(car_images_folder)

    plates_folder = plates_folder + sep + 'plates_' + dt.now().strftime('%Y-%m-%d')
    if not os.path.exists(plates_folder):
        os.mkdir(plates_folder)

    ftp = FTP(url=config['ftp'].get('server_url'),
              port=int(config['ftp'].get('server_port', '9999')),
              folder=config['ftp'].get('server_folder', '.'),
              user=config['ftp'].get('user', 'anonymous'),
              password=config['ftp'].get('password', '')
              )
    if ftp.connect():
        exit(0)
    if ftp.login():
        exit(0)
    # ftp.change_folder()

    api = API(api_url=config['api'].get('API_URL', ''),
              token=config['api'].get('API_TOKEN', ''),
              )

    car_detector = CarDetector(model=os.curdir + sep + config['car_detect'].get('model'),
                               threshold=int(config['car_detect'].get('threshold', '50')),
                               car_percent=float(config['car_detect'].get('car_percent', '4.5'))
                               )

    proc_put_file = Process(target=put_file_in_queue,
                            args=(ftp,)
                            )

    proc_put_file.start()

    print('Ready and working..\n')

    json_result = config['folders'].get('results', '.') + sep + f'plates_result_{dt.now().strftime("%Y-%m-%d")}.json'
    with open(json_result, 'a') as fp:
        try:
            fp.write('[\n')

            while True:
                check_folder_day(car_images_folder, 'cars_')
                check_folder_day(plates_folder, 'plates_')
                images = get_file_in_queue()
                if not images:
                    time.sleep(1)
                    continue
                images_list = []

                for image, image_name in images:
                    img = np.frombuffer(image, dtype=np.uint8)
                    img = cv2.imdecode(img, 1)
                    if ftp_images_folder:
                        cv2.imwrite(ftp_images_folder + sep + image_name, img)
                    images_list.append(img)
                # car_images, dict_unique_cars = car_detector.detect(images_list)

                dict_unique_cars = car_detector.detect(images_list)
                for key in dict_unique_cars.keys():
                    folder = car_images_folder + sep + key + dt.now().strftime('%Y-%m-%d_%H:%M:%S')
                    os.mkdir(folder)
                    for car in dict_unique_cars[key]:
                        car_image_path = folder + sep + 'car_' + str(len(os.listdir(folder))) + '.jpg'
                        if not cv2.imwrite(car_image_path, car):
                            print('No se pudo guardar la imagen en {}'.format(car_image_path))
                            continue

                        response = api.request(car_image_path)
                        result = response['results'] if 'results' in response else []
                        if len(result) == 0:
                            print('No results for plate')
                            continue
                        plate, plate_box, right_format = api.improve_plate(result)

                        img_plate = cut_and_save(car_image_path, plate_box, plate, plates_folder)
                        print('Plate detected: {}'.format(plate))
                        fp.write(str(dict(plate=plate, image_path=car_image_path)))  # b64encode(img_plate.tobytes())
                        fp.flush()
                        if not right_format:
                            continue
                        break
        except KeyboardInterrupt:
            fp.write(']\n')
            raise KeyboardInterrupt


if __name__ == '__main__':
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        while True:
            try:
                main()
            except KeyboardInterrupt:
                if (proc_put_file is not None) and proc_put_file.is_alive():
                    proc_put_file.kill()
                print('Shut down system...')
                break
            except Exception:
                if (proc_put_file is not None) and proc_put_file.is_alive():
                    proc_put_file.kill()
