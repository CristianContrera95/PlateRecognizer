# Plate Recongnizer 

Sistema para renconcer autos en images y detectar las patentes de estos.

### Proyect structure
   | Name | Descripcion |
| --- | ---  |
| **models** | carpeta con un modelo [yolo-tiny](https://pjreddie.com/darknet/yolo/) entrenado para detectar los vehiculos |
| **car_images** | carpeta donde se guardaran por dia y por hora los autos detectados  |
| **plates** | carpeta donde se guardaran por dia las patentes detectadas |
| **output** | carpeta donde se guardaran por dia un json con la patente como texto y el path a la imagen de la patente |
| **config.ini** | Archivo de configuracion con los parametros para el sistema |
| **utils.py** | Funciones de utilidad del sistema (server ftp, API, vehicles detection) |
| **main.py** | modulo pricipal del sistema |
| **Dokcerfile** | Archivo Docker para crear el entorno de ejecucion |
| **requirements.txt** | archivo con las librerias a utilizar por el sistema |

###  Config file

| Field | Descripcion |
| --- | --- |
| **ftp_images** |  |
| **car_images** | carpeta local para gurdar las imagenes de los autos |
| **plates** | carpeta local para gurdar las imagenes de las patentes |
| **results** | carpeta local para gurdar los json con los resultados |
| **server_url** | ip del servidor ftp |
| **server_port** | puerto del servidor ftp |
| **server_folder** | carpeta en el server ftp donde estan las imagenes  |
| **user** | usuario para loguearse al servidor ftp |
| **password** | contraseña para loguearse al servidor ftp |
| **regions** | **deprecated** |
| **API_TOKEN** | token dado por platerecognizer web |
| **API_URL** | direccion de la api de de platerecognizer |
| **model** | path al model yolo para la deteccion de vehiculos |
| **threshold** | humbral para la deteccion de vehiculos (recomendamos entre 0.3 y 0.6) |
| **car_percent** | minimo porcetage del tamaño del auto en la imagen |

### RUN

para correr el proyecto se posiciona en la carpeta donde se escuentre el proyecto y se ejecuta:
```
 docker run --rm -v $PWD:/app -p 5000:5000 -it plates_detect
```

