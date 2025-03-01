Software recopilador de datos de sustrato usando un sensor de suelo TH-PH de halisense. 

# Materiales

* Sensor TH-PH de salida RS485
* Adaptador RS485 a USB
* Computadora que sirva como cliente lector del sensor (por ejemplo, un Raspberry Pi)
* Servidor que sirva como backend
* Alguna forma de comunicación entre ambas computadoras (puede ser un vpn, o un reverse proxy)

# Instrucciones

Las imágenes de docker para el backend y el cliente pueden ser construidas con sus respectivos Dockerfile's, o ser descargadas usando
`docker pull cooldockerizer93/tuxedb` y `docker pull cooldockerizer93/tuxclient` respectivamente.

## Backend

Levanta el contenedor usando archivo docker-compose.yaml correspondiente.

## Client

Conecta tu sensor RS485 al adaptador USB, conectalo a la máquina cliente, encuentra el nombre del dispositivo bajo el directorio /dev/ (normalmente es algo como ttyUSB0) y mapealo de manera correspondiente en el docker-compose.yaml en el directorio client/, levanta el contenedor. 

Para ambos contenedores, no olvides configurar tus variables de entorno según sea necesario. Puedes comprobar si tu backend está disponible desde tu máquina cliente usando `curl "{url_basse}/mediciones"`. El servidor debería de responder con un array de todas las mediciones disponibles (o un array vacío, de no haber ninguna).

