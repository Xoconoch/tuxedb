Software recopilador de datos de sustrato usando un sensor de suelo TH-PH de halisense. 

Materiales necesarios para usar este proyecto:

* Sensor TH-PH de salida RS485
* Adaptador RS485 a USB
* Computadora que sirva como cliente lector del sensor (por ejemplo, un Raspberry Pi)
* Servidor que sirva como backend
* Alguna forma de comunicación entre ambas computadoras (puede ser un vpn, o un reverse proxy)

Las imágenes de docker para el backend y el cliente pueden ser construidas con sus respectivos Dockerfile's, o ser descargadas usando
`docker pull cooldockerizer93/tuxedb` y `docker pull cooldockerizer93/tuxclient` respectivamente.