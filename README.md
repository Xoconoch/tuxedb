# Tuxedb

## 1. Introducción

El proyecto tiene como objetivo la monitorización de parámetros ambientales en un mesocosmos mediante la integración de sensores de suelo y ambientales. El sistema se compone de dos aplicaciones principales:  
- **Cliente (tuxclient):** Ejecutado en una Raspberry Pi 4b, se encarga de la adquisición de datos de sensores y de su envío al backend.  
- **Backend (tuxedb):** Desarrollado en Flask, recibe y almacena las mediciones en una base de datos SQLite, ofreciendo además una API REST para consultas y gestión de datos.

---

## 2. Requisitos de Hardware

- **Raspberry Pi 4b:** Plataforma principal de ejecución.
- **DHT22:** Sensor para medir la temperatura y humedad ambiental exterior.
- **Sensor de suelo TH-PH RS485 (halisense):** Mide parámetros como humedad, temperatura interior y pH.
- **Adaptador RS485 a USB:** Facilita la comunicación entre el sensor TH-PH y el Raspberry Pi.

---

## 3. Componentes de Software

### 3.1. Cliente (client.py)
- **Funcionalidad:**  
  - Lee datos del sensor TH-PH a través de la interfaz RS485 usando el módulo `minimalmodbus`.
  - Mide la temperatura y humedad exterior con el sensor DHT22. Si éste falla, se utiliza la API de OpenWeatherMap como respaldo.
  - Genera un identificador único para cada medición basándose en la marca temporal y los valores registrados.
  - Realiza un respaldo local de las mediciones en una base de datos SQLite ubicada en el directorio `backup`.
  - Sincroniza las mediciones con el backend, reintentando envíos fallidos y actualizando el estado de sincronización.

### 3.2. Backend (app.py)
- **Funcionalidad:**  
  - Implementa una API REST utilizando Flask y SQLAlchemy.
  - Define un único modelo de datos (`Medicion`) para almacenar las mediciones con campos compactos.
  - Expone los siguientes endpoints:
    - **POST /mediciones:** Para recibir y almacenar nuevas mediciones.
    - **GET /mediciones:** Para obtener mediciones con filtros opcionales (rango de tiempo, valores mínimos/máximos, etc.).
    - **GET /mediciones/last:** Para recuperar la última medición registrada.

---

## 4. Instalación y Configuración

### 4.1. Cliente (Tuxclient)

#### Variables de Entorno
- **UID y GID:** Identificadores de usuario y grupo (configurables según el entorno).
- **W_API_KEY:** Clave API de OpenWeatherMap (opcional; se utiliza como respaldo en caso de fallo del sensor DHT22).
- **BASE_URL:** URL base del backend (por ejemplo, `http://tuxedb.local/` o `http://192.168.1.30:6060/`).
- **TZ:** Zona horaria.
- **SLEEP_DURATION:** Intervalo entre lecturas en segundos (por defecto, 60).
- **LOCATION:** Ubicación para la consulta de OpenWeatherMap (formato: `Ciudad,Código_País`).

#### Volúmenes y Dispositivos
- **Volumen:** Se monta el directorio `./backup` para almacenar la base de datos de respaldo.
- **Dispositivo:** Se asigna el dispositivo USB del sensor (`/dev/ttyUSB0`).

#### Ejemplo de Configuración Docker
```yaml
name: tuxclient
services:
  tuxclient:
    image: cooldockerizer93/tuxclient
    environment:
      - UID=1000  # Cambiar según necesidad
      - GID=1000  # Cambiar según necesidad
      - W_API_KEY=OpenWeatherMap_api_key
      - BASE_URL=http://tuxedb.local/
      - TZ=Your_Timezone
      - SLEEP_DURATION=60
      - LOCATION=City Name,Country_Code
    volumes:
      - ./backup:/app/backup
    devices:
      - "/dev/ttyUSB0:/dev/ttyUSB0"  # Ajustar según el dispositivo real
    restart: unless-stopped
```

### 4.2. Backend (Tuxedb)

#### Variables de Entorno
- **UID y GID:** Identificadores de usuario y grupo (ajustables según el entorno).

#### Puertos y Volúmenes
- **Puerto:** Se expone el puerto `6060` para el acceso a la API.
- **Volumen:** Se monta el directorio `./db` para la persistencia de la base de datos `mesocosmos.db`.

#### Ejemplo de Configuración Docker
```yaml
name: tuxedb
services:
  tuxedb:
    image: cooldockerizer93/tuxedb
    environment:
      - UID=1000  # Cambiar según necesidad
      - GID=1000  # Cambiar según necesidad
    ports:
      - "6060:6060"
    volumes:
      - ./db:/app/instance
```

---

## 5. Funcionamiento del Sistema

### 5.1. Cliente
- **Inicialización:**  
  Se configuran las variables de entorno, se inicializa el sensor DHT22 (una sola vez) y se asegura la existencia del directorio y la base de datos de respaldo.
  
- **Lectura de Sensores:**  
  - Se obtienen las mediciones del sensor TH-PH mediante Modbus RTU.
  - Se obtiene la temperatura y humedad exterior desde el sensor DHT22; en caso de error, se consulta la API de OpenWeatherMap.
  
- **Generación del ID y Respaldo:**  
  Se genera un identificador único para cada registro utilizando la marca temporal y los valores medidos. Los datos se respaldan localmente en una base de datos SQLite antes de intentar enviarlos al backend.
  
- **Sincronización:**  
  Se realizan reintentos de sincronización de registros no enviados, y se sincronizan los datos remotos con el respaldo local.

### 5.2. Backend
- **Gestión de Datos:**  
  La aplicación Flask utiliza SQLAlchemy para gestionar la base de datos SQLite. La tabla `mediciones` almacena cada registro con los siguientes campos:
  - `id` (clave primaria)
  - `timestamp`
  - `temp_int`
  - `hum_int`
  - `ph`
  - `temp_ext`
  - `hum_ext`
  
- **Endpoints de la API REST:**  
  - **POST /mediciones:** Recibe nuevos registros y los almacena, generando un identificador único basado en la marca temporal y los datos.
  - **GET /mediciones:** Permite la consulta de registros con múltiples filtros (por rangos, paginación, etc.).
  - **GET /mediciones/last:** Devuelve la última medición registrada.

---

## 6. API Endpoints

### 6.1. POST /mediciones
- **Descripción:**  
  Almacena una nueva medición.
- **Datos Requeridos (JSON):**
  - `timestamp`: Fecha y hora en formato ISO 8601.
  - `temp_int`: Temperatura interior (float).
  - `hum_int`: Humedad interior (float).
  - `ph`: pH (float).
  - `temp_ext`: Temperatura exterior (float).
  - `hum_ext`: Humedad exterior (float).
- **Respuesta Exitosa:**  
  Código 201 con mensaje de confirmación y el ID generado.

### 6.2. GET /mediciones
- **Descripción:**  
  Recupera mediciones almacenadas con posibilidad de aplicar filtros:
  - Filtros exactos y por rangos para pH, temperatura interior, humedad interior, temperatura y humedad exterior.
  - Filtros por rango de tiempo mediante los parámetros `inicio` y `fin` (ISO 8601).
  - Paginación mediante `offset` y `limit`.
- **Respuesta:**  
  Lista de mediciones en formato JSON ordenadas por fecha descendente.

### 6.3. GET /mediciones/last
- **Descripción:**  
  Devuelve la última medición registrada en la base de datos.
- **Respuesta:**  
  Datos de la última medición en formato JSON o un mensaje informativo en caso de ausencia de registros.

---

## 7. Consideraciones Adicionales

- **Respaldo Local y Sincronización:**  
  El cliente almacena localmente las mediciones en una base de datos SQLite y realiza intentos de sincronización con el backend, lo que permite garantizar la persistencia de los datos incluso en caso de fallos de conexión.
  
- **Manejo de Errores:**  
  Se implementa un manejo de excepciones tanto en el cliente como en el backend para gestionar errores en la lectura de sensores, en la comunicación con la API de OpenWeatherMap, y en la persistencia de datos.

- **Compatibilidad:**  
  El sistema está diseñado para operar con los componentes de hardware mencionados, y se ha validado su funcionamiento en un entorno Raspberry Pi 4b.

- **Imágenes Docker:**  
  Se ofrecen imágenes Docker para facilitar la implementación y el despliegue de los componentes:
  - **Cliente:** `cooldockerizer93/tuxclient`
  - **Backend:** `cooldockerizer93/tuxedb`
