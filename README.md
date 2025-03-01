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

---

Para ambos contenedores, no olvides configurar tus variables de entorno según sea necesario. Puedes comprobar si tu backend está disponible desde tu máquina cliente usando `curl "http://{ip_backend}:6060/mediciones"`. El servidor debería de responder con un array de todas las mediciones disponibles (o un array vacío, de no haber ninguna).

# API

El backend tiene una api que puedes implementar en alguna otra aplicación (o sencillamente puedes usar la base de datos de manera directa)
---

## Endpoint: POST /mediciones

**Descripción:**  
Crea una nueva medición en la base de datos del mesocosmos.

**Método HTTP:**  
POST

**URL:**  
`/mediciones`

**Cuerpo de la Solicitud (JSON):**  
Debe enviarse un objeto JSON con los siguientes campos obligatorios:
- **timestamp** (string): Fecha y hora de la medición en formato ISO 8601.  
  _Ejemplo:_ `"2025-02-28T14:30:00"`
- **temp_int** (number): Temperatura interna.
- **hum_int** (number): Humedad interna.
- **ph** (number): pH del medio.
- **temp_ext** (number): Temperatura externa.
- **hum_ext** (number): Humedad externa.

**Ejemplo de Solicitud:**
```json
{
  "timestamp": "2025-02-28T14:30:00",
  "temp_int": 25.5,
  "hum_int": 60.2,
  "ph": 6.8,
  "temp_ext": 28.0,
  "hum_ext": 70.1
}
```

**Respuesta Exitosa (201):**  
Se retorna un objeto JSON indicando que la medición fue guardada exitosamente, junto con el ID generado para la medición.
```json
{
  "mensaje": "Medición guardada exitosamente.",
  "id": "2025-02-28-14-30-00-25.5-60.2-6.8-28.0-70.1"
}
```

**Errores Potenciales:**
- **400 Bad Request:**  
  - Si no se envía ningún dato.
  - Si faltan campos obligatorios (se indicará cuáles faltan).
  - Si el `timestamp` no tiene el formato ISO 8601.
  - Si alguno de los valores de las métricas no es numérico.
- **500 Internal Server Error:**  
  - Si ocurre un error al guardar la medición en la base de datos (se incluye el detalle del error).

---

## Endpoint: GET /mediciones

**Descripción:**  
Recupera una lista de mediciones almacenadas en la base de datos, permitiendo la aplicación de filtros opcionales por rangos y valores exactos.

**Método HTTP:**  
GET

**URL:**  
`/mediciones`

**Parámetros de Consulta (Query):**  
- **ph** (number, opcional): Filtra mediciones que tengan exactamente este valor de pH.
- **ph_min** (number, opcional): Filtra mediciones con pH mayor o igual a este valor.
- **ph_max** (number, opcional): Filtra mediciones con pH menor o igual a este valor.
- **temp_int_min** (number, opcional): Valor mínimo de temperatura interna.
- **temp_int_max** (number, opcional): Valor máximo de temperatura interna.
- **hum_int_min** (number, opcional): Valor mínimo de humedad interna.
- **hum_int_max** (number, opcional): Valor máximo de humedad interna.
- **temp_ext_min** (number, opcional): Valor mínimo de temperatura externa.
- **temp_ext_max** (number, opcional): Valor máximo de temperatura externa.
- **hum_ext_min** (number, opcional): Valor mínimo de humedad externa.
- **hum_ext_max** (number, opcional): Valor máximo de humedad externa.
- **inicio** (string, opcional): Fecha y hora mínima (formato ISO 8601) para filtrar las mediciones.
- **fin** (string, opcional): Fecha y hora máxima (formato ISO 8601) para filtrar las mediciones.
- **offset** (integer, opcional): Número de resultados a saltar (útil para paginación).
- **limit** (integer, opcional): Límite de resultados a retornar (útil para paginación).

**Ejemplo de URL con Parámetros:**
```
/mediciones?ph_min=6.5&ph_max=7.5&temp_int_min=20&temp_int_max=30&offset=0&limit=10
```

**Respuesta Exitosa (200):**  
Se retorna un array JSON de objetos, cada uno representando una medición. Cada objeto contiene:
- `id`
- `timestamp` (en formato ISO 8601)
- `temp_int`
- `hum_int`
- `ph`
- `temp_ext`
- `hum_ext`

**Ejemplo de Respuesta:**
```json
[
  {
    "id": "2025-02-28-14-30-00-25.5-60.2-6.8-28.0-70.1",
    "timestamp": "2025-02-28T14:30:00",
    "temp_int": 25.5,
    "hum_int": 60.2,
    "ph": 6.8,
    "temp_ext": 28.0,
    "hum_ext": 70.1
  },
  {
    "id": "2025-02-28-14-15-00-24.0-58.0-6.7-27.0-68.0",
    "timestamp": "2025-02-28T14:15:00",
    "temp_int": 24.0,
    "hum_int": 58.0,
    "ph": 6.7,
    "temp_ext": 27.0,
    "hum_ext": 68.0
  }
]
```

**Errores Potenciales:**
- **400 Bad Request:**  
  - Si algún parámetro numérico o de fecha tiene un formato incorrecto (por ejemplo, `ph_min` no es numérico o `inicio` no es una fecha en formato ISO 8601).

---

## Endpoint: GET /mediciones/last

**Descripción:**  
Recupera la última medición registrada en la base de datos, basada en el campo `timestamp`.

**Método HTTP:**  
GET

**URL:**  
`/mediciones/last`

**Respuesta Exitosa (200):**  
Se retorna un objeto JSON con la última medición registrada. El objeto incluye:
- `id`
- `timestamp` (en formato ISO 8601)
- `temp_int`
- `hum_int`
- `ph`
- `temp_ext`
- `hum_ext`

**Ejemplo de Respuesta:**
```json
{
  "id": "2025-02-28-14-30-00-25.5-60.2-6.8-28.0-70.1",
  "timestamp": "2025-02-28T14:30:00",
  "temp_int": 25.5,
  "hum_int": 60.2,
  "ph": 6.8,
  "temp_ext": 28.0,
  "hum_ext": 70.1
}
```

**Errores Potenciales:**
- **404 Not Found:**  
  - Si no existen mediciones en la base de datos, se retorna un mensaje indicando que no hay mediciones disponibles.
