#!/usr/bin/env python3
import minimalmodbus
import serial
import time
import requests
import sqlite3
from datetime import datetime
import os
import board
import adafruit_dht

# Respect the TZ environment variable (Unix only)
if "TZ" in os.environ:
    time.tzset()

# Ensure the backup directory exists and set the database path
BACKUP_DIR = "backup"
os.makedirs(BACKUP_DIR, exist_ok=True)
BACKUP_DB_PATH = os.path.join(BACKUP_DIR, "backup.db")

# Inicializar el sensor DHT22 una sola vez.
try:
    dht22_sensor = adafruit_dht.DHT22(board.D4)
    print("Sensor DHT22 inicializado correctamente.")
except Exception as e:
    print(f"Error inicializando el sensor DHT22: {e}")
    dht22_sensor = None

def get_weather_fallback():
    """
    Consulta la API de OpenWeatherMap para obtener la temperatura y humedad
    actual usando unidades métricas.
    
    Si la variable de entorno LOCATION está definida, se utiliza como la ubicación.
    Si no está definida, se omite la consulta y se devuelven 0.0 para ambas mediciones.
    """
    location = os.getenv("LOCATION")
    if not location:
        print("No se proporcionó la variable LOCATION; se omite la consulta de OpenWeatherMap.")
        return 0.0, 0.0
    appid = os.getenv("W_API_KEY")
    if not appid:
        raise ValueError("La variable de entorno W_API_KEY no está configurada.")
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={appid}&units=metric"
    try:
        response = requests.get(weather_url, timeout=5)
        if response.status_code == 200:
            weather_data = response.json()
            temp_ext = weather_data["main"]["temp"]
            hum_ext = weather_data["main"]["humidity"]
            print(f"Datos meteorológicos obtenidos de OpenWeatherMap para '{location}': Temp_ext = {temp_ext}°C, Hum_ext = {hum_ext}%")
        else:
            print(f"Error al obtener datos meteorológicos: {response.status_code} - {response.text}")
            temp_ext = 0.0
            hum_ext = 0.0
    except Exception as e:
        print(f"Excepción al obtener datos meteorológicos: {e}")
        temp_ext = 0.0
        hum_ext = 0.0
    return temp_ext, hum_ext

def get_weather():
    """
    Intenta obtener la temperatura y la humedad del sensor DHT22.
    Si falla, se utiliza la API de OpenWeatherMap (si LOCATION está definida)
    como respaldo.
    """
    global dht22_sensor
    if dht22_sensor is None:
        print("Sensor DHT22 no disponible. Usando OpenWeatherMap como respaldo.")
        return get_weather_fallback()
    try:
        temp_ext = dht22_sensor.temperature
        hum_ext = dht22_sensor.humidity
        print(f"Datos del sensor DHT22: Temp_ext = {temp_ext:.1f}°C, Hum_ext = {hum_ext:.1f}%")
        return temp_ext, hum_ext
    except Exception as error:
        print(f"Error al leer sensor DHT22: {error}. Usando OpenWeatherMap como respaldo.")
        return get_weather_fallback()

def generate_id(timestamp, temp_int, hum_int, ph, temp_ext, hum_ext):
    ts_str = timestamp.strftime("%Y-%m-%d-%H-%M-%S")
    data_str = f"{ts_str}-{temp_int}-{hum_int}-{ph}-{temp_ext}-{hum_ext}"
    return data_str

def init_local_backup_db():
    conn = sqlite3.connect(BACKUP_DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS mediciones (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            temp_int REAL NOT NULL,
            hum_int REAL NOT NULL,
            ph REAL NOT NULL,
            temp_ext REAL NOT NULL,
            hum_ext REAL NOT NULL,
            synced INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def save_local_backup(data):
    conn = sqlite3.connect(BACKUP_DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT OR IGNORE INTO mediciones (id, timestamp, temp_int, hum_int, ph, temp_ext, hum_ext, synced)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (data['id'], data['timestamp'], data['temp_int'], data['hum_int'],
          data['ph'], data['temp_ext'], data['hum_ext'], 0))
    conn.commit()
    conn.close()

def mark_as_synced(record_id):
    conn = sqlite3.connect(BACKUP_DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE mediciones SET synced = 1 WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

def sync_unsynced(backend_url):
    conn = sqlite3.connect(BACKUP_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, timestamp, temp_int, hum_int, ph, temp_ext, hum_ext FROM mediciones WHERE synced = 0")
    unsynced_records = c.fetchall()
    conn.close()
    
    for record in unsynced_records:
        data = {
            "id": record[0],
            "timestamp": record[1],
            "temp_int": record[2],
            "hum_int": record[3],
            "ph": record[4],
            "temp_ext": record[5],
            "hum_ext": record[6]
        }
        try:
            response = requests.post(backend_url, json=data, timeout=5)
            if response.status_code == 201:
                print(f"Registro {record[0]} sincronizado correctamente (por reintento).")
                mark_as_synced(record[0])
            else:
                print(f"Error al sincronizar el registro {record[0]}: {response.status_code} - {response.text}")
        except Exception as http_err:
            print(f"Excepción al sincronizar el registro {record[0]}: {http_err}")

def sync_from_remote(last_url):
    """
    Sincroniza insertando en el respaldo local las mediciones que aún no se tienen.
    Se utiliza el endpoint /mediciones/last para obtener únicamente las últimas 5 mediciones.
    """
    try:
        # Se consulta el endpoint con el parámetro q=5 para obtener las últimas 5 mediciones
        url = last_url + "?q=5"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            remote_records = response.json()
        else:
            print(f"Error al obtener datos remotos: {response.status_code} - {response.text}")
            return
    except Exception as e:
        print(f"Excepción al obtener datos remotos: {e}")
        return

    conn = sqlite3.connect(BACKUP_DB_PATH)
    c = conn.cursor()
    for med in remote_records:
        med_id = med["id"]
        c.execute("SELECT 1 FROM mediciones WHERE id = ?", (med_id,))
        if not c.fetchone():
            print(f"Insertando registro remoto faltante {med_id} en el respaldo local.")
            c.execute('''
                INSERT INTO mediciones (id, timestamp, temp_int, hum_int, ph, temp_ext, hum_ext, synced)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (med_id, med["timestamp"], med["temp_int"], med["hum_int"],
                  med["ph"], med["temp_ext"], med["hum_ext"], 1))
    conn.commit()
    conn.close()

def main():
    sleep_duration = float(os.getenv("SLEEP_DURATION", 1))
    init_local_backup_db()
    
    instrument = minimalmodbus.Instrument('/dev/ttyUSB0', 1)
    instrument.serial.baudrate = 4800
    instrument.serial.bytesize = 8
    instrument.serial.parity   = serial.PARITY_NONE
    instrument.serial.stopbits = 1
    instrument.serial.timeout  = 1
    instrument.mode = minimalmodbus.MODE_RTU
    
    base_url = os.getenv("BASE_URL", "http://localhost:6060")
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    mediciones_url = base_url + "/mediciones"
    mediciones_last_url = base_url + "/mediciones/last"
    backend_url = mediciones_url  # Para POST

    print(f"Usando backend URL para POST: {backend_url}")
    print(f"Usando backend Last URL para GET: {mediciones_last_url}")
    
    while True:
        try:
            humidity_raw = instrument.read_register(registeraddress=0, number_of_decimals=0, functioncode=3, signed=False)
            humidity = humidity_raw / 10.0

            temperature_raw = instrument.read_register(registeraddress=1, number_of_decimals=0, functioncode=3, signed=True)
            temperature = temperature_raw / 10.0

            ph_raw = instrument.read_register(registeraddress=3, number_of_decimals=0, functioncode=3, signed=False)
            ph = ph_raw / 10.0

            print(f"Humedad: {humidity:.1f}%  |  Temperatura: {temperature:.1f}°C  |  pH: {ph:.1f}")

            # Obtener datos meteorológicos desde el sensor DHT22 o, en caso de error, de OpenWeatherMap
            temp_ext, hum_ext = get_weather()

            timestamp = datetime.now()
            med_id = generate_id(timestamp, temperature, humidity, ph, temp_ext, hum_ext)
            data = {
                "id": med_id,
                "timestamp": timestamp.isoformat(),
                "temp_int": temperature,
                "hum_int": humidity,
                "ph": ph,
                "temp_ext": temp_ext,
                "hum_ext": hum_ext
            }
            
            save_local_backup(data)
            
            try:
                response = requests.post(backend_url, json=data, timeout=5)
                if response.status_code == 201:
                    print("Medición enviada correctamente al backend.")
                    mark_as_synced(med_id)
                else:
                    print(f"Error al enviar la medición: {response.status_code} - {response.text}")
            except Exception as http_err:
                print(f"Excepción al enviar la medición: {http_err}")
            
            sync_unsynced(backend_url)
            # Se sincronizan las últimas 5 mediciones desde el endpoint /mediciones/last?q=5
            sync_from_remote(mediciones_last_url)
        except Exception as e:
            print(f"Error al leer los datos: {e}")
        
        time.sleep(sleep_duration)

if __name__ == '__main__':
    main()
