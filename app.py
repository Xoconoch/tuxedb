from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import hashlib

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mesocosmos.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Modelo único para las mediciones con nombres compactos
class Medicion(db.Model):
    id = db.Column(db.String(16), primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    temp_int = db.Column(db.Float, nullable=False)
    hum_int = db.Column(db.Float, nullable=False)
    ph = db.Column(db.Float, nullable=False)
    temp_ext = db.Column(db.Float, nullable=False)
    hum_ext = db.Column(db.Float, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "temp_int": self.temp_int,
            "hum_int": self.hum_int,
            "ph": self.ph,
            "temp_ext": self.temp_ext,
            "hum_ext": self.hum_ext
        }

# Función para generar el ID de la medición
def generate_id(timestamp, temp_int, hum_int, ph, temp_ext, hum_ext):
    # Formatear el timestamp a YYYYMMDDHHMMSS
    ts_str = timestamp.strftime("%Y-%m-%d-%H-%M-%S")
    # Concatenar el timestamp y los valores medidos en un único string
    data_str = f"{ts_str}-{temp_int}-{hum_int}-{ph}-{temp_ext}-{hum_ext}"

    return data_str

# Endpoint para crear una nueva medición
@app.route('/mediciones', methods=['POST'])
def crear_medicion():
    data = request.get_json()
    
    # Validar que se envíe información
    if not data:
        return jsonify({"error": "No se recibió ningún dato."}), 400

    # Lista de campos requeridos (nombres compactos)
    required_fields = [
        "timestamp",
        "temp_int",
        "hum_int",
        "ph",
        "temp_ext",
        "hum_ext"
    ]
    
    # Verificar campos faltantes
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({"error": f"Campos faltantes: {', '.join(missing_fields)}"}), 400
    
    # Convertir el timestamp de string a objeto datetime (formato ISO 8601)
    try:
        timestamp = datetime.fromisoformat(data["timestamp"])
    except ValueError:
        return jsonify({"error": "El formato del timestamp es incorrecto. Use ISO 8601."}), 400

    # Convertir los demás campos a float
    try:
        temp_int = float(data["temp_int"])
        hum_int = float(data["hum_int"])
        ph = float(data["ph"])
        temp_ext = float(data["temp_ext"])
        hum_ext = float(data["hum_ext"])
    except ValueError:
        return jsonify({"error": "Los valores de las métricas deben ser numéricos."}), 400

    # Generar el ID único de la medición
    med_id = generate_id(timestamp, temp_int, hum_int, ph, temp_ext, hum_ext)

    # Crear la instancia de la medición
    nueva_medicion = Medicion(
        id=med_id,
        timestamp=timestamp,
        temp_int=temp_int,
        hum_int=hum_int,
        ph=ph,
        temp_ext=temp_ext,
        hum_ext=hum_ext
    )

    # Guardar la medición en la base de datos
    try:
        db.session.add(nueva_medicion)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Error al guardar la medición en la base de datos.", "detalle": str(e)}), 500

    return jsonify({"mensaje": "Medición guardada exitosamente.", "id": med_id}), 201

# Endpoint para obtener mediciones con filtros opcionales
@app.route('/mediciones', methods=['GET'])
def obtener_mediciones():
    query = Medicion.query

    # Filtro exacto para pH (si se quiere filtrar por un valor específico)
    if 'ph' in request.args:
        try:
            ph_val = float(request.args.get('ph'))
            query = query.filter(Medicion.ph == ph_val)
        except ValueError:
            return jsonify({"error": "El valor de ph debe ser numérico."}), 400

    # Filtros por rango para pH
    if 'ph_min' in request.args:
        try:
            ph_min = float(request.args.get('ph_min'))
            query = query.filter(Medicion.ph >= ph_min)
        except ValueError:
            return jsonify({"error": "El valor de ph_min debe ser numérico."}), 400
    if 'ph_max' in request.args:
        try:
            ph_max = float(request.args.get('ph_max'))
            query = query.filter(Medicion.ph <= ph_max)
        except ValueError:
            return jsonify({"error": "El valor de ph_max debe ser numérico."}), 400

    # Filtros por rango para temperatura interior
    if 'temp_int_min' in request.args:
        try:
            temp_int_min = float(request.args.get('temp_int_min'))
            query = query.filter(Medicion.temp_int >= temp_int_min)
        except ValueError:
            return jsonify({"error": "El valor de temp_int_min debe ser numérico."}), 400
    if 'temp_int_max' in request.args:
        try:
            temp_int_max = float(request.args.get('temp_int_max'))
            query = query.filter(Medicion.temp_int <= temp_int_max)
        except ValueError:
            return jsonify({"error": "El valor de temp_int_max debe ser numérico."}), 400

    # Filtros por rango para humedad interior
    if 'hum_int_min' in request.args:
        try:
            hum_int_min = float(request.args.get('hum_int_min'))
            query = query.filter(Medicion.hum_int >= hum_int_min)
        except ValueError:
            return jsonify({"error": "El valor de hum_int_min debe ser numérico."}), 400
    if 'hum_int_max' in request.args:
        try:
            hum_int_max = float(request.args.get('hum_int_max'))
            query = query.filter(Medicion.hum_int <= hum_int_max)
        except ValueError:
            return jsonify({"error": "El valor de hum_int_max debe ser numérico."}), 400

    # Filtros por rango para temperatura exterior
    if 'temp_ext_min' in request.args:
        try:
            temp_ext_min = float(request.args.get('temp_ext_min'))
            query = query.filter(Medicion.temp_ext >= temp_ext_min)
        except ValueError:
            return jsonify({"error": "El valor de temp_ext_min debe ser numérico."}), 400
    if 'temp_ext_max' in request.args:
        try:
            temp_ext_max = float(request.args.get('temp_ext_max'))
            query = query.filter(Medicion.temp_ext <= temp_ext_max)
        except ValueError:
            return jsonify({"error": "El valor de temp_ext_max debe ser numérico."}), 400

    # Filtros por rango para humedad exterior
    if 'hum_ext_min' in request.args:
        try:
            hum_ext_min = float(request.args.get('hum_ext_min'))
            query = query.filter(Medicion.hum_ext >= hum_ext_min)
        except ValueError:
            return jsonify({"error": "El valor de hum_ext_min debe ser numérico."}), 400
    if 'hum_ext_max' in request.args:
        try:
            hum_ext_max = float(request.args.get('hum_ext_max'))
            query = query.filter(Medicion.hum_ext <= hum_ext_max)
        except ValueError:
            return jsonify({"error": "El valor de hum_ext_max debe ser numérico."}), 400

    # Filtros por rango de tiempo (parámetros 'inicio' y 'fin' en formato ISO 8601)
    if 'inicio' in request.args:
        try:
            inicio_dt = datetime.fromisoformat(request.args.get('inicio'))
            query = query.filter(Medicion.timestamp >= inicio_dt)
        except ValueError:
            return jsonify({"error": "El formato del parámetro 'inicio' es incorrecto. Use ISO 8601."}), 400
    if 'fin' in request.args:
        try:
            fin_dt = datetime.fromisoformat(request.args.get('fin'))
            query = query.filter(Medicion.timestamp <= fin_dt)
        except ValueError:
            return jsonify({"error": "El formato del parámetro 'fin' es incorrecto. Use ISO 8601."}), 400

    # Parámetros de paginación
    if 'offset' in request.args:
        try:
            offset_val = int(request.args.get('offset'))
            query = query.offset(offset_val)
        except ValueError:
            return jsonify({"error": "El valor de offset debe ser entero."}), 400

    if 'limit' in request.args:
        try:
            limit_val = int(request.args.get('limit'))
            query = query.limit(limit_val)
        except ValueError:
            return jsonify({"error": "El valor de limit debe ser entero."}), 400

    # Ordenar resultados por timestamp descendente
    query = query.order_by(Medicion.timestamp.desc())

    mediciones = query.all()
    mediciones_list = [med.to_dict() for med in mediciones]
    return jsonify(mediciones_list), 200

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=6060, debug=True)
