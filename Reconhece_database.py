import os
import csv
from datetime import datetime, timedelta
from imutils import face_utils
import dlib
import cv2
import face_recognition
import numpy as np

# Inicializar o detector de faces e o preditor de marcos faciais
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# Capturar vídeo da webcam
cap = cv2.VideoCapture(0)

# Função para calcular a média das codificações faciais
def compute_mean_encoding(encodings):
    if len(encodings) == 0:
        return None
    mean_encoding = np.mean(encodings, axis=0)
    return mean_encoding

# Carregar as imagens conhecidas e extrair as codificações faciais
known_face_encodings = []
known_face_names = []
person_ids = []

known_faces_dir = "fotos"
person_id = 1

for person_name in os.listdir(known_faces_dir):
    person_dir = os.path.join(known_faces_dir, person_name)
    if os.path.isdir(person_dir):
        encodings = []
        for filename in os.listdir(person_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(person_dir, filename)
                image = face_recognition.load_image_file(image_path)
                image_encodings = face_recognition.face_encodings(image)
                if image_encodings:
                    encodings.append(image_encodings[0])
        mean_encoding = compute_mean_encoding(encodings)
        if mean_encoding is not None:
            known_face_encodings.append(mean_encoding)
            known_face_names.append(person_name)
            person_ids.append(person_id)
            person_id += 1

# Inicializar a lista de reconhecimento e o horário de último reconhecimento
recognized_people = {}
recognition_interval = timedelta(minutes=30)
csv_file = "recognized_people.csv"

# Dados adicionais (carregar do CSV)
database_csv = "database.csv"
database = {}

# Carregar dados adicionais do CSV
if os.path.exists(database_csv):
    with open(database_csv, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            database[row["Nome"]] = row

# Função para registrar reconhecimentos em um arquivo CSV
def log_recognition(person_id, name, recognition_time):
    # Atualizar o banco de dados com a data da última visita
    if name in database:
        database[name]["Data da Última Visita"] = recognition_time.strftime("%Y-%m-%d %H:%M:%S")
        recognized_data = database[name]
    else:
        # Se o nome não estiver no banco de dados, adicione com informações padrão
        recognized_data = {
            "ID": person_id,
            "Nome": name,
            "Email": "",
            "CPF": "",
            "Telefone": "",
            "Interesse": "",
            "Comprou": "não",
            "Data da Última Visita": recognition_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        database[name] = recognized_data

    # Registrar no CSV de reconhecimentos
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            recognized_data["ID"],
            recognized_data["Nome"],
            recognized_data["Email"],
            recognized_data["CPF"],
            recognized_data["Telefone"],
            recognized_data["Interesse"],
            recognized_data["Comprou"],
            recognition_time.strftime("%Y-%m-%d %H:%M:%S")
        ])

    # Atualizar o banco de dados CSV
    with open(database_csv, mode='w', newline='') as file:
        fieldnames = ["ID", "Nome", "Email", "CPF", "Telefone", "Interesse", "Comprou", "Data da Última Visita"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for entry in database.values():
            writer.writerow(entry)

# Criar o arquivo CSV e escrever o cabeçalho se não existir
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Nome", "Email", "CPF", "Telefone", "Interesse", "Comprou", "Data da Última Visita"])

while True:
    # Ler imagem da webcam e convertê-la para escala de cinza
    ret, image = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar faces na imagem em escala de cinza
    rects = detector(gray, 0)

    # Para cada face detectada, prever marcos faciais
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Desenhar círculos em cada coordenada (x, y) dos marcos faciais
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # Extraindo a região da face detectada
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_image = image[y:y+h, x:x+w]
        rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Codificar a face detectada
        face_encodings = face_recognition.face_encodings(rgb_face_image)

        name = "Desconhecido"
        confidence = 0
        person_id = None
        if face_encodings:
            face_encoding = face_encodings[0]
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin()
            confidence = 1 - face_distances[best_match_index]  # Confiança como 1 - distância
            if face_recognition.compare_faces([known_face_encodings[best_match_index]], face_encoding)[0]:
                name = known_face_names[best_match_index]
                person_id = person_ids[best_match_index]
                current_time = datetime.now()
                if name not in recognized_people or (current_time - recognized_people[name]) >= recognition_interval:
                    log_recognition(person_id, name, current_time)  # Registrar reconhecimento
                    recognized_people[name] = current_time

        # Exibir o nome da pessoa e a precisão na imagem
        text = f"{name} ({confidence * 100:.2f}%)"
        cv2.putText(image, text, (rect.left(), rect.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)

    # Mostrar a imagem com os pontos de interesse
    cv2.imshow("Output", image)

    # Esperar pela tecla ESC para sair
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# Liberar os recursos
cv2.destroyAllWindows()
cap.release()
