import os
import urllib.request
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(
    page_title="Detección de Rostros y Género - OpenCV & Streamlit",
    page_icon="🧑",
    layout="wide"
)

PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

GENDER_PROTOTXT_URLS = [
    "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
    "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_deploy.prototxt"
]
GENDER_MODEL_URLS = [
    "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_net.caffemodel",
    "https://github.com/eveningglow/age-and-gender-classification/raw/master/model/gender_net.caffemodel"
]
GENEROS = ["Hombre", "Mujer"]
MEAN_GENERO = (78.4263377603, 87.7689143744, 114.895847746)

def descargar_con_respaldo(urls, destino, min_bytes=500):
    if os.path.exists(destino) and os.path.getsize(destino) >= min_bytes:
        return True
    for url in urls:
        try:
            urllib.request.urlretrieve(url, destino)
            if os.path.exists(destino) and os.path.getsize(destino) >= min_bytes:
                return True
        except Exception:
            continue
    return False

@st.cache_resource
def cargar_detectores():
    descargar_con_respaldo([PROTOTXT_URL], "deploy.prototxt", 500)
    descargar_con_respaldo([MODEL_URL], "res10_300x300_ssd_iter_140000.caffemodel", 1_000_000)

    net_dnn = cv2.dnn.readNetFromCaffe(
        "deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel"
    )

    haar_face = None
    if hasattr(cv2, "CascadeClassifier"):
        haar_face = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    net_genero = None
    ok_proto = descargar_con_respaldo(GENDER_PROTOTXT_URLS, "gender_deploy.prototxt", 500)
    ok_model = descargar_con_respaldo(GENDER_MODEL_URLS, "gender_net.caffemodel", 1_000_000)
    if ok_proto and ok_model:
        net_genero = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")

    return net_dnn, haar_face, net_genero

def clasificar_genero(frame_bgr, x1, y1, x2, y2, net_genero):
    if net_genero is None:
        return "Rostro", 0.0, (40, 200, 85)
    h, w = frame_bgr.shape[:2]
    pad_x = int((x2 - x1) * 0.15)
    pad_y = int((y2 - y1) * 0.15)
    rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    rx2, ry2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    recorte = frame_bgr[ry1:ry2, rx1:rx2]
    if recorte.size == 0:
        return "Rostro", 0.0, (40, 200, 85)

    blob = cv2.dnn.blobFromImage(recorte, 1.0, (227, 227), MEAN_GENERO, swapRB=False)
    net_genero.setInput(blob)
    preds = net_genero.forward()[0]
    idx = int(np.argmax(preds))
    genero = GENEROS[idx]
    conf = float(preds[idx])
    color = (220, 130, 40) if genero == "Hombre" else (180, 70, 220)
    return genero, conf, color

def dibujar_etiqueta(img, texto, x1, y1, color):
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_top = max(0, y1 - th - 8)
    cv2.rectangle(img, (x1, y_top), (min(w, x1 + tw + 8), y_top + th + 8), color, -1)
    cv2.putText(img, texto, (x1 + 4, y_top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def detectar_rostros(frame_bgr, net_dnn, haar_face, net_genero, metodo, conf_min):
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    cajas = []

    if metodo == "OpenCV DNN (ResNet-10 SSD)" or haar_face is None:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0)
        )
        net_dnn.setInput(blob)
        dets = net_dnn.forward()
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf >= conf_min:
                box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 > x1 and y2 > y1:
                    cajas.append((x1, y1, x2, y2))
    else:
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = haar_face.detectMultiScale(gris, scaleFactor=1.15, minNeighbors=5, minSize=(45, 45))
        for (x, y, rw, rh) in rects:
            cajas.append((int(x), int(y), int(x + rw), int(y + rh)))

    hombres, mujeres = 0, 0
    for (x1, y1, x2, y2) in cajas:
        genero, conf_g, color_caja = clasificar_genero(frame_bgr, x1, y1, x2, y2, net_genero)
        if genero == "Hombre":
            hombres += 1
        elif genero == "Mujer":
            mujeres += 1
        etiqueta = f"{genero} ({conf_g * 100:.0f}%)" if conf_g > 0 else genero
        cv2.rectangle(img, (x1, y1), (x2, y2), color_caja, 2)
        dibujar_etiqueta(img, etiqueta, x1, y1, color_caja)

    cv2.rectangle(img, (0, 0), (w, 30), (15, 20, 28), -1)
    cv2.putText(
        img,
        f"Rostros: {len(cajas)} | Hombre: {hombres} | Mujer: {mujeres}",
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2
    )
    return img, len(cajas), hombres, mujeres

st.title("Aplicación Web de Detección de Rostros y Género")
st.caption("Detección facial y clasificación Hombre / Mujer con Python, OpenCV y Streamlit")

net_dnn, haar_face, net_genero = cargar_detectores()

with st.sidebar:
    st.header("Parámetros de Detección")
    metodo = st.selectbox(
        "Detector facial de OpenCV:",
        ["OpenCV DNN (ResNet-10 SSD)", "OpenCV Haar Cascade"]
    )
    conf_min = st.slider("Umbral mínimo de confianza (DNN)", 0.25, 0.95, 0.50, 0.05)

tab_vivo, tab_foto = st.tabs(["Cámara en Vivo (Tiempo Real)", "Captura / Subir Imagen"])

with tab_vivo:
    st.info("Presiona **START** y autoriza el acceso a tu cámara web.")

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img_proc, _, _, _ = detectar_rostros(img, net_dnn, haar_face, net_genero, metodo, conf_min)
        return av.VideoFrame.from_ndarray(img_proc, format="bgr24")

    webrtc_streamer(
        key="opencv-face-gender-detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )

with tab_foto:
    fuente = st.radio("Selecciona origen:", ["Tomar foto con cámara", "Subir imagen"], horizontal=True)
    archivo = (
        st.camera_input("Capturar rostro")
        if fuente == "Tomar foto con cámara"
        else st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png", "webp"])
    )

    if archivo is not None:
        bytes_img = np.asarray(bytearray(archivo.read()), dtype=np.uint8)
        frame_img = cv2.imdecode(bytes_img, cv2.IMREAD_COLOR)
        if fuente == "Tomar foto con cámara":
            frame_img = cv2.flip(frame_img, 1)
        res_img, total_rostros, total_h, total_m = detectar_rostros(
            frame_img, net_dnn, haar_face, net_genero, metodo, conf_min
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Rostros detectados", total_rostros)
        c2.metric("Hombre", total_h)
        c3.metric("Mujer", total_m)
        st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_container_width=True)
