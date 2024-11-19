import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import tempfile
import time
import io
import numpy as np
import torch
from torchvision import transforms  # Import transform

# Styling untuk tampilan
st.markdown(""" 
    <style>
    /* Styling untuk sidebar */
    .css-18e3th9 { 
        background: linear-gradient(to right, #6a11cb, #2575fc); 
        color: #fff; 
    } 

    /* Styling untuk judul */
    .css-1d391kg { 
        font-size: 2.5em; 
        color: #fff; 
        font-family: 'Arial', sans-serif; 
    } 

    /* Styling untuk tombol */
    .css-1v0mbdj { 
        background-color: #4CAF50; 
        color: white; 
        padding: 15px 32px; 
        font-size: 16px; 
        border-radius: 5px; 
        box-shadow: 0 4px 8px rgba(0,0,0,0.2); 
        transition: 0.3s; 
    } 

    .css-1v0mbdj:hover { 
        background-color: #45a049; 
        box-shadow: 0 6px 12px rgba(0,0,0,0.3); 
    } 

    /* Styling gambar */
    .stImage img { 
        border-radius: 10px; 
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); 
    } 

    /* Styling untuk video dan gambar */
    .stVideo, .stImage { 
        border-radius: 15px; 
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.1); 
        margin-bottom: 20px; 
    }

    /* Styling untuk sidebar logo */
    .css-1d391kg img { 
        border-radius: 15px; 
    }

    /* Styling untuk tombol sidebar */
    .css-1v0mbdj {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        font-size: 14px;
        transition: 0.3s;
    }

    .css-1v0mbdj:hover {
        background-color: #45a049;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    }

    /* Styling untuk kartu deteksi (gambar dan video) */
    .stVideo, .stImage {
        border-radius: 15px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }

    /* Styling untuk tampilan sidebar logo */
    .css-1d391kg img {
        border-radius: 15px;
    }

    /* Styling untuk header aplikasi */
    .css-1d391kg {
        font-size: 2.5em;
        color: white;
        font-family: 'Arial', sans-serif;
        text-align: center;
        margin-top: 20px;
    }

    /* Styling untuk teks */
    .css-1d391kg p {
        color: white;
        font-size: 1.2em;
        font-family: 'Arial', sans-serif;
        text-align: center;
    }
</style>

""", unsafe_allow_html=True)

# Mendefinisikan device (GPU jika tersedia, jika tidak gunakan CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model YOLOv8
@st.cache_resource
def load_model():
    model = YOLO('best.pt').to(device)  # Pindahkan model ke GPU atau CPU
    return model

model = load_model()

# Fungsi untuk menambahkan padding agar ukuran gambar sesuai dengan input model
def add_padding_to_square(image, target_size=(640, 640)):
    img_width, img_height = image.size
    pad_x = (target_size[0] - img_width) // 2
    pad_y = (target_size[1] - img_height) // 2

    # Menambahkan padding pada gambar untuk menjaga rasio aspek
    padded_image = Image.new("RGB", target_size, (128, 128, 128))  # Warna padding abu-abu
    padded_image.paste(image, (pad_x, pad_y))

    return padded_image

# Fungsi untuk mengonversi gambar PIL ke tensor dan mengirimkannya ke GPU
def pil_to_tensor(pil_img, device):
    transform = transforms.ToTensor()  # Transformasi untuk mengonversi gambar PIL ke tensor
    img_tensor = transform(pil_img)  # Konversi ke tensor
    return img_tensor.to(device)  # Pindahkan ke GPU jika tersedia

# Menambahkan logo dengan st.image sebagai tambahan
st.sidebar.image("logo-gundar.jpg", use_container_width=True, caption="Universitas Gunadarma")

# Menu Sidebar
st.sidebar.title("Menu")

# Judul aplikasi
st.title("KELOMPOK 4 NI BOS SENGGOL DONG")

# Pilihan Mode Deteksi
mode = st.sidebar.selectbox("Pilih Mode Deteksi", ("Unggah Video", "Unggah Gambar", "Deteksi Real-Time"))

# Mode deteksi berdasarkan pilihan
if mode == "Unggah Video":
    uploaded_file = st.file_uploader("Unggah video untuk dideteksi", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(uploaded_file.read())
            video_path = tmpfile.name

        st.video(uploaded_file)

        # Membuka video untuk diproses
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            st.error("Tidak dapat membuka video.")
        else:
            st.write("Memproses video...")

            # Progress bar
            progress_bar = st.progress(0)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames > 0:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Mengonversi frame ke format gambar RGB
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img)

                    # Menambahkan padding pada gambar agar ukurannya 640x640
                    padded_image = add_padding_to_square(pil_img)

                    # Mengonversi gambar ke tensor dan kirim ke GPU
                    img_tensor = pil_to_tensor(padded_image, device)

                    results = model(img_tensor)  # Deteksi objek

                    result_image = results[0].plot()

                    if isinstance(result_image, np.ndarray):
                        result_image = Image.fromarray(result_image)

                    # Menampilkan gambar hasil deteksi
                    st.image(result_image, caption="Hasil Deteksi Frame", use_container_width=True)

                    try:
                        # Menyimpan gambar sebagai file PNG dan menyediakan tombol unduh
                        with io.BytesIO() as buffer:
                            result_image.save(buffer, format="PNG")
                            buffer.seek(0)
                            st.download_button(
                                label="Unduh Hasil Deteksi Frame",
                                data=buffer,
                                file_name=f"deteksi_frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.png",  
                                mime="image/png",
                                key=f"download_frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}"  
                            )
                    except Exception as e:
                        st.error(f"Gagal menyimpan gambar: {e}")

                    # Menampilkan progress deteksi
                    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    progress_bar.progress(int(current_frame / total_frames * 100))

            else:
                st.error("Video tidak memiliki frame yang valid.")
            
            cap.release()

elif mode == "Unggah Gambar":
    uploaded_image = st.file_uploader("Unggah gambar untuk deteksi", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Gambar yang Diupload", use_container_width=True)

        # Menambahkan padding pada gambar agar ukurannya 640x640
        padded_image = add_padding_to_square(image)

        # Mengonversi gambar ke tensor dan kirim ke GPU
        img_tensor = pil_to_tensor(padded_image, device)
        results = model(img_tensor)  # Kirim gambar ke GPU untuk deteksi

        result_image = results[0].plot()

        if isinstance(result_image, np.ndarray):
            result_image = Image.fromarray(result_image)

        st.image(result_image, caption="Hasil Deteksi Gambar", use_container_width=True)

        try:
            with io.BytesIO() as buffer:
                result_image.save(buffer, format="PNG")
                buffer.seek(0)
                st.download_button(
                    label="Unduh Hasil Deteksi Gambar",
                    data=buffer,
                    file_name="deteksi_gambar.png",
                    mime="image/png"
                )
        except Exception as e:
            st.error(f"Gagal menyimpan gambar: {e}")

elif mode == "Deteksi Real-Time":
    # Menambahkan tombol Start dan Stop
    start_camera = st.button("Mulai Deteksi Real-Time")
    stop_camera = st.button("Stop Deteksi Real-Time")

    frame_placeholder = st.empty()

    # Variabel status untuk deteksi real-time
    if start_camera:
        st.session_state.detecting = True  # Mengaktifkan deteksi
    elif stop_camera:
        st.session_state.detecting = False  # Menonaktifkan deteksi

    if 'detecting' not in st.session_state:
        st.session_state.detecting = False  # Awalnya deteksi dimatikan

    # Mulai deteksi kamera jika 'detecting' True
    if st.session_state.detecting:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Tidak dapat membuka kamera.")
        else:
            st.write("Memulai deteksi... Tekan 'Stop' untuk menghentikan.")

            frame_count = 0  # Menambahkan hitungan frame

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Gagal mendapatkan frame dari kamera.")
                    break

                # Mengubah frame dari BGR ke RGB
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)

                # Menambahkan padding pada gambar agar ukurannya 640x640
                padded_image = add_padding_to_square(pil_img)

                # Mengonversi gambar ke tensor dan kirim ke GPU
                img_tensor = pil_to_tensor(padded_image, device)
                results = model(img_tensor)  # Deteksi objek

                result_image = results[0].plot()

                if isinstance(result_image, np.ndarray):
                    result_image = Image.fromarray(result_image)

                # Menampilkan gambar hasil deteksi secara dinamis di Streamlit
                frame_placeholder.image(result_image, caption="Deteksi Real-Time", use_container_width=True)

                # Menyimpan gambar hasil deteksi sebagai file sementara
                with io.BytesIO() as buffer:
                    result_image.save(buffer, format="PNG")
                    buffer.seek(0)
                    st.download_button(
                        label="Unduh Hasil Deteksi Frame",
                        data=buffer,
                        file_name=f"deteksi_frame_{frame_count}.png",  # Nama file berdasarkan frame
                        mime="image/png",
                        key=f"download_frame_{frame_count}"  # Membuat key unik untuk setiap frame
                    )

                frame_count += 1

                # Periksa jika kamera terputus atau frame habis
                if frame_count > 200 or not ret or not st.session_state.detecting:
                    break

            cap.release()
    else:
        st.write("Kamera Tidak Aktif.")
