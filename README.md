<img width="1600" height="929" alt="image" src="https://github.com/user-attachments/assets/57b6cfbb-7ad6-4fd8-9a5d-80f1d08a7c35" />
Aplikasi deteksi keselamatan pengendara motor menggunakan YOLO.

.ptnya kok gk ada bang? Join Discord https://discord.gg/KsEwEQyVDq 

📋 Persyaratan

- Windows 10/11
- Webcam atau file video
- Model YOLO (.pt, sangat disarankan)
- NVIDIA GTX/RTX (Untuk akselerasi GPU)

🚀 Cara Penggunaan

1. Persiapkan File

📁 Folder Aplikasi/
├── Deteksi_Keselamatan_Berkendara_Sepeda_Motor.exe
├── 📁 Voice/ (auto-detect)
│   ├── helm.mp3
│   ├── pakaian.mp3
│   └── sepatu.mp3
└── 📁 Model/ (opsional)
    └── best.pt

2. Jalankan Aplikasi
- Double click `Deteksi_Keselamatan_Berkendara_Sepeda_Motor.exe`
- Tunggu aplikasi loading

3. Konfigurasi
1. Pilih Model YOLO
   - Klik "Browse Model"
   - Pilih file `.pt` (disarankan) atau `.engine` (hanya untuk advanced user, lihat catatan di bawah)

2. Pilih Sumber Video
   - Webcam: Langsung gunakan kamera
   - Video File: Pilih file video (.mp4, .avi, dll)

3. Pilih Direktori Screenshot
   - Klik "Browse Direktori"
   - Pilih folder untuk menyimpan hasil

4. Mulai Deteksi
- Klik "Mulai Deteksi"
- Aplikasi akan menampilkan video dengan deteksi real-time

🎯 Fitur Deteksi

- ✅ Helm: Deteksi penggunaan helm
- ✅ Pakaian: Deteksi pakaian tertutup/terbuka
- ✅ Sepatu: Deteksi penggunaan sepatu
- 🔊 Audio: Peringatan suara otomatis
- 📸 Screenshot: Auto-capture pelanggaran
- 📊 Log Excel: Laporan lengkap

📁 Output

Screenshot
- Pelanggaran: Disimpan di `bad/`
- Kepatuhan: Tidak disimpan (hanya log Excel)

Log Excel
- File: `detection_log_YYYYMMDD_HHMMSS.xlsx`
- Berisi: Waktu, Jenis Deteksi, Confidence, Path Screenshot

⚙️ Pengaturan

Threshold Confidence
- Pelanggaran: ≥ 90%
- Kepatuhan: ≥ 90%
- Khusus untuk "Tidak Pakai Helm": threshold ≥ 85%

Interval
- Screenshot: 0.3 detik
- Audio: 8 detik

🖥️ Kompatibilitas Model

Format Model
- .pt (PyTorch): Kompatibel di semua sistem (DIREKOMENDASIKAN)
- .engine (TensorRT): Hanya untuk advanced user, TIDAK kompatibel di semua perangkat. Hanya bisa dijalankan pada perangkat & environment yang SAMA saat proses konversi. Jika Anda tidak yakin, gunakan .pt saja.

GPU NVIDIA Support
- GTX/RTX Series: Mendukung akselerasi GPU (gunakan .pt)
- Integrated/Laptop: Bisa jalan (gunakan .pt, lebih lambat)

Catatan Penting Model .engine:
- File .engine TIDAK portable, hanya bisa dijalankan pada perangkat & environment yang sama saat proses konversi.
- Jika model .engine gagal dijalankan, GANTI ke model .pt.
- Penggunaan .engine hanya untuk user yang paham proses konversi .pt → .onnx → .engine dan environment TensorRT.

Rekomendasi Model
- Semua pengguna: Gunakan model .pt (PyTorch) untuk kompatibilitas maksimal.
- Model .engine hanya untuk advanced user.

🎯 Troubleshooting

Aplikasi tidak buka: Jalankan sebagai Administrator
Webcam error: Pastikan tidak digunakan aplikasi lain
Audio tidak ada: Cek file MP3 di folder Voice
Model .engine error: Ganti dengan model .pt
GPU tidak terdeteksi: Pastikan driver NVIDIA terinstall
GTX/RTX/Integrated: Gunakan model .pt

🔧 Troubleshooting

Aplikasi Tidak Buka
- Pastikan antivirus tidak memblokir
- Jalankan sebagai Administrator

Webcam Tidak Terdeteksi
- Pastikan webcam terhubung
- Cek permission aplikasi

Audio Tidak Berfungsi
- Pastikan speaker/headphone terhubung
- Aplikasi akan auto-detect file suara di folder Voice

Model Tidak Ditemukan
- Pastikan file model ada
- Cek format file (.pt/.engine)
- Jika .engine error, ganti ke .pt

Binggung?? Join Discord https://discord.gg/KsEwEQyVDq
