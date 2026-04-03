# Real-time Speech Translator

Ứng dụng dịch giọng nói thời gian thực. Tự động nhận diện ngôn ngữ đang nói (kể cả bài hát), chuyển thành text, và dịch sang tiếng Việt — hiển thị subtitle nổi trên màn hình.

## Tính năng

- **Auto-detect ngôn ngữ**: Nhận diện 90+ ngôn ngữ tự động
- **Real-time**: Nghe → Nhận diện → Dịch → Hiển thị chỉ trong vài giây
- **Overlay subtitle**: Cửa sổ nổi luôn trên top, có thể kéo thả, thu nhỏ
- **Mic + System Audio**: Hỗ trợ cả microphone và âm thanh hệ thống
- **Voice Activity Detection**: Tự động phát hiện khi có người nói
- **Hotkeys**: Điều khiển bằng phím tắt, không cần mở app

## Cài đặt

### 1. Yêu cầu
- Python 3.9+
- Microphone (cho chế độ mic)
- Để nghe system audio: cần virtual audio device (xem bên dưới)

### 2. Cài đặt dependencies

```bash
cd realtime-translator
pip install faster-whisper deep-translator sounddevice numpy keyboard
```

### 3. Cài đặt System Audio (tuỳ chọn)

**Windows:**
- Vào Sound Settings → Recording → Click phải → Show Disabled Devices
- Bật "Stereo Mix"

**macOS:**
- Cài [BlackHole](https://github.com/ExistentialAudio/BlackHole): `brew install blackhole-2ch`
- Hoặc cài [Soundflower](https://github.com/mattingalls/Soundflower)

**Linux:**
- PulseAudio monitor source được detect tự động

## Sử dụng

### Chạy cơ bản
```bash
# Nghe từ mic, dịch sang tiếng Việt
python main.py

# Nghe từ system audio
python main.py --source system

# Dịch sang tiếng Anh
python main.py --target en

# Dùng model chính xác hơn (chậm hơn)
python main.py --model small
```

### Xem danh sách audio devices
```bash
python main.py --list-devices
```

### Tuỳ chỉnh đầy đủ
```bash
python main.py \
  --source mic \
  --target vi \
  --model base \
  --position bottom \
  --opacity 0.85 \
  --font-size 20 \
  --threshold 0.01
```

### Phím tắt
| Phím tắt | Chức năng |
|---|---|
| `Ctrl+Shift+T` | Bắt đầu / Dừng dịch |
| `Ctrl+Shift+M` | Chuyển Mic ↔ System Audio |
| `Ctrl+Shift+O` | Bật / Tắt overlay |
| `Ctrl+Shift+Q` | Thoát ứng dụng |

## Whisper Models

| Model | Kích thước | Tốc độ | Độ chính xác |
|---|---|---|---|
| `tiny` | ~75 MB | Rất nhanh | Thấp |
| `base` | ~150 MB | Nhanh | Tốt |
| `small` | ~500 MB | Trung bình | Rất tốt |
| `medium` | ~1.5 GB | Chậm | Xuất sắc |
| `large-v3` | ~3 GB | Rất chậm | Tốt nhất |

**Khuyến nghị**: Dùng `base` cho real-time, `small` nếu cần chính xác hơn.

Nếu có NVIDIA GPU, thêm `--device cuda` để tăng tốc đáng kể.

## Cấu trúc project

```
realtime-translator/
├── main.py              # File chính, kết nối tất cả modules
├── audio_capture.py     # Thu âm từ mic / system audio
├── speech_recognizer.py # Nhận diện giọng nói (Whisper)
├── translator.py        # Dịch thuật (Google Translate)
├── overlay.py           # Overlay subtitle trên màn hình
├── requirements.txt     # Dependencies
└── README.md            # Hướng dẫn này
```
