import librosa
from rmvpe import E2E, SAMPLE_RATE, extract_melody

audio_path = r"test.wav"

y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
model = E2E(hop_length=int(0.02*SAMPLE_RATE), n_blocks=4, n_gru=1, kernel_size=(2, 2))
# 如有权重：model.load_state_dict(torch.load("checkpoint.pt", map_location="cpu"))

cents, _ = extract_melody(y.astype("float32"), model, hop_length_ms=20)
print(cents[:10])  # 每帧美分，0 表示无声

