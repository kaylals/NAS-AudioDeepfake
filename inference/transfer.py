from pydub import AudioSegment
from pathlib import Path

input_dir = Path("human_voice")         # MP3 文件目录
output_dir = Path("converted_bonafide") # 输出 FLAC 文件目录
output_dir.mkdir(parents=True, exist_ok=True)

# 遍历所有 mp3 文件
for idx, mp3_file in enumerate(sorted(input_dir.glob("*.mp3"))):
    audio = AudioSegment.from_mp3(mp3_file)
    audio = audio.set_frame_rate(16000).set_channels(1)  # 转为16kHz、单声道

    new_filename = f"bonafide_{idx+1:04d}.flac"
    audio.export(output_dir / new_filename, format="flac")
    print(f"✅ {mp3_file.name} → {new_filename}")
