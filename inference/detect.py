import torch
import librosa
import numpy as np
import time
from pathlib import Path

from models.model import Network
from utils.utils import Genotype

class AudioSpoofDetector:
    """模型单例"""
    # def __init__(self, model_path, layers=4, init_channels=16, threshold=0.880696):
    # def __init__(self, model_path, layers=4, init_channels=16, threshold=-0.107686):
    def __init__(self, model_path, layers=4, init_channels=16, threshold=-0.031941):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sr = 16000
        self.threshold = threshold
        self.layers = layers
        self.init_channels = init_channels

        print(f"📦 Loading model: {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()
        print("✅ Model loaded successfully")

    def _load_model(self, model_path):
        arch = "Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 3), ('avg_pool_3x3', 2), ('avg_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 3), ('dil_conv_3x3', 2), ('avg_pool_3x3', 4)], reduce_concat=range(2, 6))"
        genotype = eval(arch)

        class Args:
            def __init__(self, layers, init_channels):
                self.nfft = 1024
                self.hop = 4
                self.nfilter = 70
                self.num_ceps = 20
                self.is_log = True
                self.is_cmvn = False
                self.is_mask = False
                self.sr = 16000
                self.drop_path_prob = 0.0
                self.layers = layers
                self.init_channels = init_channels

        args = Args(self.layers, self.init_channels)
        model = Network(self.init_channels, self.layers, args, 2, genotype, 'LFCC')
        model.drop_path_prob = 0.0

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        return model.to(self.device)

    def detect(self, audio_path):
        try:
            start = time.time()
            audio, sr = librosa.load(audio_path, sr=self.sr)
            fix_len = sr * 4
            if len(audio) < fix_len:
                audio = np.tile(audio, fix_len // len(audio) + 1)
            audio = audio[:fix_len]
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(audio_tensor, is_mask=False)
                if hasattr(self.model, 'forward_classifier'):
                    output = self.model.forward_classifier(output)

                spoof_score = output[0][0].item()
                bonafide_score = output[0][1].item()
                prediction = "REAL" if bonafide_score > self.threshold else "FAKE"
                return {
                    "prediction": prediction,
                    "bonafide_score": bonafide_score,
                    "spoof_score": spoof_score,
                    "time": time.time() - start
                }
        except Exception as e:
            return {"error": str(e)}

# ============ CLI 入口 ============
def main():
    model_path = "finetune_models/optuna_best.pth"
    detector = AudioSpoofDetector(model_path)

    print("\n🎧 Audio Spoof Detection CLI")
    print("Type an audio file path to detect, or type 'exit' to quit.\n")

    while True:
        audio_path = input("🔍 Audio file: ").strip()
        if audio_path.lower() == "exit":
            print("👋 Bye!")
            break
        if not Path(audio_path).exists():
            print("❌ File not found.")
            continue
        result = detector.detect(audio_path)
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            icon = "✅" if result["prediction"] == "REAL" else "❌"
            print(f"{icon} Result: {result['prediction']}")

if __name__ == "__main__":
    main()
