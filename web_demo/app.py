from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
import torch
import librosa
import numpy as np
import time
import os
import uuid
from pathlib import Path
from werkzeug.utils import secure_filename

from models.model import Network
from utils.utils import Genotype

class AudioSpoofDetector:
    """模型单例"""
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_path, layers=4, init_channels=16, threshold=0.880696):
        if hasattr(self, '_initialized'):
            return
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sr = 16000
        self.threshold = threshold
        self.layers = layers
        self.init_channels = init_channels

        print(f"📦 Loading model: {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()
        print("✅ Model loaded successfully")
        self._initialized = True

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
                confidence = abs(bonafide_score - self.threshold)
                
                return {
                    "prediction": prediction,
                    "bonafide_score": bonafide_score,
                    "spoof_score": spoof_score,
                    "confidence": confidence,
                    "processing_time": time.time() - start,
                    "success": True
                }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }

# Flask应用配置
app = Flask(__name__)
print("Flask will search templates in:", app.template_folder)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 允许的文件格式
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'aac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 全局检测器
detector = None

def init_detector():
    """初始化检测器"""
    global detector
    model_path = "finetune_models/optuna_best.pth"
    try:
        detector = AudioSpoofDetector(model_path)
        return True
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和检测"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported. Please upload audio files (wav, mp3, flac, etc.)'}), 400
    
    try:
        # 保存文件
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # 检测
        if detector is None:
            return jsonify({'error': 'Model not initialized'}), 500
            
        result = detector.detect(filepath)
        
        # 清理临时文件
        try:
            os.remove(filepath)
        except:
            pass
        
        if result['success']:
            return jsonify({
                'success': True,
                'prediction': result['prediction'],
                'bonafide_score': round(result['bonafide_score'], 4),
                'spoof_score': round(result['spoof_score'], 4),
                'confidence': round(result['confidence'], 4),
                'processing_time': round(result['processing_time'], 3),
                'filename': filename
            })
        else:
            return jsonify({'error': result['error']}), 500
            
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/status')
def api_status():
    """API状态检查"""
    return jsonify({
        'status': 'ready' if detector is not None else 'not_ready',
        'device': str(detector.device) if detector else 'unknown',
        'model_info': {
            'layers': detector.layers if detector else None,
            'channels': detector.init_channels if detector else None,
            'threshold': detector.threshold if detector else None
        }
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

if __name__ == '__main__':
    print("🚀 Starting Audio Spoof Detection Web App")
    print("=" * 50)
    
    # 初始化模型
    if init_detector():
        print("✅ Model initialized successfully")
        print("🌐 Starting Flask server...")
        print("📱 Open http://localhost:5000 in your browser")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to start: Model initialization failed")