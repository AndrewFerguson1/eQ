from flask import Flask, render_template_string, request, send_file, jsonify
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import os
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# HTML template embedded in Python
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🎵 Audio EQ</title>
    <style>
        body { font-family: Arial; background: #1a1a1a; color: white; padding: 20px; }
        .container { max-width: 600px; margin: 0 auto; background: #333; padding: 30px; border-radius: 15px; }
        h1 { text-align: center; color: #ff6b6b; }
        .control { margin: 20px 0; padding: 15px; background: #444; border-radius: 10px; }
        .slider { width: 100%; height: 10px; background: #666; border-radius: 5px; }
        .btn { background: #ff6b6b; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
        .btn:disabled { opacity: 0.5; }
        #status { text-align: center; margin: 20px 0; padding: 10px; background: #555; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 3-Band Audio EQ</h1>
        
        <input type="file" id="audioFile" accept="audio/*" style="margin: 20px 0;">
        <div id="fileInfo"></div>
        
        <div class="control">
            <label>Low (20-250 Hz): <span id="lowValue">0.0 dB</span></label><br>
            <input type="range" id="lowGain" class="slider" min="-12" max="12" step="0.1" value="0">
        </div>
        
        <div class="control">
            <label>Mid (250-4000 Hz): <span id="midValue">0.0 dB</span></label><br>
            <input type="range" id="midGain" class="slider" min="-12" max="12" step="0.1" value="0">
        </div>
        
        <div class="control">
            <label>High (4000-20000 Hz): <span id="highValue">0.0 dB</span></label><br>
            <input type="range" id="highGain" class="slider" min="-12" max="12" step="0.1" value="0">
        </div>
        
        <div style="text-align: center;">
            <button id="resetBtn" class="btn">Reset</button>
            <button id="processBtn" class="btn" disabled>Apply EQ</button>
            <button id="downloadBtn" class="btn" disabled>Download</button>
        </div>
        
        <div id="status">Upload an audio file to start</div>
    </div>

    <script>
        let currentFilename = null;
        let processedFilename = null;
        
        // Update slider displays
        ['lowGain', 'midGain', 'highGain'].forEach((id, i) => {
            const valueIds = ['lowValue', 'midValue', 'highValue'];
            document.getElementById(id).oninput = (e) => {
                document.getElementById(valueIds[i]).textContent = parseFloat(e.target.value).toFixed(1) + ' dB';
            };
        });
        
        // File upload
        document.getElementById('audioFile').onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('audio_file', file);
            document.getElementById('status').textContent = 'Uploading...';
            
            try {
                const response = await fetch('/upload', { method: 'POST', body: formData });
                const result = await response.json();
                
                if (result.success) {
                    currentFilename = result.filename;
                    document.getElementById('fileInfo').innerHTML = `<strong>Loaded:</strong> ${result.filename}`;
                    document.getElementById('processBtn').disabled = false;
                    document.getElementById('status').textContent = 'File loaded! Adjust EQ and click Apply.';
                } else {
                    document.getElementById('status').textContent = 'Error: ' + result.error;
                }
            } catch (error) {
                document.getElementById('status').textContent = 'Upload failed: ' + error.message;
            }
        };
        
        // Process audio
        document.getElementById('processBtn').onclick = async () => {
            document.getElementById('status').textContent = 'Processing...';
            
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        filename: currentFilename,
                        low_gain: document.getElementById('lowGain').value,
                        mid_gain: document.getElementById('midGain').value,
                        high_gain: document.getElementById('highGain').value
                    })
                });
                
                const result = await response.json();
                if (result.success) {
                    processedFilename = result.processed_filename;
                    document.getElementById('downloadBtn').disabled = false;
                    document.getElementById('status').textContent = '✅ EQ applied! Click Download.';
                } else {
                    document.getElementById('status').textContent = 'Error: ' + result.error;
                }
            } catch (error) {
                document.getElementById('status').textContent = 'Processing failed: ' + error.message;
            }
        };
        
        // Download
        document.getElementById('downloadBtn').onclick = () => {
            if (processedFilename) window.location.href = `/download/${processedFilename}`;
        };
        
        // Reset
        document.getElementById('resetBtn').onclick = () => {
            ['lowGain', 'midGain', 'highGain'].forEach(id => document.getElementById(id).value = 0);
            ['lowValue', 'midValue', 'highValue'].forEach(id => document.getElementById(id).textContent = '0.0 dB');
        };
    </script>
</body>
</html>
'''

temp_dir = tempfile.mkdtemp()

def apply_eq(audio, sr, low_gain, mid_gain, high_gain):
    nyquist = sr / 2
    
    # Design filters
    b_low, a_low = signal.butter(4, min(250, nyquist-1) / nyquist, btype='low')
    b_mid, a_mid = signal.butter(4, [max(250, 1)/nyquist, min(4000, nyquist-1)/nyquist], btype='band')
    b_high, a_high = signal.butter(4, max(4000, 1) / nyquist, btype='high')
    
    # Apply filters and gains
    low_band = signal.filtfilt(b_low, a_low, audio) * (10 ** (low_gain / 20))
    mid_band = signal.filtfilt(b_mid, a_mid, audio) * (10 ** (mid_gain / 20))
    high_band = signal.filtfilt(b_high, a_high, audio) * (10 ** (high_gain / 20))
    
    # Combine and normalize
    processed = low_band + mid_band + high_band
    max_val = np.max(np.abs(processed))
    if max_val > 1.0:
        processed = processed / max_val
    
    return processed

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['audio_file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(temp_dir, filename)
        file.save(filepath)
        
        # Test load
        librosa.load(filepath, sr=None)
        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.json
        filepath = os.path.join(temp_dir, data['filename'])
        audio, sr = librosa.load(filepath, sr=None)
        
        processed = apply_eq(audio, sr, float(data['low_gain']), 
                           float(data['mid_gain']), float(data['high_gain']))
        
        output_filename = f"eq_{data['filename'].split('.')[0]}.wav"
        output_path = os.path.join(temp_dir, output_filename)
        sf.write(output_path, processed, sr)
        
        return jsonify({'success': True, 'processed_filename': output_filename})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(temp_dir, filename), as_attachment=True)

if __name__ == '__main__':
    print("🎵 Starting Audio EQ Server...")
    print("📁 Install packages first: pip install flask librosa soundfile scipy numpy")
    app.run(host='0.0.0.0', port=5000, debug=True)
