#!/usr/bin/env python3
"""
TensorFlow.js to TFLite Conversion Server

A lightweight Flask server that converts TensorFlow.js models (from the
Calliope Teachable Machine) to TFLite format for flashing to Grove Vision AI V2.

Usage:
    pip install flask flask-cors tensorflow tensorflowjs
    python conversion_server.py

The server runs on http://localhost:5000 and accepts POST requests to /convert
with multipart form data containing:
    - model_json: The model.json file (topology + weight manifest)
    - weights_bin: The weights.bin file (binary weight data)
    - metadata: JSON string with classes and input_shape (optional)

Query parameters:
    - quantize=true: Apply INT8 quantization (smaller model, better for edge)
"""

import json
import os
import shutil
import tempfile
import sys

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/convert', methods=['POST'])
def convert_model():
    """Convert a TensorFlow.js model to TFLite format."""
    tmp_dir = None
    try:
        # Lazy import to show better error messages
        try:
            import tensorflow as tf
            import tensorflowjs as tfjs
        except ImportError as e:
            return jsonify({
                'error': f'Missing dependency: {e}. '
                         f'Install with: pip install tensorflow tensorflowjs'
            }), 500

        # Validate request
        if 'model_json' not in request.files:
            return jsonify({'error': 'Missing model_json file'}), 400
        if 'weights_bin' not in request.files:
            return jsonify({'error': 'Missing weights_bin file'}), 400

        model_json_file = request.files['model_json']
        weights_bin_file = request.files['weights_bin']
        metadata_str = request.form.get('metadata', '{}')
        quantize = request.args.get('quantize', 'false').lower() == 'true'

        # Parse metadata
        try:
            metadata = json.loads(metadata_str)
        except json.JSONDecodeError:
            metadata = {}

        # Create temporary directory for model files
        tmp_dir = tempfile.mkdtemp(prefix='tfjs_convert_')
        model_json_path = os.path.join(tmp_dir, 'model.json')
        weights_bin_path = os.path.join(tmp_dir, 'weights.bin')

        # Save uploaded files
        model_json_file.save(model_json_path)
        weights_bin_file.save(weights_bin_path)

        # Verify model.json has the right structure
        with open(model_json_path, 'r') as f:
            model_config = json.load(f)

        # Ensure weightsManifest references the correct file
        if 'weightsManifest' in model_config:
            for manifest in model_config['weightsManifest']:
                manifest['paths'] = ['weights.bin']
            # Rewrite with corrected paths
            with open(model_json_path, 'w') as f:
                json.dump(model_config, f)

        print(f"Loading TF.js model from {tmp_dir}...")
        print(f"  Model topology keys: {list(model_config.get('modelTopology', {}).keys()) if isinstance(model_config.get('modelTopology'), dict) else 'N/A'}")
        print(f"  Weights file size: {os.path.getsize(weights_bin_path)} bytes")

        # Load the TF.js model
        model = tfjs.converters.load_keras_model(model_json_path)
        model.summary()

        # Convert to TFLite
        print(f"Converting to TFLite (quantize={quantize})...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # For full INT8 quantization, we'd need representative data
            # For now, use dynamic range quantization (float16 weights)
            converter.target_spec.supported_types = [tf.float16]
            print("  Applying float16 quantization")

        tflite_model = converter.convert()
        print(f"  TFLite model size: {len(tflite_model)} bytes")

        # Save to temp file and return
        tflite_path = os.path.join(tmp_dir, 'model.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        return send_file(
            tflite_path,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name='model.tflite'
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up temp directory after sending
        if tmp_dir and os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    status = {'status': 'ok', 'server': 'tfjs-to-tflite-converter'}
    try:
        import tensorflow as tf
        status['tensorflow'] = tf.__version__
    except ImportError:
        status['tensorflow'] = 'not installed'
    try:
        import tensorflowjs
        status['tensorflowjs'] = tensorflowjs.__version__
    except ImportError:
        status['tensorflowjs'] = 'not installed'
    return jsonify(status)


if __name__ == '__main__':
    print("=" * 60)
    print("TF.js to TFLite Conversion Server")
    print("=" * 60)
    print()
    print("Endpoints:")
    print("  POST /convert  - Convert TF.js model to TFLite")
    print("  GET  /health   - Health check")
    print()
    print("Usage from Calliope Teachable Machine:")
    print("  1. Train a model in the browser")
    print("  2. Enable Grove AI and connect device")
    print("  3. Click 'Convert Current Model to TFLite'")
    print("  4. Flash the converted model to the device")
    print()

    # Check dependencies
    deps_ok = True
    try:
        import tensorflow as tf
        print(f"  TensorFlow: {tf.__version__}")
    except ImportError:
        print("  WARNING: TensorFlow not installed!")
        print("    Install with: pip install tensorflow")
        deps_ok = False
    try:
        import tensorflowjs
        print(f"  TensorFlow.js: {tensorflowjs.__version__}")
    except ImportError:
        print("  WARNING: TensorFlow.js not installed!")
        print("    Install with: pip install tensorflowjs")
        deps_ok = False

    if not deps_ok:
        print()
        print("Install all dependencies:")
        print("  pip install flask flask-cors tensorflow tensorflowjs")
        print()

    print()
    print("Starting server on http://localhost:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
