import os
import glob
import uuid
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pytorch_lightning as pl
import torch
from patchcore_test_alldata import AnomalyModel  

app = Flask(__name__)
CORS(app)

loaded_model = None
model_checkpoint_path = None
OUTPUT_DIR = os.path.abspath("./output")
MODEL_ROOT = "C:/Users/user/Desktop/dataset/patchcore_result"

def find_latest_checkpoint(model_path):
    ckpt_dir = os.path.join(model_path, "lightning_logs", "version_0", "checkpoints")
    ckpt_list = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpt_list:
        raise FileNotFoundError(f"找不到 checkpoint 於: {ckpt_dir}")
    return max(ckpt_list, key=os.path.getmtime)

def load_model_from_path(model_path, output_dir=OUTPUT_DIR):
    checkpoint_path = find_latest_checkpoint(model_path)
    model = AnomalyModel.load_from_checkpoint(
        checkpoint_path,
        dataset_path=None,
        model_path=model_path,
        output_path=output_dir,
        input_size=512,
        n_neighbors=9,
        save_anomaly_map=True,
        progress_callback=None,
    )
    model.eval()
    return model, checkpoint_path

@app.route('/load_model', methods=['POST'])
def load_model_api():
    global loaded_model, model_checkpoint_path
    data = request.get_json()
    model_path = data.get('model_path')

    if not model_path or not os.path.exists(model_path):
        return jsonify({'error': '請提供有效模型路徑'}), 400

    try:
        loaded_model, model_checkpoint_path = load_model_from_path(model_path)
        return jsonify({'message': '模型載入成功', 'checkpoint': model_checkpoint_path})
    except Exception as e:
        return jsonify({'error': f'載入模型失敗: {str(e)}'}), 500

@app.route("/models", methods=["GET"])
def list_models():
    if not os.path.exists(MODEL_ROOT):
        return jsonify({"error": "模型根目錄不存在"}), 500

    models = []
    for dir_name in os.listdir(MODEL_ROOT):
        full_path = os.path.join(MODEL_ROOT, dir_name)
        if os.path.isdir(full_path):
            models.append({
                "name": dir_name,
                "path": full_path.replace("\\", "/")
            })

    return jsonify(models)

@app.route("/run_test", methods=["POST"])
def run_test():
    global loaded_model

    if loaded_model is None:
        return jsonify({"error": "模型尚未載入"}), 400

    if 'image' not in request.files:
        return jsonify({"error": "請上傳影像"}), 400

    try:
        image_file = request.files['image']
        image = Image.open(image_file).convert("RGB")

        # 產生唯一ID，確保路徑不覆蓋
        unique_id = str(uuid.uuid4())
        print(unique_id)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        anomaly_maps_dir = os.path.join(OUTPUT_DIR, "anomaly_maps")
        os.makedirs(anomaly_maps_dir, exist_ok=True)

        input_filename = f"{unique_id}.jpg"
        heatmap_filename = f"amap_on_img_{unique_id}.jpg"
        print(heatmap_filename)

        input_path = os.path.join(OUTPUT_DIR, input_filename)
        heatmap_path = os.path.join(anomaly_maps_dir, heatmap_filename)
        print(heatmap_path)

        image.save(input_path)

        lo_ratio = float(request.form.get('lo_ratio', 0.6))
        threshold = float(request.form.get('threshold', 1.5))

        loaded_model.single_test_image_path = input_path
        loaded_model.manual_threshold = threshold
        loaded_model.manual_lo_ratio = lo_ratio

        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False
        )
        trainer.test(loaded_model)
        print(f"anomaly_maps 目錄檔案：{os.listdir(anomaly_maps_dir)}")

        if not os.path.exists(heatmap_path):
            return jsonify({"error": f"無法產生熱圖: {heatmap_path}"}), 500

        score = getattr(loaded_model, "latest_score", None)
        result_label = getattr(loaded_model, "latest_label", None)
        heatmap_max = getattr(loaded_model, "latest_heatmap_max", None)

        if score is None or result_label is None:
            return jsonify({"error": "模型尚未產生結果"}), 500

        heatmap_url = f"http://127.0.0.1:5000/heatmap/amap_on_img_{unique_id}"

        return jsonify({
            "message": "辨識完成",
            "score": round(score, 4),
            "heatmap_max": round(heatmap_max, 4) if heatmap_max else None,
            "result": "異常" if result_label == 1 else "正常",
            "threshold": threshold,
            "heatmap_url": heatmap_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/heatmap/<img_id>")
def get_heatmap(img_id):
    heatmap_path = os.path.join(OUTPUT_DIR, "anomaly_maps", f"{img_id}.jpg")
    print(f"尋找 heatmap 檔案: {heatmap_path}")
    if os.path.exists(heatmap_path):
        return send_file(heatmap_path, mimetype='image/jpeg')
    else:
        return "Heatmap not found", 404

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
