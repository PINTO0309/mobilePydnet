import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ONNX モデルのパス
onnx_model_path = "pydnet_384x640.onnx"

# ONNX Runtime セッションを作成
session = ort.InferenceSession(
    path_or_bytes=onnx_model_path,
    providers=["CPUExecutionProvider"],
)

# モデルの入力情報を取得
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # [1,3,384,640]
input_dtype = session.get_inputs()[0].type   # float32

print(f"Model Loaded: {onnx_model_path}")
print(f"Input Name: {input_name}")
print(f"Input Shape: {input_shape}")
print(f"Input Type: {input_dtype}")

def preprocess_image(image_path):
    """
    画像を読み込み、前処理を行い、ONNX モデルに適した形状に変換する。
    入力: 画像ファイルのパス
    出力: float32のNumPy配列 (1,3,384,640)
    """
    # 画像を読み込む
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # OpenCV は BGR で読み込むため、RGB に変換
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 画像のリサイズ (384x640)
    img_resized = cv2.resize(img_rgb, (640, 384))

    # チャンネルの順序を (H, W, C) -> (C, H, W) に変更
    img_transposed = img_resized.transpose(2, 0, 1)

    # 正規化 (0-255 を 0-1 にスケール変換)
    img_normalized = img_transposed.astype(np.float32) / 255.0

    # バッチ次元を追加 (1, C, H, W)
    img_tensor = np.expand_dims(img_normalized, axis=0)

    return img_tensor, img_resized  # 前処理後の入力と元画像を返す

def postprocess_depth_map(depth_map, original_image):
    """
    デプス推定結果をヒートマップ化し、元画像と重畳表示する。
    入力:
        - depth_map: (1, 1, 384, 640) の NumPy 配列 (モデルの出力)
        - original_image: (384, 640, 3) の元画像
    出力:
        - ヒートマップ重畳画像
    """
    # デプスマップの形状変更 (1, 1, H, W) → (H, W)
    depth_map = depth_map.squeeze()

    # デプスマップの正規化 (0-1)
    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)

    # ヒートマップ化 (OpenCV の COLORMAP_JET を適用)
    depth_colormap = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # 画像のチャンネル順序を RGB に変更
    depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)

    # ヒートマップと元画像を合成 (αブレンディング)
    overlay = cv2.addWeighted(original_image, 0.6, depth_colormap, 0.4, 0)

    return overlay

def run_inference(image_path):
    """
    指定された画像に対して ONNX モデルの推論を実行し、デプス推定結果を可視化する。
    """
    # 画像の前処理
    input_tensor, original_image = preprocess_image(image_path)

    # 推論実行
    outputs = session.run(None, {input_name: input_tensor})

    # モデルの出力 (通常は [1, 1, 384, 640])
    depth_map = outputs[0]  # モデルの出力が 1 つの場合

    # デプスマップを可視化
    depth_overlay = postprocess_depth_map(depth_map, original_image)

    # 結果を表示
    plt.figure(figsize=(10, 12))
    plt.subplot(2, 1, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 1, 2)
    plt.imshow(depth_overlay)
    plt.title("Depth Map Overlay")
    plt.axis("off")

    plt.show()

    return depth_overlay

if __name__ == "__main__":
    # 画像のパスを指定
    image_path = "000000012069.jpg"

    # 推論実行
    output_overlay = run_inference(image_path)

    # 結果を保存
    cv2.imwrite("depth_overlay.png", cv2.cvtColor(output_overlay, cv2.COLOR_RGB2BGR))
