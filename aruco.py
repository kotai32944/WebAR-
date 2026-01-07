import cv2
from cv2 import aruco
import numpy as np
import os

# 設定
marker_size = 4          # 4x4 bits
marker_img_size = 300    # PNG画像のサイズ
base_dir = os.path.dirname(os.path.abspath(__file__))

# カスタムマーカービット定義
bits_list = [
    np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.uint8),
    np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 0]], dtype=np.uint8),
    np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 1, 0]], dtype=np.uint8),
]

# Dictionary 作成
byte_list_rows = [aruco.Dictionary_getByteListFromBits(bits) for bits in bits_list]
bytes_list = np.concatenate(byte_list_rows, axis=0)
custom_dict = aruco.Dictionary_create(len(bits_list), marker_size)
custom_dict.bytesList = bytes_list

print(f"マーカー画像の生成を開始します... 保存先: {base_dir}")

for marker_id, bits in enumerate(bits_list):
    # マーカー画像の生成と保存
    img = aruco.drawMarker(custom_dict, marker_id, marker_img_size)
    png_name = f"marker_custom_{marker_id}.png"
    cv2.imwrite(os.path.join(base_dir, png_name), img)
    print(f"保存完了: {png_name}")

print("\n生成されたPNGファイルを公式ジェネレーターにアップロードしてください。")