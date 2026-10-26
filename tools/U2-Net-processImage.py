"""批量分割叶片图像，按非透明区域裁剪并合成为黑底 RGB 图像。"""

import os

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove
from tqdm import tqdm


def process_directory(input_root, output_root):
    """处理输入目录下各类别子目录；分割失败的图像保存到隔离目录。"""
    if input_root == "**" or output_root == "**":
        raise ValueError("请先将 INPUT_ROOT 和 OUTPUT_ROOT 中的 '**' 替换为本地路径。")
    if not os.path.isdir(input_root):
        raise NotADirectoryError(input_root)

    session = new_session("u2net")

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    classes = sorted(os.listdir(input_root))
    for cls_name in classes:
        cls_dir = os.path.join(input_root, cls_name)
        out_cls_dir = os.path.join(output_root, cls_name)
        quarantine_dir = out_cls_dir + "_failed_quarantine"

        if not os.path.isdir(cls_dir):
            continue

        if not os.path.exists(out_cls_dir):
            os.makedirs(out_cls_dir)
        if not os.path.exists(quarantine_dir):
            os.makedirs(quarantine_dir)

        images = [
            f for f in sorted(os.listdir(cls_dir))
            if f.lower().endswith((".jpg", ".png"))
        ]
        print(f"正在处理类别 [{cls_name}] 的图像。")

        for img_name in tqdm(images):
            in_path = os.path.join(cls_dir, img_name)
            out_path = os.path.join(out_cls_dir, img_name)
            quarantine_path = os.path.join(quarantine_dir, img_name)

            try:
                input_image = Image.open(in_path).convert("RGB")
                output_image = remove(input_image, session=session)

                img_array = np.array(output_image)
                alpha_channel = img_array[:, :, 3]
                coords = cv2.findNonZero(alpha_channel)

                if coords is not None:
                    # 保留原脚本的非透明像素数量判定规则。
                    if len(coords) > 1000:
                        x, y, w, h = cv2.boundingRect(coords)
                        cropped_rgba = img_array[y:y + h, x:x + w]

                        black_bg = Image.new("RGB", (w, h), (0, 0, 0))
                        leaf_img = Image.fromarray(cropped_rgba)
                        black_bg.paste(leaf_img, mask=leaf_img.split()[3])
                        black_bg.save(out_path)
                    else:
                        print(f"\n{img_name} 的非透明区域过小，已保存到隔离目录。")
                        input_image.save(quarantine_path)
                else:
                    print(f"\n{img_name} 未识别到非透明区域，已保存到隔离目录。")
                    input_image.save(quarantine_path)

            except Exception as exc:
                print(f"\n处理 {img_name} 时出错：{exc}；尝试保存到隔离目录。")
                try:
                    Image.open(in_path).convert("RGB").save(quarantine_path)
                except Exception:
                    pass


if __name__ == "__main__":
    INPUT_ROOT = "**"
    OUTPUT_ROOT = "**"

    process_directory(INPUT_ROOT, OUTPUT_ROOT)
    print("\n图像处理结束，请检查输出目录及隔离目录。")
