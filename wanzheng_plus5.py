import cv2
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib.font_manager as fm
import logging
import glob
import pandas as pd
import traceback
import random
#hu矩只计算外轮廓相似度
# 配置日志系统
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ======================== 双语言标签函数（仅保留英文） ========================
def get_bilingual_labels():
    """返回英文标签映射"""
    return {
        'Rubbing Contours': 'Rubbing Contours',
        'Copy Contours': 'Copy Contours',
        'Completeness Score': 'Completeness Score',
        'Hu Moment Similarity': 'Hu Moment Similarity',
        'Image Issue Diagnosis Report': 'Image Issue Diagnosis Report',
        'Quality Score': 'Quality Score',
        'Initial Contours': 'Initial Contours',
        'Filtered Contours': 'Filtered Contours'
    }

# ======================== 新增工具函数 ========================
def get_top_n_contours(contours, n):
    """按面积降序排序，返回前n个轮廓（适配动态匹配数量）"""
    if len(contours) <= n:
        return contours
    # 按轮廓面积降序排序
    sorted_contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    return sorted_contours[:n]

def calculate_greedy_matching_similarity(tuopian_contours, muben_contours):
    """贪心算法实现多轮廓最优匹配，避免顺序偏差"""
    if len(tuopian_contours) == 0 or len(muben_contours) == 0:
        logging.warning("Rubbing or Copy has no valid contours, similarity is 0")
        return 0.0, []
    
    # 计算所有轮廓对的相似度矩阵
    sim_matrix = []
    for t_cnt in tuopian_contours:
        t_moments = cv2.moments(t_cnt)
        if t_moments['m00'] == 0:
            sim_matrix.append([0.0]*len(muben_contours))
            continue
        t_hu = cv2.HuMoments(t_moments).flatten()
        row_sims = []
        for m_cnt in muben_contours:
            m_moments = cv2.moments(m_cnt)
            if m_moments['m00'] == 0:
                row_sims.append(0.0)
                continue
            m_hu = cv2.HuMoments(m_moments).flatten()
            dist = distance.euclidean(t_hu, m_hu)
            row_sims.append(1.0/(1.0+dist))  # 距离转相似度
        sim_matrix.append(row_sims)
    
    # 贪心匹配：每次选相似度最高的未匹配对
    used_t = set()  # 已匹配的拓片轮廓索引
    used_m = set()  # 已匹配的摹本轮廓索引
    total_sim = 0.0
    match_pairs = []  # 记录匹配对（拓片索引, 摹本索引, 相似度）
    
    # 按相似度降序排列所有可能的匹配对
    all_matches = []
    for t_idx in range(len(tuopian_contours)):
        for m_idx in range(len(muben_contours)):
            all_matches.append( (sim_matrix[t_idx][m_idx], t_idx, m_idx) )
    all_matches.sort(reverse=True, key=lambda x: x[0])
    
    # 选择最优匹配（无重复）
    for sim, t_idx, m_idx in all_matches:
        if t_idx not in used_t and m_idx not in used_m:
            total_sim += sim
            used_t.add(t_idx)
            used_m.add(m_idx)
            match_pairs.append( (t_idx, m_idx, sim) )
            logging.info(f"Best Match: Rubbing Contour {t_idx+1} ↔ Copy Contour {m_idx+1} (Similarity: {sim:.3f})")
    
    # 计算平均相似度
    avg_sim = total_sim / len(match_pairs) if match_pairs else 0.0
    logging.info(f"Multi-contour greedy matching completed, average similarity: {avg_sim:.3f}")
    return avg_sim, match_pairs

# 新增：颜色生成与同图合并函数
def generate_unique_colors(n):
    """生成n种唯一明亮RGB颜色（用于标记匹配对）"""
    colors = []
    random.seed(42)  # 固定种子，确保颜色一致性
    for _ in range(n):
        # 避免暗色，保证区分度
        r = random.randint(60, 255)
        g = random.randint(60, 255)
        b = random.randint(60, 255)
        colors.append((r, g, b))
    return colors

def merge_contours_on_canvas(tuopian_binary, muben_binary, tuopian_contours, muben_contours, match_pairs):
    """将拓片和摹本轮廓按原始尺寸拼接在同一张画布上，用颜色标记匹配对"""
    # 获取两者原始尺寸
    t_h, t_w = tuopian_binary.shape
    m_h, m_w = muben_binary.shape
    
    # 计算画布尺寸：宽度=两者宽度之和（横向拼接），高度=两者最大高度
    canvas_w = t_w + m_w
    canvas_h = max(t_h, m_h)
    
    # 创建白色画布（RGB格式）
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    # 1. 绘制拓片轮廓（左侧）
    # 未匹配轮廓用灰色（128,128,128）
    for cnt in tuopian_contours:
        cv2.drawContours(canvas, [cnt], -1, (128, 128, 128), 2)
    
    # 2. 绘制摹本轮廓（右侧，x坐标偏移拓片宽度）
    muben_offset = np.array([[t_w, 0]])  # 只偏移x轴，y轴对齐顶部
    for cnt in muben_contours:
        cnt_shifted = cnt + muben_offset  # 偏移后不改变原始尺寸
        cv2.drawContours(canvas, [cnt_shifted], -1, (128, 128, 128), 2)
    
    # 3. 用唯一颜色标记匹配对
    num_matches = len(match_pairs)
    if num_matches > 0:
        match_colors = generate_unique_colors(num_matches)
        for pair_idx, (t_idx, m_idx, sim) in enumerate(match_pairs):
            color = match_colors[pair_idx]
            
            # 标记匹配的拓片轮廓（左侧）
            t_cnt = tuopian_contours[t_idx]
            cv2.drawContours(canvas, [t_cnt], -1, color, 3)  # 加粗突出
            # 绘制拓片文字标注
            t_mom = cv2.moments(t_cnt)
            if t_mom['m00'] != 0:
                t_cx = int(t_mom['m10'] / t_mom['m00'])
                t_cy = int(t_mom['m01'] / t_mom['m00'])
                cv2.putText(
                    canvas, f"R{t_idx+1}", (t_cx, t_cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (color[0], color[1], color[2]), 1
                )
            
            # 标记匹配的摹本轮廓（右侧，带偏移）
            m_cnt = muben_contours[m_idx]
            m_cnt_shifted = m_cnt + muben_offset
            cv2.drawContours(canvas, [m_cnt_shifted], -1, color, 3)  # 同色加粗
            # 绘制摹本文字标注
            m_mom = cv2.moments(m_cnt)
            if m_mom['m00'] != 0:
                m_cx = int(m_mom['m10'] / m_mom['m00']) + t_w  # 叠加偏移量
                m_cy = int(m_mom['m01'] / m_mom['m00'])
                cv2.putText(
                    canvas, f"C{m_idx+1}", (m_cx, m_cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (color[0], color[1], color[2]), 1
                )
    
    # 添加分隔线和标题
    cv2.line(canvas, (t_w, 0), (t_w, canvas_h), (0, 0, 0), 2)  # 黑白分隔线
    cv2.putText(canvas, "Rubbing Contours (Left)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(canvas, "Copy Contours (Right)", (t_w + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return canvas

# ======================== 图像预处理函数（保持不变） ========================
def preprocess_image(image_path, output_dir=None, image_name="image", save_intermediate=True, is_muben=False):
    if is_muben:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read copy image: {image_path}")
        # 摹本二值化：白底黑字（文字0，背景255）
        _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        denoised_img = binary_img
        gray_img = img
        is_color = False
        logging.info(f"Copy preprocessing completed: White background with black text binarization (text 0, background 255)")
    else:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read rubbing image: {image_path}")
        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            is_color = True
        else:
            gray_img = img
            is_color = False
        denoised_img = cv2.medianBlur(gray_img, 3)
        _, binary_img = cv2.threshold(denoised_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logging.info(f"Rubbing preprocessing completed: Black background with white text binarization (text 255, background 0)")
    
    if save_intermediate and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if is_muben:
            cv2.imwrite(os.path.join(output_dir, f"{image_name}_Original_Grayscale.png"), img)
            cv2.imwrite(os.path.join(output_dir, f"{image_name}_Binarized.png"), binary_img)
        else:
            if is_color:
                cv2.imwrite(os.path.join(output_dir, f"{image_name}_Original_Color.png"), img)
            cv2.imwrite(os.path.join(output_dir, f"{image_name}_Grayscale.png"), gray_img)
            cv2.imwrite(os.path.join(output_dir, f"{image_name}_Denoised.png"), denoised_img)
            cv2.imwrite(os.path.join(output_dir, f"{image_name}_Binarized.png"), binary_img)
        create_preprocessing_flowchart(img, gray_img, denoised_img, binary_img, output_dir, image_name, is_muben)
    
    return {
        'original': img,
        'gray': gray_img,
        'denoised': denoised_img,
        'binary': binary_img,
        'is_color': is_color
    }

def create_preprocessing_flowchart(original, gray, denoised, binary, output_dir, image_name, is_muben):
    if is_muben:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original Grayscale (Copy)')
        axes[0].axis('off')
        axes[1].imshow(binary, cmap='gray')
        axes[1].set_title('Binarized (White background, Black text)')
        axes[1].axis('off')
        plt.suptitle(f'{image_name} - Copy Preprocessing Flow')
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        if len(original.shape) == 3:
            axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Original Color Image')
        else:
            axes[0, 0].imshow(original, cmap='gray')
            axes[0, 0].set_title('Original Grayscale Image')
        axes[0, 0].axis('off')
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('Grayscale')
        axes[0, 1].axis('off')
        axes[1, 0].imshow(denoised, cmap='gray')
        axes[1, 0].set_title('Median Filtered')
        axes[1, 0].axis('off')
        axes[1, 1].imshow(binary, cmap='gray')
        axes[1, 1].set_title('Binarized (Black background, White text)')
        axes[1, 1].axis('off')
        plt.suptitle(f'{image_name} - Rubbing Preprocessing Flow')
    plt.tight_layout()
    flowchart_path = os.path.join(output_dir, f"{image_name}_Preprocessing_Flowchart.png")
    plt.savefig(flowchart_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Preprocessing flowchart saved: {flowchart_path}")

# ======================== 核心处理函数（保持拓片/摹本各自提取逻辑不变） ========================
def extract_rubbing_contours(binary_image, min_area=30, 
                            save_dir=None, image_name="rubbing"):
    """提取拓片（黑底白字）的白色文字外轮廓"""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        initial_contour_img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(initial_contour_img, contours, -1, (0, 0, 255), 2)
        save_path = os.path.join(save_dir, f"{image_name}_Initial_Contours.png")
        cv2.imwrite(save_path, initial_contour_img)
        logging.info(f"Rubbing initial contours saved: {save_path}")
    
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    if save_dir:
        filtered_contour_img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(filtered_contour_img, filtered_contours, -1, (0, 0, 255), 2)
        save_path = os.path.join(save_dir, f"{image_name}_Filtered_Contours.png")
        cv2.imwrite(save_path, filtered_contour_img)
        logging.info(f"Rubbing filtered contours saved: {save_path}")
    
    return filtered_contours

def extract_copy_contours(binary_image, min_area=30, 
                          save_dir=None, image_name="copy"):
    """提取摹本（白底黑字）的黑色文字外轮廓"""
    # 生成黑色文字的掩码（文字255，背景0）
    text_mask = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY_INV)[1]
    
    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # 在原始摹本二值图上绘制轮廓（白底黑字）
        initial_contour_img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(initial_contour_img, contours, -1, (0, 255, 0), 2)
        save_path = os.path.join(save_dir, f"{image_name}_Initial_Contours.png")
        cv2.imwrite(save_path, initial_contour_img)
        logging.info(f"Copy initial contours saved: {save_path}")
    
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    if save_dir:
        filtered_contour_img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(filtered_contour_img, filtered_contours, -1, (0, 255, 0), 2)
        save_path = os.path.join(save_dir, f"{image_name}_Filtered_Contours.png")
        cv2.imwrite(save_path, filtered_contour_img)
        logging.info(f"Copy filtered contours saved: {save_path}")
    
    return filtered_contours

def detect_image_type(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    height, width = img.shape
    border_pixels = []
    border_pixels.extend(img[0, :])
    border_pixels.extend(img[-1, :])
    border_pixels.extend(img[:, 0])
    border_pixels.extend(img[:, -1])
    avg_border_brightness = np.mean(border_pixels)
    return 'white_bg_black_text' if avg_border_brightness > 127 else 'black_bg_white_text'

# ======================== 相似度计算函数（保持不变） ========================
def calculate_hu_similarity(contours1, contours2):
    if len(contours1) == 0 or len(contours2) == 0:
        return 0
    moments1 = [cv2.moments(cnt) for cnt in contours1]
    moments2 = [cv2.moments(cnt) for cnt in contours2]
    hu_moments1 = [cv2.HuMoments(m) for m in moments1 if m['m00'] > 0]
    hu_moments2 = [cv2.HuMoments(m) for m in moments2 if m['m00'] > 0]
    if not hu_moments1 or not hu_moments2:
        return 0
    min_distances = []
    for hu1 in hu_moments1:
        distances = [distance.euclidean(hu1.flatten(), hu2.flatten()) for hu2 in hu_moments2]
        min_distances.append(min(distances))
    avg_min_distance = np.mean(min_distances) if min_distances else 1.0
    similarity = 1.0 / (1.0 + avg_min_distance)
    logging.info(f"Hu moment similarity calculation completed: {similarity:.3f}")
    return similarity

# ======================== 高级噪声过滤函数（保持不变） ========================
def advanced_noise_filtering(contours, image_shape, 
                            min_area=10, max_area=500,
                            min_aspect_ratio=0.1, max_aspect_ratio=0.9,
                            circularity_threshold=0.3):
    filtered_contours = []
    image_area = image_shape[0] * image_shape[1]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = min(w, h) / max(w, h)
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < circularity_threshold:
                continue
        filtered_contours.append(cnt)
    logging.info(f"Advanced noise filtering: Original count={len(contours)}, Filtered count={len(filtered_contours)}")
    return filtered_contours

# ======================== 诊断与评估函数（保持不变） ========================
def ensure_directory_exists(path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Directory created: {directory}")

def diagnose_image_issues(rubbing_path, copy_path, output_dir):
    ensure_directory_exists(os.path.join(output_dir, "dummy.txt"))
    rubbing = cv2.imread(rubbing_path, cv2.IMREAD_GRAYSCALE)
    copy = cv2.imread(copy_path, cv2.IMREAD_GRAYSCALE)
    if rubbing is None or copy is None:
        raise ValueError("Failed to read images")
    _, rubbing_binary = cv2.threshold(rubbing, 127, 255, cv2.THRESH_BINARY)
    _, copy_binary = cv2.threshold(copy, 127, 255, cv2.THRESH_BINARY_INV)  # 摹本二值图反转用于诊断
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    labels = get_bilingual_labels()
    axes[0, 0].imshow(rubbing, cmap='gray')
    axes[0, 0].set_title('Original Rubbing')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(copy, cmap='gray')
    axes[0, 1].set_title('Original Copy')
    axes[0, 1].axis('off')
    axes[0, 2].imshow(rubbing_binary, cmap='gray')
    axes[0, 2].set_title('Rubbing Binary')
    axes[0, 2].axis('off')
    axes[0, 3].imshow(copy_binary, cmap='gray')
    axes[0, 3].set_title('Copy Binary (Inverted)')
    axes[0, 3].axis('off')
    contours_rubbing, _ = cv2.findContours(rubbing_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_copy, _ = cv2.findContours(copy_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rubbing_diagnostic = cv2.cvtColor(rubbing_binary, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(rubbing_diagnostic, contours_rubbing, -1, (0, 0, 255), 2)
    axes[1, 0].imshow(rubbing_diagnostic)
    axes[1, 0].set_title(f'{labels["Rubbing Contours"]} ({len(contours_rubbing)} contours)')
    axes[1, 0].axis('off')
    copy_diagnostic = cv2.cvtColor(copy_binary, cv2.COLOR_GRAY2BGR)
    for i, cnt in enumerate(contours_copy):
        cv2.drawContours(copy_diagnostic, [cnt], -1, (0, 255, 0), 2)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # 摹本诊断图标注
            fig_temp, ax_temp = plt.subplots(figsize=(0.1, 0.1))
            ax_temp.text(cx, cy, f'Contour {i+1}', fontsize=6, color='green')
            ax_temp.axis('off')
            fig_temp.canvas.draw()
            img_rgba = np.frombuffer(fig_temp.canvas.buffer_rgba().tobytes(), dtype=np.uint8)
            img_rgba = img_rgba.reshape(fig_temp.canvas.get_width_height()[::-1] + (4,))
            img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
            img_bgr = cv2.resize(img_bgr, (copy_diagnostic.shape[1], copy_diagnostic.shape[0]))
            copy_diagnostic = cv2.addWeighted(copy_diagnostic, 1, img_bgr, 1, 0)
            plt.close(fig_temp)
    axes[1, 1].imshow(copy_diagnostic)
    axes[1, 1].set_title(f'{labels["Copy Contours"]} ({len(contours_copy)} contours)')
    axes[1, 1].axis('off')
    axes[1, 2].axis('off')
    info_text = (
        f"{labels['Image Issue Diagnosis Report']}\n"
        f"Rubbing Size: {rubbing.shape}\n"
        f"Copy Size: {copy.shape}\n"
        f"Rubbing Contour Count: {len(contours_rubbing)}\n"
        f"Copy Contour Count: {len(contours_copy)}\n"
        f"Rubbing Non-zero Pixels: {np.count_nonzero(rubbing_binary)}\n"
        f"Copy Non-zero Pixels: {np.count_nonzero(copy_binary)}\n"
        f"Recommended Parameters:\n"
        f"- Reduce min_area to 10\n"
        f"- Check copy image quality"
    )
    axes[1, 2].text(0.1, 0.5, info_text, fontsize=12, va='center')
    axes[1, 3].axis('off')
    problem_text = (
        "Identified Issues:\n"
        "1. Insufficient noise removal\n"
        "2. Incomplete contour extraction\n"
        "3. Abnormal copy contour display\n"
        "\nSolutions:\n"
        "• Use advanced noise filtering\n"
        "• Optimize contour extraction strategy\n"
        "• Verify copy image integrity"
    )
    axes[1, 3].text(0.1, 0.5, problem_text, fontsize=12, va='center', color='red')
    plt.tight_layout()
    output_path = os.path.join(output_dir, labels['Image Issue Diagnosis Report'] + '.png')
    ensure_directory_exists(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Diagnosis report saved: {output_path}")

# 单独保存拓片匹配后轮廓图像的函数
def save_matched_contours_image(binary_image, matched_contours, output_dir, image_name="Matched_Rubbing_Contours"):
    """单独保存仅含匹配后轮廓的图像"""
    img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img, matched_contours, -1, (0, 0, 255), 2)  # 红色绘制匹配轮廓
    save_path = os.path.join(output_dir, f"{image_name}.png")
    cv2.imwrite(save_path, img)
    logging.info(f"Matched contours image saved: {save_path}")

# ======================== 核心评估函数 ========================
def assess_completeness_robust(rubbing_path, copy_path, output_dir=None, 
                              min_area=200, save_intermediate=True):
    if output_dir:
        ensure_directory_exists(os.path.join(output_dir, "dummy.txt"))
        if save_intermediate:
            intermediate_dir = os.path.join(output_dir, "Intermediate_Results")
            os.makedirs(intermediate_dir, exist_ok=True)
            logging.info(f"Intermediate results directory created: {intermediate_dir}")
        else:
            intermediate_dir = None
    else:
        intermediate_dir = None
    
    # 1. 预处理拓片和摹本图像
    logging.info("Preprocessing rubbing image...")
    rubbing_preprocessed = preprocess_image(
        rubbing_path, 
        intermediate_dir if save_intermediate else None, 
        "rubbing", 
        save_intermediate=save_intermediate,
        is_muben=False
    )
    rubbing_binary = rubbing_preprocessed['binary']
    
    logging.info("Preprocessing copy image...")
    copy_preprocessed = preprocess_image(
        copy_path, 
        intermediate_dir if save_intermediate else None, 
        "copy", 
        save_intermediate=save_intermediate,
        is_muben=True
    )
    copy_binary = copy_preprocessed['binary']
    
    # 2. 提取初始轮廓
    logging.info("Extracting rubbing contours...")
    rubbing_contours = extract_rubbing_contours(
        rubbing_binary, 
        min_area=min_area,  
        save_dir=intermediate_dir if save_intermediate else None, 
        image_name="rubbing"
    )
    logging.info("Extracting copy contours...")
    copy_contours = extract_copy_contours(
        copy_binary, 
        min_area=min_area,  
        save_dir=intermediate_dir if save_intermediate else None, 
        image_name="copy"
    )
    logging.info(f"Rubbing initial contour count: {len(rubbing_contours)}")
    logging.info(f"Copy initial contour count: {len(copy_contours)}")
    
    # 3. 动态匹配轮廓数量 + 多轮廓最优匹配
    n = len(copy_contours)  # 以摹本轮廓数量为标准
    logging.info(f"Using copy contour count as standard (n={n}), extracting top {n} largest rubbing contours")
    rubbing_top_n_contours = get_top_n_contours(rubbing_contours, n)  # 拓片前n大轮廓
    logging.info(f"Rubbing top {n} contours count: {len(rubbing_top_n_contours)}")
    
    # 贪心算法最优匹配
    hu_similarity, match_pairs = calculate_greedy_matching_similarity(rubbing_top_n_contours, copy_contours)
    matched_rubbing_contours = rubbing_top_n_contours  # 匹配后轮廓即拓片前n大轮廓
    
    # 4. 降级处理
    if not matched_rubbing_contours:
        matched_rubbing_contours = rubbing_contours
        logging.warning("No matching rubbing contours found, using original rubbing contours for evaluation")
    else:
        logging.info(f"Matched rubbing contour count (top {n}): {len(matched_rubbing_contours)}")
    
    # 5. 单独保存匹配后轮廓图像
    if output_dir and save_intermediate:
        save_matched_contours_image(
            rubbing_binary, 
            matched_rubbing_contours,
            intermediate_dir if save_intermediate else output_dir
        )
    
    # 6. 计算相似度（fallback逻辑）
    try:
        if hu_similarity == 0 and match_pairs == []:
            hu_similarity = calculate_hu_similarity(matched_rubbing_contours, copy_contours)
    except Exception as e:
        logging.error(f"Hu moment similarity calculation failed: {e}")
        hu_similarity = 0
    
    # 7. 生成评估报告
    report = {
        'completeness_score': hu_similarity,
        'hu_similarity': hu_similarity,
        'contour_counts': {
            'rubbing': len(matched_rubbing_contours),
            'copy': len(copy_contours),
            'rubbing_original': len(rubbing_contours),
            'matched_rate': len(matched_rubbing_contours) / len(copy_contours) if len(copy_contours) > 0 else 0,
            'contour_count_match': len(matched_rubbing_contours) == len(copy_contours)
        },
        'contour_areas': {
            'rubbing_top_n_areas': [round(cv2.contourArea(cnt)) for cnt in matched_rubbing_contours],
            'copy_areas': [round(cv2.contourArea(cnt)) for cnt in copy_contours]
        },
        'match_details': {
            'match_pairs': match_pairs
        },
        'parameters': {
            'min_area': min_area,
            'similarity_threshold': 0.5,
            'duplicate_removal': True
        }
    }
    
    # 8. 可视化结果
    if output_dir and save_intermediate:
        visualize_results(
            rubbing_binary, copy_binary,
            matched_rubbing_contours,
            copy_contours,
            report, output_dir
        )
    
    return report

# ======================== 可视化函数 ========================
def visualize_results(rubbing_binary, copy_binary, 
                    rubbing_contours, copy_contours,
                    report, output_dir):
    labels = get_bilingual_labels()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 拓片原始二值图
    axes[0, 0].imshow(rubbing_binary, cmap='gray')
    axes[0, 0].set_title('Original Rubbing Binary')
    axes[0, 0].axis('off')
    
    # 2. 摹本原始二值图
    axes[0, 1].imshow(copy_binary, cmap='gray')
    axes[0, 1].set_title('Original Copy Binary')
    axes[0, 1].axis('off')
    
    # 3. 拓片匹配后轮廓（标注序号和面积）
    rubbing_color = cv2.cvtColor(rubbing_binary, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(rubbing_color, rubbing_contours, -1, (0, 0, 255), 2)
    for i, cnt in enumerate(rubbing_contours):
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # 绘制标注
            fig_temp, ax_temp = plt.subplots(figsize=(0.1, 0.1))
            ax_temp.text(cx, cy, f'Contour {i+1}\nArea: {round(area)}', 
                        fontsize=6, color='red')
            ax_temp.axis('off')
            fig_temp.canvas.draw()
            img_rgba = np.frombuffer(fig_temp.canvas.buffer_rgba().tobytes(), dtype=np.uint8)
            img_rgba = img_rgba.reshape(fig_temp.canvas.get_width_height()[::-1] + (4,))
            img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
            img_bgr = cv2.resize(img_bgr, (rubbing_color.shape[1], rubbing_color.shape[0]))
            rubbing_color = cv2.addWeighted(rubbing_color, 1, img_bgr, 1, 0)
            plt.close(fig_temp)
    axes[1, 0].imshow(rubbing_color)
    axes[1, 0].set_title(f"Matched Rubbing Contours (Top {len(copy_contours)})")
    axes[1, 0].axis('off')
    
    # 4. 摹本轮廓（标注序号和面积）
    copy_color = cv2.cvtColor(copy_binary, cv2.COLOR_GRAY2BGR)
    for i, cnt in enumerate(copy_contours):
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # 绘制标注
            fig_temp, ax_temp = plt.subplots(figsize=(0.1, 0.1))
            ax_temp.text(cx, cy, f'Contour {i+1}\nArea: {round(area)}', 
                        fontsize=6, color='green')
            ax_temp.axis('off')
            fig_temp.canvas.draw()
            img_rgba = np.frombuffer(fig_temp.canvas.buffer_rgba().tobytes(), dtype=np.uint8)
            img_rgba = img_rgba.reshape(fig_temp.canvas.get_width_height()[::-1] + (4,))
            img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
            img_bgr = cv2.resize(img_bgr, (copy_color.shape[1], copy_color.shape[0]))
            copy_color = cv2.addWeighted(copy_color, 1, img_bgr, 1, 0)
            plt.close(fig_temp)
    cv2.drawContours(copy_color, copy_contours, -1, (0, 255, 0), 2)
    axes[1, 1].imshow(copy_color)
    axes[1, 1].set_title('Copy Contours')
    axes[1, 1].axis('off')
    
    # 底部信息标注
    contour_counts = report['contour_counts']
    info_text = (
        f"{labels['Completeness Score']}: {report['completeness_score']:.3f}\n"
        f"Rubbing original contour count: {contour_counts['rubbing_original']}\n"
        f"Matched rubbing contour count: {len(rubbing_contours)} (Top {len(copy_contours)})\n"
        f"Copy contour count: {contour_counts['copy']}\n"
        f"Contour count match: {'Yes' if contour_counts['contour_count_match'] else 'No'}\n"
        f"Matching success rate: {contour_counts['matched_rate']:.1%}\n"
        f"Parameters: min_area={report['parameters']['min_area']}, similarity threshold={report['parameters']['similarity_threshold']}"
    )
    plt.figtext(0.5, 0.01, info_text, 
               ha='center', fontsize=10, 
               bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'Completeness_Analysis.png')
    ensure_directory_exists(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Visualization result saved: {output_path}")
    
    # 匹配对同图可视化
    match_pairs = report['match_details']['match_pairs']
    merged_canvas = merge_contours_on_canvas(
        rubbing_binary, copy_binary,
        rubbing_contours, copy_contours,
        match_pairs
    )
    # 保存合并后的匹配图
    merged_path = os.path.join(output_dir, 'Matched_Contours_Merged.png')
    cv2.imwrite(merged_path, merged_canvas)
    logging.info(f"Merged matched contours visualization saved: {merged_path}")
    
    # 生成匹配图例
    num_matches = len(match_pairs)
    if num_matches > 0:
        legend_canvas = np.ones((100 + num_matches*30, 400, 3), dtype=np.uint8) * 255
        # 图例标题
        cv2.putText(legend_canvas, "Match Pair Color Legend", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        match_colors = generate_unique_colors(num_matches)
        for i, (t_idx, m_idx, sim) in enumerate(match_pairs):
            y = 60 + i * 30
            color = match_colors[i]
            # 绘制颜色块
            cv2.rectangle(legend_canvas, (20, y), (60, y + 20), color, -1)
            # 绘制匹配关系文字
            text = f"Rubbing {t_idx+1} ↔ Copy {m_idx+1} (Similarity: {sim:.3f})"
            cv2.putText(legend_canvas, text, (80, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        legend_path = os.path.join(output_dir, 'Match_Legend.png')
        cv2.imwrite(legend_path, legend_canvas)
        logging.info(f"Match legend saved: {legend_path}")

# ======================== 综合质量评估函数 ========================
def assess_quality(rubbing_path, copy_path, output_dir=None, save_intermediate=True):
    completeness_report = assess_completeness_robust(rubbing_path, copy_path, output_dir, save_intermediate=save_intermediate)
    completeness_score = completeness_report['completeness_score'] * 100
    quality_score = completeness_score
    report = {
        'quality_score': round(quality_score, 2),
        'completeness_score': round(completeness_score, 2),
        'completeness_details': completeness_report,
        'image_type': detect_image_type(rubbing_path)
    }
    if output_dir:
        save_quality_report(report, output_dir)
    return report

def save_quality_report(report, output_dir):
    report_path = os.path.join(output_dir, "Quality_Report.txt")
    ensure_directory_exists(report_path)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Rubbing Completeness Quality Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Image Type: {report['image_type']}\n")
        f.write(f"Quality Score: {report['quality_score']:.2f}/100\n\n")
        f.write("Detailed Scores:\n")
        f.write(f"  Completeness Score: {report['completeness_score']:.2f}\n\n")
        f.write("Evaluation Criteria:\n")
        f.write("• Completeness: Optimal matching similarity between top n rubbing contours and n copy contours, higher score indicates better completeness\n")
        f.write("• Black background with white text feature: Text is 255, background is 0\n")
        f.write("• Matching rule: Uses copy contour count n as standard, extracts top n largest rubbing contours, and applies greedy algorithm for optimal matching\n")
        f.write("• External contour extraction: Uses RETR_EXTERNAL throughout, excluding internal sub-contours\n\n")
        f.write("Completeness Assessment Details:\n")
        details = report['completeness_details']
        f.write(f"  Hu Moment Similarity: {details['hu_similarity']:.3f}\n")
        f.write(f"  Contour Counts: Rubbing original={details['contour_counts']['rubbing_original']}, Matched rubbing (top n)={details['contour_counts']['rubbing']}, Copy={details['contour_counts']['copy']}\n")
        f.write(f"  Contour Count Match: {'Yes' if details['contour_counts']['contour_count_match'] else 'No'}\n")
        f.write(f"  Rubbing Top n Contour Areas: {details['contour_areas']['rubbing_top_n_areas']} pixels\n")
        f.write(f"  Copy Contour Areas: {details['contour_areas']['copy_areas']} pixels\n")
        f.write(f"  Matching Success Rate: {details['contour_counts']['matched_rate']:.1%}\n")
        f.write(f"  Parameters: min_area={details['parameters']['min_area']}, similarity threshold={details['parameters']['similarity_threshold']}\n")
        # 匹配对详情
        f.write("\nMatch Pair Details (Rubbing Index - Copy Index - Similarity):\n")
        for t_idx, m_idx, sim in details['match_details']['match_pairs']:
            f.write(f"  Rubbing {t_idx+1} ↔ Copy {m_idx+1} (Similarity: {sim:.3f})\n")
    logging.info(f"Quality report saved: {report_path}")

# ======================== 质量评估可视化 ========================
def visualize_quality_results(report, output_dir):
    labels = get_bilingual_labels()
    fig, ax = plt.subplots(figsize=(8, 6))
    scores = [report['completeness_score'], report['quality_score']]
    score_labels = ['Completeness', 'Quality Score']
    colors = ['#ff9999', '#99ff99']
    bars = ax.bar(score_labels, scores, color=colors, alpha=0.7)
    ax.set_title('Rubbing Quality Scores')
    ax.set_ylabel('Score (0-100)')
    ax.set_ylim(0, 100)
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{score:.1f}', ha='center', va='bottom')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'Quality_Assessment_Summary.png')
    ensure_directory_exists(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Quality assessment summary visualization saved: {output_path}")

# ======================== 主程序 ========================
def main():
    rubbing_path = "/work/home/succuba/BRISQUE/庭_xe5j0rm6ao/xe5j0rm6ao.png"
    copy_path = "/work/home/succuba/BRISQUE/test_tapian/庭_xe5j0rm6ao/xe5j0rm6ao.png"
    output_dir = "/work/home/succuba/BRISQUE/results_test_ting4"
    print("Starting rubbing completeness quality assessment...")
    print("=" * 50)
    try:
        print("Performing image diagnosis...")
        diagnose_image_issues(rubbing_path, copy_path, output_dir)
        print("Performing quality assessment...")
        quality_report = assess_quality(rubbing_path, copy_path, output_dir, save_intermediate=True)
        visualize_quality_results(quality_report, output_dir)
        print("\nAssessment completed!")
        print("=" * 50)
        details = quality_report['completeness_details']
        print(f"Image Type: {quality_report['image_type']}")
        print(f"Overall Quality Score: {quality_report['quality_score']:.2f}/100")
        print(f"Completeness Score: {quality_report['completeness_score']:.2f}")
        print(f"Rubbing Original Contour Count: {details['contour_counts']['rubbing_original']}")
        print(f"Matched Rubbing Contour Count (Top {details['contour_counts']['copy']}): {details['contour_counts']['rubbing']}")
        print(f"Copy Contour Count: {details['contour_counts']['copy']}")
        print(f"Contour Count Match: {'Yes' if details['contour_counts']['contour_count_match'] else 'No'}")
        print(f"Matching Success Rate: {details['contour_counts']['matched_rate']:.1%}")
        print(f"\nAll results saved to: {output_dir}")
    except Exception as e:
        print(f"Error occurred during processing: {e}")
        traceback.print_exc()

def batch_process(rubbing_dir, copy_dir, output_dir):
    rubbing_files = []
    for root, _, files in os.walk(rubbing_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                rel_path = os.path.relpath(root, rubbing_dir)
                rubbing_files.append({
                    'path': os.path.join(root, file),
                    'rel_path': rel_path,
                    'filename': file
                })
    if not rubbing_files:
        print(f"No rubbing images found in directory {rubbing_dir}")
        return
    results = []
    processed_count = 0
    for rubbing_info in rubbing_files:
        rubbing_path = rubbing_info['path']
        rel_path = rubbing_info['rel_path']
        filename = rubbing_info['filename']
        copy_subdir = os.path.join(copy_dir, rel_path)
        copy_path = os.path.join(copy_subdir, filename)
        if not os.path.exists(copy_path):
            print(f"Warning: Copy file {copy_path} not found (corresponding to rubbing: {rubbing_path})")
            continue
        print(f"Processing rubbing: {rel_path}/{filename}")
        try:
            file_output_dir = os.path.join(output_dir, rel_path, os.path.splitext(filename)[0])
            os.makedirs(file_output_dir, exist_ok=True)
            quality_report = assess_quality(rubbing_path, copy_path, file_output_dir, save_intermediate=False)
            details = quality_report['completeness_details']
            results.append({
                'filename': filename,
                'subfolder': rel_path,
                'quality_score': quality_report['quality_score'],
                'completeness_score': quality_report['completeness_score'],
                'rubbing_original_count': details['contour_counts']['rubbing_original'],
                'rubbing_matched_count': details['contour_counts']['rubbing'],
                'copy_count': details['contour_counts']['copy'],
                'contour_count_match': details['contour_counts']['contour_count_match'],
                'matched_rate': details['contour_counts']['matched_rate']
            })
            processed_count += 1
            print(f"✅ Completed: Quality Score={quality_report['quality_score']:.2f}, Contour Match={'Success' if details['contour_counts']['contour_count_match'] else 'Failure'}, Matching Success Rate={details['contour_counts']['matched_rate']:.1%}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "Batch_Quality_Results.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Batch processing results saved to: {csv_path}")
        print("\nBatch Processing Summary:")
        print(f"Processed Images: {processed_count}/{len(rubbing_files)}")
        print(f"Average Quality Score: {df['quality_score'].mean():.2f}")
        print(f"Contour Match Rate: {df['contour_count_match'].sum()}/{len(df)} ({df['contour_count_match'].mean()*100:.1f}%)")
        print(f"Average Matching Success Rate: {df['matched_rate'].mean():.1%}")
        if 'subfolder' in df.columns:
            print("\nBy Subfolder:")
            grouped = df.groupby('subfolder').agg({
                'quality_score': 'mean',
                'contour_count_match': 'mean',
                'matched_rate': 'mean',
                'filename': 'count'
            }).rename(columns={'filename': 'count', 'contour_count_match': 'Contour Match Rate'})
            grouped['Contour Match Rate'] = grouped['Contour Match Rate'] * 100
            print(grouped.round(2))
    else:
        print("No images processed successfully")

# ======================== 主程序入口 ========================
if __name__ == "__main__":
    main()
    # 批量处理示例（取消注释以使用）
    # rubbing_dir = "/work/home/succuba/BRISQUE/results"
    # copy_dir = "/work/home/succuba/BRISQUE/dat
    # a/moben"
    # output_dir = "/work/home/succuba/BRISQUE/batch_results"
    # batch_process(rubbing_dir, copy_dir, output_dir)