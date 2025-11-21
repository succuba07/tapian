import cv2
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import os
from datetime import datetime
import logging
import traceback
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ======================== Label Function (English only) ========================
def get_english_labels():
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

# ======================== Core Utility Functions ========================
def get_top_n_contour_groups(groups, n):
    """Sort contour groups by outer contour area (descending) and return top n"""
    if len(groups) <= n:
        return groups
    return sorted(groups, key=lambda x: cv2.contourArea(x['outer']), reverse=True)[:n]

def calculate_combined_hu_similarity(group1, group2):
    """Calculate combined Hu moment similarity (outer + inner contours)"""
    def get_hu_vector(contour):
        """Hu moment vector for a single contour"""
        moments = cv2.moments(contour)
        if moments['m00'] < 1e-6:  # Avoid zero-area contours
            return np.zeros(7)
        return cv2.HuMoments(moments).flatten()
    
    # Outer contour Hu moments
    hu1_outer = get_hu_vector(group1['outer'])
    hu2_outer = get_hu_vector(group2['outer'])
    
    # Average Hu moments of inner contours (zero vector if no inner contours)
    hu1_inners = np.mean([get_hu_vector(inner) for inner in group1['inners']], axis=0) if group1['inners'] else np.zeros(7)
    hu2_inners = np.mean([get_hu_vector(inner) for inner in group2['inners']], axis=0) if group2['inners'] else np.zeros(7)
    
    # Combine outer and inner features
    hu1_combined = np.concatenate([hu1_outer, hu1_inners])
    hu2_combined = np.concatenate([hu2_outer, hu2_inners])
    
    # Convert distance to similarity (higher = more similar)
    dist = distance.euclidean(hu1_combined, hu2_combined)
    return 1.0 / (1.0 + dist)

def calculate_greedy_matching_similarity(rubbing_groups, copy_groups):
    """Greedy algorithm to match contour groups (outer + inner)"""
    if len(rubbing_groups) == 0 or len(copy_groups) == 0:
        logging.warning("No valid contour groups for matching")
        return 0.0, []
    
    # Calculate similarity matrix for all group pairs
    sim_matrix = []
    for rg in rubbing_groups:
        row_sims = [calculate_combined_hu_similarity(rg, cg) for cg in copy_groups]
        sim_matrix.append(row_sims)
    
    # Greedy matching (prioritize highest similarity unmatched pairs)
    used_r, used_c = set(), set()
    total_sim, match_pairs = 0.0, []
    all_matches = sorted([(sim_matrix[r][c], r, c) 
                         for r in range(len(rubbing_groups)) 
                         for c in range(len(copy_groups))], 
                         reverse=True, key=lambda x: x[0])
    
    for sim, r_idx, c_idx in all_matches:
        if r_idx not in used_r and c_idx not in used_c:
            total_sim += sim
            used_r.add(r_idx)
            used_c.add(c_idx)
            match_pairs.append((r_idx, c_idx, sim))
            logging.info(f"Match: Rubbing Group {r_idx+1} ↔ Copy Group {c_idx+1} (Similarity: {sim:.3f})")
    
    avg_sim = total_sim / len(match_pairs) if match_pairs else 0.0
    return avg_sim, match_pairs

# Color generation and visualization tools
def generate_unique_colors(n):
    """Generate n unique bright colors"""
    random.seed(42)
    return [(random.randint(60, 255), random.randint(60, 255), random.randint(60, 255)) for _ in range(n)]

def merge_contours_on_canvas(rubbing_binary, copy_binary, rubbing_groups, copy_groups, match_pairs):
    """Merge rubbing and copy contours on a canvas, mark matches with same color"""
    t_h, t_w = rubbing_binary.shape
    m_h, m_w = copy_binary.shape
    canvas = np.ones((max(t_h, m_h), t_w + m_w, 3), dtype=np.uint8) * 255  # White canvas
    
    # Draw rubbing contour groups (thick gray outer, thin gray inner)
    for group in rubbing_groups:
        cv2.drawContours(canvas, [group['outer']], -1, (128, 128, 128), 2)
        for inner in group['inners']:
            cv2.drawContours(canvas, [inner], -1, (128, 128, 128), 1)
    
    # Draw copy contour groups (with horizontal offset)
    m_offset = np.array([[t_w, 0]])  # Horizontal offset for copy
    for group in copy_groups:
        outer_shifted = group['outer'] + m_offset
        cv2.drawContours(canvas, [outer_shifted], -1, (128, 128, 128), 2)
        for inner in group['inners']:
            inner_shifted = inner + m_offset
            cv2.drawContours(canvas, [inner_shifted], -1, (128, 128, 128), 1)
    
    # Mark matched pairs (highlight with same color)
    if match_pairs:
        colors = generate_unique_colors(len(match_pairs))
        for (r_idx, c_idx, _), color in zip(match_pairs, colors):
            # Mark rubbing matched group
            r_group = rubbing_groups[r_idx]
            cv2.drawContours(canvas, [r_group['outer']], -1, color, 3)
            for inner in r_group['inners']:
                cv2.drawContours(canvas, [inner], -1, color, 2)
            # Mark copy matched group (with offset)
            c_group = copy_groups[c_idx]
            outer_shifted = c_group['outer'] + m_offset
            cv2.drawContours(canvas, [outer_shifted], -1, color, 3)
            for inner in c_group['inners']:
                inner_shifted = inner + m_offset
                cv2.drawContours(canvas, [inner_shifted], -1, color, 2)
            # Add text labels
            r_mom = cv2.moments(r_group['outer'])
            if r_mom['m00'] != 0:
                cv2.putText(canvas, f"R{r_idx+1}", 
                           (int(r_mom['m10']/r_mom['m00']), int(r_mom['m01']/r_mom['m00'])),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            c_mom = cv2.moments(c_group['outer'])
            if c_mom['m00'] != 0:
                cv2.putText(canvas, f"C{c_idx+1}", 
                           (int(c_mom['m10']/c_mom['m00']) + t_w, int(c_mom['m01']/c_mom['m00'])),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Add titles and separator line
    cv2.line(canvas, (t_w, 0), (t_w, canvas.shape[0]), (0, 0, 0), 2)
    cv2.putText(canvas, "Rubbing (Outer+Inner Contours)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(canvas, "Copy (Outer+Inner Contours)", (t_w + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return canvas

# ======================== Image Preprocessing ========================
def preprocess_image(image_path, output_dir=None, image_name="image", save_intermediate=True, is_copy=False):
    """Unified preprocessing: ensure text is 255, background is 0 (for contour extraction)"""
    if is_copy:
        # Copy: white background with black text → invert to text=255, background=0
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read copy image: {image_path}")
        # Critical: invert threshold to ensure text (target) is 255
        _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        denoised_img = cv2.medianBlur(binary_img, 3)  # Add denoising
        gray_img = img
        is_color = False
        logging.info("Copy preprocessing completed: Text=255, Background=0 (with denoising)")
    else:
        # Rubbing: black background with white text → direct threshold (text=255, background=0)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read rubbing image: {image_path}")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        denoised_img = cv2.medianBlur(gray_img, 3)
        _, binary_img = cv2.threshold(denoised_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        is_color = len(img.shape) == 3
        logging.info("Rubbing preprocessing completed: Text=255, Background=0 (with denoising)")
    
    # Save intermediate results
    if save_intermediate and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if is_copy:
            cv2.imwrite(os.path.join(output_dir, f"{image_name}_Original.png"), img)
            cv2.imwrite(os.path.join(output_dir, f"{image_name}_Denoised.png"), denoised_img)
            cv2.imwrite(os.path.join(output_dir, f"{image_name}_Binary.png"), binary_img)
        else:
            if is_color:
                cv2.imwrite(os.path.join(output_dir, f"{image_name}_Original.png"), img)
            cv2.imwrite(os.path.join(output_dir, f"{image_name}_Grayscale.png"), gray_img)
            cv2.imwrite(os.path.join(output_dir, f"{image_name}_Denoised.png"), denoised_img)
            cv2.imwrite(os.path.join(output_dir, f"{image_name}_Binary.png"), binary_img)
        create_preprocessing_flowchart(img, gray_img, denoised_img, binary_img, output_dir, image_name, is_copy)
    
    return {
        'original': img,
        'gray': gray_img,
        'denoised': denoised_img,
        'binary': binary_img,
        'is_color': is_color
    }

def create_preprocessing_flowchart(original, gray, denoised, binary, output_dir, image_name, is_copy):
    """Generate preprocessing flowchart"""
    if is_copy:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original Copy')
        axes[0].axis('off')
        axes[1].imshow(denoised, cmap='gray')
        axes[1].set_title('Denoised')
        axes[1].axis('off')
        axes[2].imshow(binary, cmap='gray')
        axes[2].set_title('Binary (Text=255)')
        axes[2].axis('off')
        plt.suptitle(f'{image_name} - Copy Preprocessing')
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        if len(original.shape) == 3:
            axes[0,0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            axes[0,0].imshow(original, cmap='gray')
        axes[0,0].set_title('Original')
        axes[0,0].axis('off')
        axes[0,1].imshow(gray, cmap='gray')
        axes[0,1].set_title('Grayscale')
        axes[0,1].axis('off')
        axes[1,0].imshow(denoised, cmap='gray')
        axes[1,0].set_title('Denoised')
        axes[1,0].axis('off')
        axes[1,1].imshow(binary, cmap='gray')
        axes[1,1].set_title('Binary (Text=255)')
        axes[1,1].axis('off')
        plt.suptitle(f'{image_name} - Rubbing Preprocessing')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{image_name}_Preprocessing.png"), dpi=300)
    plt.close()

# ======================== Hierarchical Contour Extraction ========================
def extract_hierarchical_contours(binary_image, min_area=30, save_dir=None, image_name="contours"):
    """
    Extract hierarchical structure (outer + inner contours)
    - Outer contours: outermost closed contours (no parent)
    - Inner contours: child contours inside outer contours
    """
    # Use RETR_CCOMP to extract all contours with hierarchy
    contours, hierarchy = cv2.findContours(
        binary_image, 
        cv2.RETR_CCOMP,   # Extract all contours with 2-level hierarchy (outer + inner)
        cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return []
    
    contour_groups = []
    # Iterate all outer contours (parent index = -1)
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:  # Mark of outer contour (no parent)
            outer_contour = contours[i]
            outer_area = cv2.contourArea(outer_contour)
            if outer_area < min_area:  # Filter small outer contours
                continue
            
            # Collect all inner contours under this outer contour
            inner_contours = []
            child_idx = hierarchy[0][i][2]  # First child contour index
            while child_idx != -1:
                inner_contour = contours[child_idx]
                inner_area = cv2.contourArea(inner_contour)
                if inner_area >= min_area:  # Filter small inner contours
                    inner_contours.append(inner_contour)
                child_idx = hierarchy[0][child_idx][0]  # Next sibling inner contour
            
            contour_groups.append({
                'outer': outer_contour,
                'inners': inner_contours,
                'outer_area': outer_area,
                'inner_count': len(inner_contours)
            })
    
    # Save contour extraction results (for verification)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        vis_img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        for group in contour_groups:
            # Outer contours with thick red lines
            cv2.drawContours(vis_img, [group['outer']], -1, (0, 0, 255), 2)
            # Inner contours with thin green lines
            for inner in group['inners']:
                cv2.drawContours(vis_img, [inner], -1, (0, 255, 0), 1)
        save_path = os.path.join(save_dir, f"{image_name}_Hierarchical_Contours.png")
        cv2.imwrite(save_path, vis_img)
        logging.info(f"Hierarchical contours saved to: {save_path}")
    
    return contour_groups

# ======================== Helper Functions ========================
def detect_image_type(image_path):
    """Determine image type (black background with white text / white background with black text)"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Check edge pixels to determine background color
    border = np.concatenate([img[0,:], img[-1,:], img[:,0], img[:,-1]])
    return 'black_bg_white_text' if np.mean(border) < 127 else 'white_bg_black_text'

def ensure_directory_exists(path):
    """Ensure directory exists"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ======================== Diagnosis and Evaluation ========================
def diagnose_image_issues(rubbing_path, copy_path, output_dir):
    """Generate image issue diagnosis report"""
    ensure_directory_exists(os.path.join(output_dir, "dummy.txt"))
    rubbing = cv2.imread(rubbing_path, cv2.IMREAD_GRAYSCALE)
    copy = cv2.imread(copy_path, cv2.IMREAD_GRAYSCALE)
    if rubbing is None or copy is None:
        raise ValueError("Failed to read images for diagnosis")
    
    # Unified binarization (text=255)
    _, rub_bin = cv2.threshold(rubbing, 127, 255, cv2.THRESH_BINARY)
    _, cop_bin = cv2.threshold(copy, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Extract contours for diagnosis
    rub_contours, _ = cv2.findContours(rub_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cop_contours, _ = cv2.findContours(cop_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw diagnosis figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes[0,0].imshow(rubbing, cmap='gray')
    axes[0,0].set_title('Original Rubbing')
    axes[0,0].axis('off')
    axes[0,1].imshow(copy, cmap='gray')
    axes[0,1].set_title('Original Copy')
    axes[0,1].axis('off')
    
    rub_vis = cv2.cvtColor(rub_bin, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(rub_vis, rub_contours, -1, (0,0,255), 2)
    axes[1,0].imshow(rub_vis)
    axes[1,0].set_title(f'Rubbing Contours ({len(rub_contours)})')
    axes[1,0].axis('off')
    
    cop_vis = cv2.cvtColor(cop_bin, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(cop_vis, cop_contours, -1, (0,255,0), 2)
    axes[1,1].imshow(cop_vis)
    axes[1,1].set_title(f'Copy Contours ({len(cop_contours)})')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Image_Diagnosis.png"), dpi=300)
    plt.close()
    logging.info("Image diagnosis report generated")

# ======================== Core Evaluation Function ========================
def assess_completeness_robust(rubbing_path, copy_path, output_dir=None, 
                              min_area=50, save_intermediate=True):
    """Evaluate rubbing-copy completeness (based on outer+inner contour matching)"""
    # Initialize directories
    intermediate_dir = os.path.join(output_dir, "Intermediate") if (output_dir and save_intermediate) else None
    if intermediate_dir:
        os.makedirs(intermediate_dir, exist_ok=True)
    
    # 1. Preprocessing (unify text=255, background=0)
    rubbing = preprocess_image(rubbing_path, intermediate_dir, "rubbing", save_intermediate, is_copy=False)
    copy = preprocess_image(copy_path, intermediate_dir, "copy", save_intermediate, is_copy=True)
    
    # 2. Extract hierarchical contour groups (outer + inner)
    rubbing_groups = extract_hierarchical_contours(
        rubbing['binary'], min_area, intermediate_dir, "rubbing"
    )
    copy_groups = extract_hierarchical_contours(
        copy['binary'], min_area, intermediate_dir, "copy"
    )
    logging.info(f"Number of rubbing contour groups: {len(rubbing_groups)}, copy contour groups: {len(copy_groups)}")
    
    # 3. Dynamic matching (use top n largest rubbing groups, n = number of copy groups)
    n = len(copy_groups)
    rubbing_top_n = get_top_n_contour_groups(rubbing_groups, n)
    logging.info(f"Rubbing groups for matching (top {n} largest): {len(rubbing_top_n)}")
    
    # 4. Greedy matching to calculate similarity
    hu_sim, match_pairs = calculate_greedy_matching_similarity(rubbing_top_n, copy_groups)
    
    # 5. Generate report
    report = {
        'completeness_score': hu_sim,
        'hu_similarity': hu_sim,
        'contour_counts': {
            'rubbing_original': len(rubbing_groups),
            'rubbing_matched': len(rubbing_top_n),
            'copy': n,
            'match_rate': len(rubbing_top_n)/n if n > 0 else 0,
            'count_match': len(rubbing_top_n) == n
        },
        'contour_details': {
            'rubbing_outer_areas': [g['outer_area'] for g in rubbing_top_n],
            'rubbing_inner_counts': [g['inner_count'] for g in rubbing_top_n],
            'copy_outer_areas': [g['outer_area'] for g in copy_groups],
            'copy_inner_counts': [g['inner_count'] for g in copy_groups]
        },
        'matches': match_pairs,
        'params': {'min_area': min_area}
    }
    
    # 6. Visualize results
    if output_dir and save_intermediate:
        visualize_results(rubbing['binary'], copy['binary'], rubbing_top_n, copy_groups, report, output_dir)
    
    return report

def visualize_results(rubbing_bin, copy_bin, rubbing_groups, copy_groups, report, output_dir):
    """Visualize matching results"""
    # 1. Draw contour group details
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes[0,0].imshow(rubbing_bin, cmap='gray')
    axes[0,0].set_title('Rubbing Binary')
    axes[0,0].axis('off')
    axes[0,1].imshow(copy_bin, cmap='gray')
    axes[0,1].set_title('Copy Binary')
    axes[0,1].axis('off')
    
    # Rubbing contour groups (label outer area and inner count)
    rub_vis = cv2.cvtColor(rubbing_bin, cv2.COLOR_GRAY2BGR)
    for i, g in enumerate(rubbing_groups):
        cv2.drawContours(rub_vis, [g['outer']], -1, (0,0,255), 2)
        for inner in g['inners']:
            cv2.drawContours(rub_vis, [inner], -1, (0,0,255), 1)
        mom = cv2.moments(g['outer'])
        if mom['m00'] > 0:
            cx, cy = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
            cv2.putText(rub_vis, f"G{i+1}\nA:{g['outer_area']:.0f}\nI:{g['inner_count']}",
                       (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
    axes[1,0].imshow(rub_vis)
    axes[1,0].set_title(f"Rubbing Groups (Top {len(copy_groups)})")
    axes[1,0].axis('off')
    
    # Copy contour groups
    cop_vis = cv2.cvtColor(copy_bin, cv2.COLOR_GRAY2BGR)
    for i, g in enumerate(copy_groups):
        cv2.drawContours(cop_vis, [g['outer']], -1, (0,255,0), 2)
        for inner in g['inners']:
            cv2.drawContours(cop_vis, [inner], -1, (0,255,0), 1)
        mom = cv2.moments(g['outer'])
        if mom['m00'] > 0:
            cx, cy = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
            cv2.putText(cop_vis, f"G{i+1}\nA:{g['outer_area']:.0f}\nI:{g['inner_count']}",
                       (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    axes[1,1].imshow(cop_vis)
    axes[1,1].set_title("Copy Groups")
    axes[1,1].axis('off')
    
    # Add statistics
    stats = report['contour_counts']
    plt.figtext(0.5, 0.01, 
               f"Completeness Score: {report['completeness_score']:.3f} | Match Rate: {stats['match_rate']:.1%} | "
               f"Rubbing Groups: {stats['rubbing_matched']} | Copy Groups: {stats['copy']}",
               ha='center', fontsize=10, bbox={"facecolor":"yellow", "alpha":0.3})
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Contour_Details.png"), dpi=300)
    plt.close()
    
    # 2. Draw merged match pairs
    merged_img = merge_contours_on_canvas(rubbing_bin, copy_bin, rubbing_groups, copy_groups, report['matches'])
    cv2.imwrite(os.path.join(output_dir, "Matched_Groups.png"), merged_img)
    
    # 3. Generate match legend
    if report['matches']:
        legend = np.ones((80 + len(report['matches'])*30, 400, 3), dtype=np.uint8)*255
        cv2.putText(legend, "Match Legend", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        colors = generate_unique_colors(len(report['matches']))
        for i, ((r, c, s), color) in enumerate(zip(report['matches'], colors)):
            y = 50 + i*30
            cv2.rectangle(legend, (20, y), (50, y+20), color, -1)
            cv2.putText(legend, f"Rubbing G{r+1} ↔ Copy G{c+1} (Sim: {s:.3f})",
                       (70, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.imwrite(os.path.join(output_dir, "Match_Legend.png"), legend)

# ======================== Comprehensive Quality Assessment ========================
def assess_quality(rubbing_path, copy_path, output_dir=None, save_intermediate=True):
    """Comprehensive quality assessment"""
    comp_report = assess_completeness_robust(rubbing_path, copy_path, output_dir, save_intermediate=save_intermediate)
    quality_score = round(comp_report['completeness_score'] * 100, 2)
    
    # Generate final report
    report = {
        'quality_score': quality_score,
        'completeness_score': round(comp_report['completeness_score'] * 100, 2),
        'details': comp_report,
        'image_type': detect_image_type(rubbing_path)
    }
    
    if output_dir:
        save_quality_report(report, output_dir)
        visualize_quality_summary(report, output_dir)
    
    return report

def save_quality_report(report, output_dir):
    """Save text report"""
    with open(os.path.join(output_dir, "Quality_Report.txt"), 'w', encoding='utf-8') as f:
        f.write("=== Oracle Bone Rubbing Completeness Assessment Report ===\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Image Type: {report['image_type']}\n")
        f.write(f"Overall Quality Score: {report['quality_score']}/100\n")
        f.write(f"Completeness Score: {report['completeness_score']}/100\n\n")
        
        f.write("=== Contour Group Matching Details ===\n")
        stats = report['details']['contour_counts']
        f.write(f"Original Rubbing Contour Groups: {stats['rubbing_original']}\n")
        f.write(f"Rubbing Groups Used for Matching: {stats['rubbing_matched']}\n")
        f.write(f"Copy Contour Groups: {stats['copy']}\n")
        f.write(f"Match Rate: {stats['match_rate']:.1%}\n\n")
        
        f.write("Rubbing Contour Group Details (Outer Area / Inner Count):\n")
        for i, (a, c) in enumerate(zip(report['details']['contour_details']['rubbing_outer_areas'],
                                       report['details']['contour_details']['rubbing_inner_counts'])):
            f.write(f"  Group {i+1}: Area={a:.0f}, Inner Contours={c}\n")  # Fixed: use f.write()
        
        f.write("\nCopy Contour Group Details (Outer Area / Inner Count):\n")
        for i, (a, c) in enumerate(zip(report['details']['contour_details']['copy_outer_areas'],
                                       report['details']['contour_details']['copy_inner_counts'])):
            f.write(f"  Group {i+1}: Area={a:.0f}, Inner Contours={c}\n")  # Fixed: use f.write()
        
        f.write("\nMatch Pair Similarities:\n")
        for r, c, s in report['details']['matches']:
            f.write(f"  Rubbing Group {r+1} ↔ Copy Group {c+1}: {s:.3f}\n")

def visualize_quality_summary(report, output_dir):
    """Visualize quality summary"""
    fig, ax = plt.subplots(figsize=(8, 5))
    scores = [report['completeness_score'], report['quality_score']]
    ax.bar(['Completeness', 'Quality'], scores, color=['#4CAF50', '#2196F3'])
    ax.set_ylim(0, 100)
    ax.set_title('Quality Assessment Summary')
    for i, v in enumerate(scores):
        ax.text(i, v+1, f"{v:.1f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Quality_Summary.png"), dpi=300)
    plt.close()

# ======================== Main Program ========================
def main():
    # Example paths (modify with your actual paths)
    rubbing_path = "/work/home/succuba/BRISQUE/庭_xe5j0rm6ao/xe5j0rm6ao.png"
    copy_path = "/work/home/succuba/BRISQUE/test_tapian/庭_xe5j0rm6ao/xe5j0rm6ao.png"
    output_dir = "/work/home/succuba/BRISQUE/results_test_ting6"
    
    print("Starting oracle bone rubbing completeness assessment...")
    try:
        # Image diagnosis
        diagnose_image_issues(rubbing_path, copy_path, output_dir)
        # Quality assessment
        report = assess_quality(rubbing_path, copy_path, output_dir, save_intermediate=True)
        # Output key results
        print("\nAssessment completed!")
        print(f"Overall Quality Score: {report['quality_score']}/100")
        print(f"Completeness Score: {report['completeness_score']}/100")
        print(f"Results saved to: {output_dir}")
    except Exception as e:
        print(f"Processing error: {str(e)}")
        traceback.print_exc()

def batch_process(rubbing_dir, copy_dir, output_dir):
    """Batch processing function"""
    for root, _, files in os.walk(rubbing_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                rub_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, rubbing_dir)
                cop_path = os.path.join(copy_dir, rel_path, file)
                if not os.path.exists(cop_path):
                    print(f"Skipping: Copy not found {cop_path}")
                    continue
                
                out_dir = os.path.join(output_dir, rel_path, os.path.splitext(file)[0])
                os.makedirs(out_dir, exist_ok=True)
                print(f"Processing {rel_path}/{file} ...")
                try:
                    report = assess_quality(rub_path, cop_path, out_dir, save_intermediate=False)
                    print(f"  Quality Score: {report['quality_score']}/100")
                except Exception as e:
                    print(f"  Processing failed: {str(e)}")

if __name__ == "__main__":
    main()
    # Uncomment for batch processing
    # batch_process("/path/to/rubbings", "/path/to/copies", "/path/to/output")