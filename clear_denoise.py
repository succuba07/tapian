# -*- coding: utf-8 -*-
"""
甲骨文拓片骨架提取系统 - 增强版
1. 先计算原始拓片的清晰度并生成报告
2. 然后生成骨架图并生成报告
保持目录结构不变
"""

import numpy as np
import cv2
import os
from datetime import datetime
import glob
import pandas as pd
import logging

# 配置日志系统
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClarityEvaluator:
    """清晰度评估器 - 专门用于评估甲骨文拓片的清晰度"""
    
    def __init__(self):
        pass
    
    def calculate_clarity_score(self, image_path):
        """
        计算单张图像文字区域的平均亮度得分（针对黑底白字拓片）
        返回: 亮度得分 (0-100)，失败返回None
        """
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                logging.error(f"无法读取图像: {image_path}")
                return None
            
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 使用Otsu阈值处理分离文字和背景（黑底白字）
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 提取文字区域（白色像素）
            text_pixels = gray[binary == 255]
            if len(text_pixels) == 0:
                logging.warning(f"图像 {os.path.basename(image_path)} 中未检测到文字区域")
                return 0.0
            
            # 计算文字区域平均灰度值（0-255范围）
            avg_brightness = np.mean(text_pixels)
            
            # 归一化到0-100分：亮度越高得分越高（黑底白字）
            brightness_score = (avg_brightness / 255) * 100
            return round(brightness_score, 2)
        
        except Exception as e:
            logging.error(f"计算清晰度得分时出错: {str(e)}")
            return None

class OracleBoneSkeletonExtractor:
    """甲骨文骨架提取器 - 专门用于生成test_skeleton图"""
    
    def __init__(self, min_area=70, smoothing=True):
        self.min_area = min_area
        self.smoothing = smoothing
        
    def extract_skeleton(self, image_path):
        """从单张图像提取骨架"""
        try:
            # 读取图像
            if isinstance(image_path, str):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                image = image_path
                
            if image is None:
                return None
            
            # 中值滤波预处理
            denoised = cv2.medianBlur(image, 3)
            
            # 自适应二值化
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 形态学操作增强连通性
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # 提取轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 创建轮廓掩码
            contour_mask = np.zeros_like(binary)
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= self.min_area]
            cv2.drawContours(contour_mask, valid_contours, -1, 255, -1)
            
            # 平滑处理
            if self.smoothing:
                contour_mask = self._smooth_contour(contour_mask)
            
            return contour_mask
            
        except Exception as e:
            logging.error(f"处理图像时出错: {e}")
            return None
    
    def _smooth_contour(self, contour_mask):
        """平滑轮廓掩码"""
        # 形态学闭操作填充小孔
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        smoothed = cv2.morphologyEx(contour_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # 形态学开操作去除毛刺
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel_open)
        
        return smoothed

class BatchProcessor:
    """批量处理器 - 处理清晰度评估和骨架提取"""
    
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.clarity_evaluator = ClarityEvaluator()
        self.skeleton_extractor = OracleBoneSkeletonExtractor(min_area=70, smoothing=True)
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
        
    def process_clarity_evaluation(self):
        """批量处理清晰度评估"""
        if not os.path.exists(self.input_dir):
            raise ValueError(f"输入目录不存在: {self.input_dir}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'failed_files': []
        }
        
        results = []
        
        print(f"开始清晰度评估: {self.input_dir}")
        print("=" * 50)
        
        # 遍历所有子目录
        for root, dirs, files in os.walk(self.input_dir):
            # 计算相对路径
            rel_path = os.path.relpath(root, self.input_dir)
            
            # 处理当前目录的文件
            for file in files:
                if self._is_image_file(file):
                    input_path = os.path.join(root, file)
                    stats['total'] += 1
                    
                    # 计算清晰度得分
                    clarity_score = self.clarity_evaluator.calculate_clarity_score(input_path)
                    
                    if clarity_score is not None:
                        # 记录结果 - 包含filename, subfolder, clarity_score
                        results.append({
                            'filename': file,
                            'subfolder': rel_path,
                            'clarity_score': clarity_score
                        })
                        stats['success'] += 1
                        print(f"✅ 成功: {file} | 清晰度: {clarity_score:.2f}")
                    else:
                        stats['failed'] += 1
                        stats['failed_files'].append(input_path)
                        print(f"❌ 失败: {file}")
        
        # 保存清晰度得分结果
        if results:
            self._save_clarity_results(results)
        
        # 生成报告
        self._generate_clarity_report(stats, results)
        return stats
    
    def process_skeleton_extraction(self):
        """批量处理骨架提取"""
        if not os.path.exists(self.input_dir):
            raise ValueError(f"输入目录不存在: {self.input_dir}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'failed_files': []
        }
        
        print(f"开始骨架提取: {self.input_dir}")
        print("=" * 50)
        
        # 遍历所有子目录
        for root, dirs, files in os.walk(self.input_dir):
            # 计算相对路径
            rel_path = os.path.relpath(root, self.input_dir)
            
            # 创建对应的输出子目录
            if rel_path != '.':
                output_subdir = os.path.join(self.output_dir, rel_path)
                os.makedirs(output_subdir, exist_ok=True)
            else:
                output_subdir = self.output_dir
            
            # 处理当前目录的文件
            for file in files:
                if self._is_image_file(file):
                    input_path = os.path.join(root, file)
                    stats['total'] += 1
                    
                    # 处理图像
                    skeleton = self.skeleton_extractor.extract_skeleton(input_path)
                    
                    if skeleton is not None:
                        # 生成输出文件名（保持原名，添加_skeleton后缀）
                        output_filename = self._get_output_filename(file)
                        output_path = os.path.join(output_subdir, output_filename)
                        
                        # 保存骨架图
                        cv2.imwrite(output_path, skeleton)
                        stats['success'] += 1
                        print(f"✅ 成功: {file} -> {output_filename}")
                    else:
                        stats['failed'] += 1
                        stats['failed_files'].append(input_path)
                        print(f"❌ 失败: {file}")
        
        # 生成报告
        self._generate_skeleton_report(stats)
        return stats
    
    def _is_image_file(self, filename):
        """检查是否为图像文件"""
        ext = os.path.splitext(filename)[1].lower()
        return ext in self.supported_formats
    
    def _get_output_filename(self, filename):
        """生成输出文件名"""
        name, ext = os.path.splitext(filename)
        return f"{name}.png"
    
    def _save_clarity_results(self, results):
        """保存清晰度得分结果"""
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.output_dir, "clarity_scores.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logging.info(f"清晰度得分已保存到: {csv_path}")
        
        # 打印统计摘要
        if not df.empty:
            avg_score = df['clarity_score'].mean()
            min_score = df['clarity_score'].min()
            max_score = df['clarity_score'].max()
            logging.info(f"清晰度得分统计: 平均={avg_score:.2f}, 最小={min_score:.2f}, 最大={max_score:.2f}")
    
    def _generate_clarity_report(self, stats, results):
        """生成清晰度评估报告"""
        report_path = os.path.join(self.output_dir, "clarity_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("甲骨文拓片清晰度评估报告\n")
            f.write("=" * 40 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入目录: {self.input_dir}\n")
            f.write(f"输出目录: {self.output_dir}\n\n")
            f.write("处理统计:\n")
            f.write(f"总文件数: {stats['total']}\n")
            f.write(f"成功评估: {stats['success']}\n")
            f.write(f"评估失败: {stats['failed']}\n")
            f.write(f"成功率: {stats['success']/stats['total']*100:.1f}%\n")
            
            if results:
                df = pd.DataFrame(results)
                avg_score = df['clarity_score'].mean()
                min_score = df['clarity_score'].min()
                max_score = df['clarity_score'].max()
                f.write("\n清晰度得分统计:\n")
                f.write(f"平均得分: {avg_score:.2f}/100\n")
                f.write(f"最低得分: {min_score:.2f}/100\n")
                f.write(f"最高得分: {max_score:.2f}/100\n")
                
                # 按子文件夹分组统计
                if 'subfolder' in df.columns:
                    grouped = df.groupby('subfolder')['clarity_score'].agg(['mean', 'count'])
                    f.write("\n按子文件夹统计:\n")
                    f.write(grouped.to_string())
            
            if stats['failed_files']:
                f.write("\n失败文件列表:\n")
                for file in stats['failed_files']:
                    f.write(f"- {file}\n")
        
        print(f"\n📊 清晰度报告已保存: {report_path}")
    
    def _generate_skeleton_report(self, stats):
        """生成骨架提取报告"""
        report_path = os.path.join(self.output_dir, "skeleton_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("甲骨文拓片骨架提取报告\n")
            f.write("=" * 40 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入目录: {self.input_dir}\n")
            f.write(f"输出目录: {self.output_dir}\n\n")
            f.write("处理统计:\n")
            f.write(f"总文件数: {stats['total']}\n")
            f.write(f"成功提取: {stats['success']}\n")
            f.write(f"提取失败: {stats['failed']}\n")
            f.write(f"成功率: {stats['success']/stats['total']*100:.1f}%\n")
            
            if stats['failed_files']:
                f.write("\n失败文件列表:\n")
                for file in stats['failed_files']:
                    f.write(f"- {file}\n")
        
        print(f"\n📊 骨架提取报告已保存: {report_path}")

def main():
    """主函数"""
    print("甲骨文拓片处理系统")
    print("=" * 40)
    
    # 配置路径 - 修改为您需要的路径
    input_directory = "/work/home/succuba/BRISQUE/data/tapian"  # 输入目录
    output_directory = "/work/home/succuba/BRISQUE/results"  # 输出目录
    
    # 检查输入目录
    if not os.path.exists(input_directory):
        print(f"错误: 输入目录不存在: {input_directory}")
        print("请修改 input_directory 变量")
        return
    
    # 创建处理器
    processor = BatchProcessor(
        input_dir=input_directory,
        output_dir=output_directory
    )
    
    try:
        # 第一步：清晰度评估
        print("\n" + "=" * 40)
        print("开始清晰度评估")
        print("=" * 40)
        clarity_stats = processor.process_clarity_evaluation()
        
        print("\n" + "=" * 40)
        print("清晰度评估完成!")
        print(f"成功评估: {clarity_stats['success']}/{clarity_stats['total']} 个文件")
        print(f"成功率: {clarity_stats['success']/clarity_stats['total']*100:.1f}%")
        
        # 第二步：骨架提取
        print("\n" + "=" * 40)
        print("开始骨架提取")
        print("=" * 40)
        skeleton_stats = processor.process_skeleton_extraction()
        
        print("\n" + "=" * 40)
        print("骨架提取完成!")
        print(f"成功提取: {skeleton_stats['success']}/{skeleton_stats['total']} 个文件")
        print(f"成功率: {skeleton_stats['success']/skeleton_stats['total']*100:.1f}%")
        print(f"结果保存在: {output_directory}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

# 单文件测试函数
def test_single_file():
    """测试单文件处理"""
    test_image = "/work/home/succuba/BRISQUE/data/tapian/h1yvbqcpot.png"  # 测试文件
    
    if not os.path.exists(test_image):
        print("测试文件不存在")
        return
    
    # 清晰度评估
    evaluator = ClarityEvaluator()
    clarity_score = evaluator.calculate_clarity_score(test_image)
    print(f"清晰度得分: {clarity_score:.2f}")
    
    # 骨架提取
    extractor = OracleBoneSkeletonExtractor()
    skeleton = extractor.extract_skeleton(test_image)
    
    if skeleton is not None:
        output_path = "/tmp/test_skeleton_result.png"
        cv2.imwrite(output_path, skeleton)
        print(f"骨架提取完成! 结果保存到: {output_path}")
    else:
        print("骨架提取失败")

if __name__ == "__main__":
    # 运行批量处理
    main()
    
    # 如果要测试单文件，取消下面的注释
    # test_single_file()