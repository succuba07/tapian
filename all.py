import pandas as pd
import os

def merge_and_calculate_scores(clarity_csv, completeness_csv, output_csv):
    """
    融合清晰度和完整性得分，计算最终质量得分
    
    参数:
    clarity_csv: 清晰度得分CSV文件路径
    completeness_csv: 完整性得分CSV文件路径
    output_csv: 输出CSV文件路径
    """
    try:
        # 读取CSV文件
        clarity_df = pd.read_csv(clarity_csv)
        completeness_df = pd.read_csv(completeness_csv)
        
        # 检查数据列
        print("清晰度文件列名:", clarity_df.columns.tolist())
        print("完整性文件列名:", completeness_df.columns.tolist())
        
        # 确保关键列是字符串类型
        for col in ['filename', 'subfolder']:
            if col in clarity_df.columns:
                clarity_df[col] = clarity_df[col].astype(str)
            if col in completeness_df.columns:
                completeness_df[col] = completeness_df[col].astype(str)
        
        # 合并数据框 - 基于filename和subfolder
        merged_df = pd.merge(
            clarity_df, 
            completeness_df, 
            on=['filename', 'subfolder'], 
            how='inner'
        )
        
        if merged_df.empty:
            print("错误: 没有匹配的记录")
            return
        
        # 计算最终质量得分 (清晰度30% + 完整性70%)
        merged_df['quality_score'] = (
            merged_df['clarity_score'] * 0.3 + 
            merged_df['completeness_score'] * 0.7
        )
        
        # 四舍五入保留两位小数
        merged_df['quality_score'] = merged_df['quality_score'].round(2)
        
        # 选择需要的列
        result_df = merged_df[['filename', 'subfolder', 'clarity_score', 'completeness_score', 'quality_score']]
        
        # 保存结果
        result_df.to_csv(output_csv, index=False)
        print(f"成功生成综合质量报告: {output_csv}")
        print(f"处理记录数: {len(result_df)}")
        
        return result_df
    
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        return None

def main():
    # 配置路径 - 修改为您的实际文件路径
    clarity_csv = "/work/home/succuba/BRISQUE/results/clarity_scores.csv"  # 清晰度得分文件
    completeness_csv = "/work/home/succuba/BRISQUE/batch_results/batch_quality_results.csv"  # 完整性得分文件
    output_csv = "/work/home/succuba/BRISQUE/results/final_quality_scores.csv"  # 输出文件
    
    # 检查文件是否存在
    if not os.path.exists(clarity_csv):
        print(f"错误: 清晰度得分文件不存在: {clarity_csv}")
        return
    
    if not os.path.exists(completeness_csv):
        print(f"错误: 完整性得分文件不存在: {completeness_csv}")
        return
    
    # 处理数据
    result = merge_and_calculate_scores(clarity_csv, completeness_csv, output_csv)
    
    if result is not None:
        # 打印前5条结果
        print("\n前5条结果:")
        print(result.head())

if __name__ == "__main__":
    main()