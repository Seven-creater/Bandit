import json
import os
import sys

def fix_json_file(filepath):
    """修复损坏的JSON文件"""
    print(f"修复 {filepath}...")
    
    try:
        # 读取文件内容
        with open(filepath, 'r') as f:
            content = f.read()
        
        # 找到最后一个完整的 } 或 ]
        # 通常问题出在文件末尾
        last_brace = content.rfind('}')
        last_bracket = content.rfind(']')
        
        if last_brace == -1 and last_bracket == -1:
            print(f"  ❌ 无法找到有效的JSON结束符")
            return False
        
        # 尝试找到正确的结束位置
        # 通常格式是: ...}]\n}
        # 我们需要找到倒数第二个 }
        
        # 先尝试解析整个文件
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            print(f"  ✅ 文件已经是有效的JSON")
            return True
        except json.JSONDecodeError as e:
            error_pos = e.pos
            print(f"  ⚠️  JSON错误位置: {error_pos}")
            
            # 截取到错误位置之前
            valid_content = content[:error_pos]
            
            # 尝试补全JSON结构
            # 计算需要的闭合符号
            open_braces = valid_content.count('{') - valid_content.count('}')
            open_brackets = valid_content.count('[') - valid_content.count(']')
            
            # 移除末尾的不完整内容
            valid_content = valid_content.rstrip().rstrip(',')
            
            # 添加闭合符号
            for _ in range(open_brackets):
                valid_content += '\n  ]'
            for _ in range(open_braces):
                valid_content += '\n}'
            
            # 尝试解析修复后的内容
            try:
                data = json.loads(valid_content)
                
                # 备份原文件
                backup_path = filepath + '.backup'
                os.rename(filepath, backup_path)
                
                # 写入修复后的文件
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"  ✅ 修复成功！备份保存为: {backup_path}")
                return True
            except Exception as e2:
                print(f"  ❌ 修复失败: {e2}")
                return False
    
    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        return False

def main():
    base_dir = "/root/autodl-tmp/ai_study/bandit_study/experiments"
    
    experiments = [
        "2_restless_bandit",
        "3_contextual_bandit",
        "4_adversarial_bandit",
        "5_sleeping_bandit"
    ]
    
    print("开始修复JSON文件...\n")
    
    success_count = 0
    for exp_name in experiments:
        json_path = os.path.join(base_dir, exp_name, "results.json")
        if os.path.exists(json_path):
            if fix_json_file(json_path):
                success_count += 1
        else:
            print(f"跳过 {exp_name}: results.json 不存在")
    
    print(f"\n✅ 成功修复 {success_count}/{len(experiments)} 个文件！")

if __name__ == "__main__":
    main()

