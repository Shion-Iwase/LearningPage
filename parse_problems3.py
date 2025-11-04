import re
import json

# 問題データを読み込み
with open('問題データ3_加工前.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# 問題を分割
problems = re.split(r'\n(?=問題3-\d+\n)', content)

# 問題11から65を処理
output_problems = []

for problem_text in problems:
    # 問題番号を抽出
    match = re.match(r'問題3-(\d+)', problem_text)
    if not match:
        continue

    problem_num = int(match.group(1))

    if problem_num < 11 or problem_num > 65:
        continue

    # 問題文を抽出（最初の行から「正解」の前まで）
    question_match = re.search(r'問題3-\d+\n(.+?)\n正解', problem_text, re.DOTALL)
    if not question_match:
        continue
    question = question_match.group(1).strip()

    # 正解を抽出
    correct_match = re.search(r'\n正解\n(.+?)(?:\n[^\n]+\n全体的な説明|\n全体的な説明)', problem_text, re.DOTALL)
    if not correct_match:
        continue
    correct_answer = correct_match.group(1).strip()

    # 選択肢を抽出（正解の前の部分から）
    choices_section = re.search(r'(?:問題3-\d+\n.+?\n)((?:.+?\n)+?)正解', problem_text, re.DOTALL)
    if choices_section:
        choices_text = choices_section.group(1)
        # 正解マーカーを除去して選択肢を抽出
        choices = [line.strip() for line in choices_text.split('\n') if line.strip() and line.strip() != '正解']
    else:
        choices = []

    # ドメインを抽出
    domain_match = re.search(r'ドメイン\n(.+?)(?:\n\n|$)', problem_text)
    domain = domain_match.group(1).strip() if domain_match else "ML Workflows"

    # 正解のインデックスを見つける
    correct_index = -1
    for i, choice in enumerate(choices):
        if choice == correct_answer or correct_answer in choice:
            correct_index = i
            break

    # データを構造化
    problem_data = {
        'number': problem_num,
        'question': question,
        'choices': choices,
        'correct_answer': correct_answer,
        'correct_index': correct_index,
        'domain': domain
    }

    output_problems.append(problem_data)
    print(f"問題3-{problem_num}: {len(choices)}個の選択肢, 正解インデックス={correct_index}, ドメイン={domain}")

# JSONとして出力
with open('problems_3_11_to_65.json', 'w', encoding='utf-8') as f:
    json.dump(output_problems, f, ensure_ascii=False, indent=2)

print(f"\n合計 {len(output_problems)} 問を処理しました")
