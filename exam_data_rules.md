# exam_data.js 問題追加ルール

## 概要
このドキュメントは、DBMLA練習問題アプリケーションに新しい問題を追加する際のルールと形式を定義します。

## ファイル構造

```javascript
// DBMLA 試験問題データ

const questions = [
    {
        // 問題オブジェクト
    },
    {
        // 問題オブジェクト
    }
    // ...
];
```

## 問題オブジェクトの必須フィールド

各問題オブジェクトは以下のフィールドを含む必要があります：

### 1. number (必須)
- **型**: `number`
- **説明**: 問題番号（連番）
- **例**: `1`, `2`, `3`...
- **ルール**:
  - 1から開始
  - 連続した整数
  - 重複不可

### 2. domain (必須)
- **型**: `string`
- **説明**: 問題のドメイン/カテゴリ
- **既存ドメイン**:
  - `"Spark ML Basics"`
  - `"ML workflows"`
  - `"AutoML"`
  - `"Databricks ML"`
  - `"Spark ML"`
  - `"SparkTrials"`
- **ルール**:
  - 新しいドメインを追加する場合は、`index.html`のドメインフィルタにも追加すること
  - 統一性を保つため、既存ドメインを優先的に使用

### 3. question (必須)
- **型**: `string`
- **説明**: 問題文
- **例**: `"Spark MLを使用したデータサイエンスプロジェクトの最初のステップは何ですか？"`
- **ルール**:
  - 明確で簡潔な日本語
  - 質問形式（「〜ですか？」など）
  - HTML タグは使用しない（プレーンテキスト）

### 4. keyPoint (必須)
- **型**: `string`
- **説明**: 問題の要点（30文字程度の要約）
- **例**: `"Spark環境のセットアップが全ての作業の前提条件"`
- **ルール**:
  - 30文字前後（厳密でなくてもOK）
  - 問題の核心を簡潔に表現
  - 苦手対策シートで使用される

#### keyPoint の記載ガイド（追記）

- 目的: keyPointを読めば「問題だけを見て正答を導ける」レベルのヒントにする。
- 形式: 具体的かつ行動指向（例: 関数名、メソッド名、必須引数、使うべきAPIなどを含める）。
- 長さ: 20～40文字目安（過度に長くならないこと）。
- 文体: 命令形または短い説明文。曖昧な語（ "重要" や "知っておく" のみ）は避ける。
- 例:
  - 良い例: "filter(col('discount') < 0)で該当行を抽出する"
  - 良い例: "VectorAssemblerで入力列をfeaturesベクトルにまとめる"
  - 悪い例: "色々な方法がある"（解答の助けにならない）

注: 新しい問題を追加する際は、keyPointが問題の解答に十分なヒントかをセルフチェックしてください。

### 5. choices (必須)
- **型**: `string[]` (文字列の配列)
- **説明**: 選択肢のリスト
- **例**:
```javascript
choices: [
    "データ前処理",
    "Spark MLライブラリのインポートとセッションの作成",
    "モデルトレーニング",
    "結果の解釈"
]
```
- **ルール**:
  - 配列の長さは通常4-5個
  - 各選択肢は簡潔に
  - HTML タグは使用しない
  - 正解は含めるが、明らかすぎないようにバランスを取る

### 6. correctIndex (必須)
- **型**: `number`
- **説明**: 正解の選択肢のインデックス（0始まり）
- **例**: `1` (配列の2番目の要素が正解)
- **ルール**:
  - 0から始まる
  - choices配列の有効なインデックスであること
  - 範囲: `0 <= correctIndex < choices.length`

### 7. explanation (必須)
- **型**: `string` (HTML文字列)
- **説明**: 正解の解説
- **形式**:
```javascript
explanation: `
    <p>解説の本文...</p>

    <div class="code-block">コード例（オプション）</div>

    <p>追加の説明...</p>

    <h4>他の選択肢が不正解な理由:</h4>
    <ul>
        <li><strong>選択肢A:</strong> 理由...</li>
        <li><strong>選択肢B:</strong> 理由...</li>
    </ul>
`
```
- **ルール**:
  - バッククォート（`` ` ``）で囲む（テンプレートリテラル）
  - HTML タグを使用可能
  - 推奨構造:
    1. 正解の理由説明（`<p>`タグ）
    2. コード例（オプション、`<div class="code-block">`タグ）
    3. 追加説明（`<p>`タグ）
    4. 不正解の理由（`<h4>`と`<ul>`タグ）
  - コードブロックは改行を含む場合、そのまま記述（CSSで自動整形）

### 8. code (オプション)
- **型**: `string`
- **説明**: 問題文に関連するコードスニペット
- **例**:
```javascript
code: `num_evals = 5, trials = SparkTrials()
space=search_space, algo=tpe.suggest, max_evals=num_evals, trials=trials`
```
- **ルール**:
  - 問題文にコードが必要な場合のみ使用
  - バッククォート（`` ` ``）で囲む
  - 改行はそのまま記述（自動整形される）
  - 不要な場合はフィールド自体を省略

## 問題追加の手順

### 1. 既存問題の確認
```javascript
// 最後の問題番号を確認
const lastQuestion = questions[questions.length - 1];
console.log(lastQuestion.number); // 例: 10
```

### 2. 新しい問題を追加
```javascript
const questions = [
    // ...既存の問題...
    {
        number: 11,  // 次の番号
        domain: "Spark ML",
        question: "新しい問題文をここに記述",
        keyPoint: "要点を30文字程度で記述",
        choices: [
            "選択肢1",
            "選択肢2",
            "選択肢3",
            "選択肢4"
        ],
        correctIndex: 1,
        explanation: `
            <p>正解の説明をここに記述</p>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>選択肢1:</strong> 理由1</li>
                <li><strong>選択肢2:</strong> 理由2</li>
                <li><strong>選択肢3:</strong> 理由3</li>
            </ul>
        `
    }
];
```

### 3. ドメインフィルタの更新（新ドメインの場合）
新しいドメインを追加した場合は、`index.html`のドメインフィルタも更新：

```html
<select id="domain-filter" onchange="filterQuestions()">
    <option value="all">すべて</option>
    <option value="Spark ML Basics">Spark ML Basics</option>
    <!-- 既存のドメイン -->
    <option value="新しいドメイン">新しいドメイン</option>
</select>
```

### 4. 動作確認
1. ブラウザで `index.html` を開く
2. 新しい問題が問題一覧に表示されることを確認
3. ドメインフィルタで絞り込めることを確認
4. 問題を解いて正解/不正解が正しく表示されることを確認
5. 解説が適切に表示されることを確認

## チェックリスト

新しい問題を追加する際は、以下を確認してください：

- [ ] `number` は連番で重複していない
- [ ] `domain` は既存のドメインか、または新規追加した場合は `index.html` にも追加済み
- [ ] `question` は明確で簡潔な日本語
- [ ] `keyPoint` は30文字前後で要点を表現
- [ ] `choices` は4-5個の選択肢を含む
- [ ] `correctIndex` は正しいインデックス（0始まり）
- [ ] `explanation` はHTML形式で、正解理由と不正解理由を含む
- [ ] `code` は必要な場合のみ追加（不要なら省略）
- [ ] カンマ（`,`）の位置が正しい（最後の要素の後ろにカンマは不要）
- [ ] JavaScriptの構文エラーがない（ブラウザのコンソールで確認）

## よくある間違い

### 1. カンマの誤り
```javascript
// ❌ 間違い: 最後の要素にカンマ
{
    number: 1,
    domain: "Spark ML",
    explanation: `...`,  // 最後の要素
}  // ← このカンマは不要

// ✅ 正しい
{
    number: 1,
    domain: "Spark ML",
    explanation: `...`  // 最後の要素にはカンマ不要
}
```

### 2. correctIndex の範囲外
```javascript
// ❌ 間違い
choices: ["A", "B", "C"],  // 3つの選択肢
correctIndex: 3  // インデックス3は範囲外（0, 1, 2のみ有効）

// ✅ 正しい
choices: ["A", "B", "C"],
correctIndex: 1  // 0-2の範囲内
```

### 3. HTMLエスケープの忘れ
```javascript
// 説明文中のバッククォートはエスケープ
explanation: `
    <p>コードは \`spark.read\` を使用します</p>
    <!-- ↑ バッククォートをエスケープ -->
`
```

### 4. 改行の誤り
```javascript
// ❌ 間違い: テンプレートリテラル外での改行
explanation: "<p>これは
長い説明です</p>"  // 通常の文字列では改行不可

// ✅ 正しい: テンプレートリテラルを使用
explanation: `<p>これは
長い説明です</p>`
```

## サンプルテンプレート

```javascript
{
    number: XX,
    domain: "ドメイン名",
    question: "問題文をここに記述？",
    keyPoint: "要点を30文字程度で",
    choices: [
        "選択肢1",
        "選択肢2",
        "選択肢3",
        "選択肢4"
    ],
    correctIndex: 0,
    explanation: `
        <p>正解の説明をここに記述します。</p>

        <h4>他の選択肢が不正解な理由:</h4>
        <ul>
            <li><strong>選択肢1:</strong> 理由を説明</li>
            <li><strong>選択肢2:</strong> 理由を説明</li>
            <li><strong>選択肢3:</strong> 理由を説明</li>
        </ul>
    `
}
```

## 参考リンク

- [JavaScript テンプレートリテラル](https://developer.mozilla.org/ja/docs/Web/JavaScript/Reference/Template_literals)
- [HTML 基本タグ](https://developer.mozilla.org/ja/docs/Web/HTML)

## 更新履歴

- 2025-11-01: 初版作成
