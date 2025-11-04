// DBMLA 試験問題データ

const questions = [
    {
        number: 1,
        domain: "Spark ML Basics",
        question: "Spark MLを使用したデータサイエンスプロジェクトの最初のステップは何ですか？",
    keyPoint: "SparkSessionを作成しSpark MLの操作が可能な状態にする",
        choices: [
            "データ前処理",
            "Spark MLライブラリのインポートとセッションの作成",
            "モデルトレーニング",
            "結果の解釈"
        ],
        correctIndex: 1,
        explanation: `
            <p>Spark MLを使用したデータサイエンスプロジェクトの最初のステップは、必要なSpark MLライブラリをインポートし、Sparkセッションを作成することです。これは、データ前処理、モデルトレーニング、結果の解釈などの操作を実行する前に、Spark環境をセットアップする必要があるためです。</p>

            <div class="code-block">from pyspark.sql import SparkSession

# Sparkセッションを作成
spark = SparkSession.builder \\
    .appName("MySparkMLProject") \\
    .getOrCreate()</div>

            <p>Sparkセッションを初期化しないと、Sparkの分散コンピューティング機能やMLライブラリを使用できません。</p>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>データ前処理:</strong> データ前処理は重要なステップですが、Sparkセッションをセットアップした後に行われます。</li>
                <li><strong>モデルトレーニング:</strong> モデルトレーニングはデータ前処理と特徴量エンジニアリングの後に実行されます。</li>
                <li><strong>結果の解釈:</strong> 結果の解釈は、モデルトレーニングと評価の後の最終ステップです。</li>
            </ul>
        `
    },
    {
        number: 2,
        domain: "ML workflows",
        question: "データサイエンティストが機械学習モデルのハイパーパラメータを調整するために以下のコードセグメントを使用しています。より精度の高いモデルを得る可能性を高めるために、どのような変更を適用できますか？",
    keyPoint: "num_evalsを増やすと探索回数が増え最適解が見つかりやすい",
        code: `num_evals = 5, trials = SparkTrials()
space=search_space, algo=tpe.suggest, max_evals=num_evals, trials=trials`,
        choices: [
            "tpe.suggestをrandom.suggestに置き換える",
            "num_evalsを50に増やす",
            "algo=tpe.suggest引数を省略する",
            "fmin()をfmax()に置き換える",
            "SparkTrials()をTrials()に切り替える"
        ],
        correctIndex: 1,
        explanation: `
            <p>パラメータ<code>num_evals</code>は、最適化プロセス中に評価されるハイパーパラメータの組み合わせの数を制御します。<code>num_evals</code>を5から50に増やすことで、アルゴリズムがハイパーパラメータ空間のより大きな部分を探索でき、より最適なハイパーパラメータのセットを見つける可能性が高まります。</p>

            <div class="code-block"># num_evalsを50に増やす
num_evals = 50
trials = SparkTrials()

best_hyperparams = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=num_evals,
    trials=trials
)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>tpe.suggestをrandom.suggestに置き換える:</strong> TPEはより効率的でインテリジェントな探索アルゴリズムです。</li>
                <li><strong>algo引数を省略:</strong> 関数が失敗するか、効率の悪い探索方法にデフォルト設定されます。</li>
                <li><strong>fmin()をfmax()に置き換える:</strong> fmax()はHyperoptライブラリの有効な関数ではありません。</li>
            </ul>
        `
    },
    {
        number: 3,
        domain: "AutoML",
        question: "Databricks AutoMLで、すべてのモデルイテレーションにわたって最良のモデルコードに移動するにはどうすればよいですか？",
    keyPoint: "AutoMLは最良モデルのノートブックを生成する（'View notebook for best model' を開く）",
        choices: [
            "AutoML実験実行後に「View Best Model」リンクをクリックする",
            "AutoML実験実行後に「View notebook for best model」リンクをクリックする",
            "AutoML実験実行後に「Get Best Model」リンクをクリックする",
            "AutoML実験実行後に「Top Model」リンクをクリックする"
        ],
        correctIndex: 1,
        explanation: `
            <p>Databricks AutoMLでは、AutoML実験が完了すると、Databricksは最良のモデルのノートブックを生成します。このノートブックには、トレーニングコード、前処理ステップ、評価メトリクスが含まれています。</p>

            <h4>アクセス方法:</h4>
            <ol>
                <li>DatabricksのAutoML実験ページに移動する</li>
                <li>UIで「View notebook for best model」リンクを探す</li>
                <li>それをクリックして、最良のモデルのコードと構成を含むノートブックを開く</li>
            </ol>

            <div class="code-block">import databricks.automl

summary = databricks.automl.classify(df, target_col="label", timeout_minutes=20)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>「View Best Model」:</strong> このオプションは存在しません。</li>
                <li><strong>「Get Best Model」:</strong> このオプションもDatabricks AutoML UIにはありません。</li>
                <li><strong>「Top Model」:</strong> このリンクもAutoML UIには存在しません。</li>
            </ul>
        `
    },
    {
        number: 4,
        domain: "AutoML",
        question: "AutoMLを使用した予測の文脈において、horizonパラメータは何を表しますか？",
    keyPoint: "horizonは予測すべき将来の期間（ステップ）数を指定する",
        choices: [
            "時系列の頻度",
            "AutoMLトライアルの最大待機時間",
            "予測を返すべき将来の期間の数",
            "予測のための時間列"
        ],
        correctIndex: 2,
        explanation: `
            <p><code>horizon</code>パラメータは、モデルが予測を生成すべき将来の時間期間の数を表します。例えば、日次売上を予測していて<code>horizon</code>を7に設定した場合、モデルは今後7日間の売上を予測します。</p>

            <div class="code-block">from databricks import automl

summary = automl.forecast(
    dataset=df,
    time_col="date",
    target_col="sales",
    horizon=7,  # 次の7期間を予測
    frequency="D"  # 日次頻度
)</div>

            <h4>他のパラメータとの違い:</h4>
            <ul>
                <li><strong>frequency:</strong> 時系列の頻度（例：daily, weekly）</li>
                <li><strong>timeout:</strong> AutoMLトライアルの最大待機時間</li>
                <li><strong>time_col:</strong> 予測のための時間列</li>
            </ul>
        `
    },
    {
        number: 5,
        domain: "Databricks ML",
        question: "データサイエンティストがローカルでGitリポジトリに変更をプッシュした後、Databricksに読み込むにはどうすればよいですか？",
    keyPoint: "変更を反映するにはDatabricks Reposで手動Pullを実行する",
        choices: [
            "Repo Gitダイアログを開き、自動同期を有効にする",
            "Repo Gitダイアログを開き、「Sync」ボタンをクリックする",
            "Repo Gitダイアログを開き、「Merge」ボタンをクリックする",
            "Repo Gitダイアログを開き、自動プルを有効にする",
            "Repo Gitダイアログを開き、「Pull」ボタンをクリックする"
        ],
        correctIndex: 4,
        explanation: `
            <p>Databricks Reposを使用している場合、オンラインGitリポジトリに新しい変更をプッシュしても、それらの変更はDatabricksワークスペースに自動的に更新されません。ユーザーは最新の変更を手動でプルして、ワークスペースを同期する必要があります。</p>

            <h4>手順:</h4>
            <ol>
                <li>Databricksを開き、Reposに移動する</li>
                <li>更新が必要なリポジトリをクリックする</li>
                <li>Repo Gitダイアログを開く</li>
                <li>「Pull」ボタンをクリックする</li>
            </ol>

            <div class="code-block">databricks repos update --path /Repos/my-repo --branch main</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>自動同期:</strong> Databricksは自動同期をサポートしていません。</li>
                <li><strong>「Sync」ボタン:</strong> このボタンは存在しません。</li>
                <li><strong>「Merge」ボタン:</strong> マージは競合解決用で、プルではありません。</li>
            </ul>
        `
    },
    {
        number: 6,
        domain: "ML workflows",
        question: "以下のスキーマを持つ特徴量セットで、どの列を最頻値を使用して補完すべきですか？\n\ncustomer_id STRING, spend DOUBLE, units INTEGER, happiness_tier STRING",
    keyPoint: "カテゴリ列は最頻値で補完、数値列は平均/中央値で補完する",
        choices: [
            "units",
            "customer_id",
            "happiness_tier",
            "spend",
            "customer_id, happiness_tier"
        ],
        correctIndex: 2,
        explanation: `
            <p><code>happiness_tier</code>列はカテゴリカル特徴量（文字列型）であり、最頻値（モード）がカテゴリカルデータの適切な補完戦略です。</p>

            <div class="code-block">from pyspark.sql.functions import mode

# happiness_tierを最頻値で補完
most_common_tier = df.select(mode("happiness_tier")).collect()[0][0]
df = df.fillna({"happiness_tier": most_common_tier})</div>

            <h4>他の列の補完方法:</h4>
            <ul>
                <li><strong>units (数値):</strong> 平均値、中央値、または定数値を使用</li>
                <li><strong>customer_id (主キー):</strong> 補完すべきではない（一意の識別子）</li>
                <li><strong>spend (数値):</strong> 平均値、中央値、または定数値を使用</li>
            </ul>

            <h4>まとめ:</h4>
            <ul>
                <li>カテゴリカル特徴量 → 最頻値（モード）</li>
                <li>数値特徴量 → 平均値、中央値、定数</li>
                <li>一意の識別子 → 補完しない</li>
            </ul>
        `
    },
    {
        number: 7,
        domain: "Spark ML",
        question: "Spark DataFrameに適用されたときに、Pandas UDF関数内でpandas API構文が互換性を持つ理由は何ですか？",
    keyPoint: "Apache ArrowでJVM⇄Pythonの高速共有が可能になりpandas APIが利用できる",
        choices: [
            "Pandas UDFが内部的にPandas Function APIを呼び出す",
            "Pandas UDFが関数内でpandas API on Sparkを利用する",
            "pandas API構文はSpark DataFrame上のPandas UDF関数内では実装できない",
            "Pandas UDFが自動的に関数をSpark DataFrame構文に変換する",
            "Pandas UDFがApache Arrowを活用してSparkとpandas形式間でデータを変換する"
        ],
        correctIndex: 4,
        explanation: `
            <p>SparkのPandas UDFは、Apache Arrowを使用してSparkとpandas形式間でデータを効率的に変換します。</p>

            <h4>動作の仕組み:</h4>
            <ol>
                <li>Sparkがデータをパーティション化し、各パーティションをワーカーノードに送信</li>
                <li>Apache ArrowがSpark DataFrameパーティションをpandas DataFrameに変換</li>
                <li>Pandas UDFが各pandas DataFrameにpandas API操作を適用</li>
                <li>結果がArrowを使用してSpark形式に戻され、Spark DataFrameに結合</li>
            </ol>

            <div class="code-block">from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf("double")
def squared(x: pd.Series) -> pd.Series:
    return x ** 2

df.withColumn("squared_value", squared("value")).show()</div>

            <h4>重要なポイント:</h4>
            <p>Pandas UDFは、pandas API構文の使いやすさとSparkのスケーラビリティを組み合わせることができ、Apache Arrowによって実現されています。</p>
        `
    },
    {
        number: 8,
        domain: "SparkTrials",
        question: "HyperoptとSparkTrialsで小さなデータセット（約10MB以下）を扱う場合、データセットをロードするための推奨アプローチは何ですか？",
    keyPoint: "小規模データはドライバーにロードして目的関数から直接使う",
        choices: [
            "Sparkを使用してデータセットを明示的にブロードキャストする",
            "ドライバーでデータセットをロードし、目的関数から直接呼び出す",
            "Databricks Runtime 6.4 ML以上を使用する",
            "データセットをDBFSに保存してワーカーにロードし直す"
        ],
        correctIndex: 1,
        explanation: `
            <p>小さなデータセット（約10MB以下）の場合、「ドライバーでデータセットをロードし、目的関数から直接呼び出す」が推奨されます。</p>

            <h4>推奨される理由:</h4>
            <ul>
                <li><strong>オーバーヘッドの削減:</strong> ブロードキャストやDBFSからのロードには追加オーバーヘッドが発生</li>
                <li><strong>メモリ効率:</strong> 小さなデータセットはドライバーのメモリに効率的にロード可能</li>
                <li><strong>高速アクセス:</strong> ドライバーから直接アクセスする方が高速</li>
                <li><strong>コードの簡素化:</strong> 明示的なブロードキャストが不要</li>
            </ul>

            <div class="code-block">import pandas as pd
from hyperopt import fmin, tpe, hp

# ドライバーでデータセットをロード
data = pd.read_csv("my_small_dataset.csv")

def objective(args):
    # ドライバーから直接データにアクセス
    model = train_model(data, args)
    return evaluate_model(model)</div>

            <p><strong>注意:</strong> 大きなデータセットの場合は、ブロードキャストやDBFSからのロードが必要になる場合があります。</p>
        `
    },
    {
        number: 9,
        domain: "ML workflows",
        question: "ハイパーパラメータ最適化のために、以前のトライアルの結果に基づいて教育的選択を行う戦術を採用しているのはどれですか？",
    keyPoint: "TPEは過去の試行結果に基づき次の候補を選ぶベイズ手法",
        choices: [
            "Grid Search最適化",
            "Random Search最適化",
            "Halving Random Search最適化",
            "Manual Search最適化",
            "Tree of Parzen Estimators最適化"
        ],
        correctIndex: 4,
        explanation: `
            <p>Tree of Parzen Estimators（TPE）最適化法は、以前のトライアルの結果に基づいてハイパーパラメータ値を教育的に選択するベイズ最適化手法です。</p>

            <div class="code-block">from hyperopt import fmin, tpe, hp, Trials

space = {
    'learning_rate': hp.loguniform('learning_rate', -5, 0),
    'max_depth': hp.choice('max_depth', range(1, 10))
}

best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,  # TPE最適化
    max_evals=50,
    trials=Trials()
)</div>

            <h4>他の手法との違い:</h4>
            <ul>
                <li><strong>Grid Search:</strong> すべての組み合わせを評価（学習なし）</li>
                <li><strong>Random Search:</strong> ランダムに選択（学習なし）</li>
                <li><strong>Halving Random Search:</strong> 早期排除を使用（確率モデルなし）</li>
                <li><strong>Manual Search:</strong> 手動選択（体系的学習なし）</li>
            </ul>

            <p><strong>重要:</strong> TPEは確率モデルを使用して、以前の結果から学習し、次の試行を最適化します。</p>
        `
    },
    {
        number: 10,
        domain: "ML workflows",
        question: "ツリーベースモデルでOne-Hot Encoding（OHE）を使用する場合、どのような問題が発生する可能性がありますか？",
    keyPoint: "OHEは高カーディナリティでスパース化し、ツリーモデルに不利になる",
        choices: [
            "データセットにスパース性を引き起こす",
            "どのオプションでもない",
            "カテゴリカル変数の分割オプションの数を制限する",
            "両方"
        ],
        correctIndex: 3,
        explanation: `
            <p>One-Hot Encoding（OHE）は、ツリーベースモデルで<strong>2つの主要な問題</strong>を引き起こします。</p>

            <h4>1. データセットにスパース性を引き起こす</h4>
            <ul>
                <li>高カーディナリティの場合、多数の列を作成</li>
                <li>スパース行列が生成され、メモリ使用量と計算コストが増加</li>
                <li>ツリーベースモデルはスパース行列を効率的に処理できない</li>
            </ul>

            <div class="code-block">from sklearn.preprocessing import OneHotEncoder

# 500の一意の値がある場合、500の特徴量を作成
encoder = OneHotEncoder(sparse_output=False)
encoded_df = encoder.fit_transform(df[['city']])</div>

            <h4>2. 分割オプションの数を制限する</h4>
            <ul>
                <li>カテゴリカル特徴量は単一変数として扱うべき</li>
                <li>OHEは個別のバイナリ特徴量で分割を強制</li>
                <li>最適ではない分割につながる</li>
            </ul>

            <h4>より良い代替案:</h4>
            <div class="code-block">from lightgbm import LGBMClassifier

# LightGBMやCatBoostはネイティブなカテゴリカルエンコーディングをサポート
model = LGBMClassifier()
model.fit(X_train, y_train, categorical_feature=['city'])</div>
        `
    },
    {
        number: 11,
        domain: "Spark ML",
        question: "以下のうち、分散機械学習フレームワークの例はどれですか？",
    keyPoint: "Apache Spark MLlibはクラスタ上で動く分散機械学習フレームワーク",
        choices: [
            "TensorFlow",
            "Apache Spark MLlib",
            "Scikit-learn",
            "XGBoost",
            "上記すべて"
        ],
        correctIndex: 1,
        explanation: `
            <p>Apache Spark MLlibは、コンピュータクラスタ全体で大規模なデータ処理と機械学習タスクを処理するように設計された分散機械学習フレームワークです。Apache Spark上に構築されており、分散コンピューティング機能を提供するため、ビッグデータアプリケーションに適しています。</p>

            <h4>Spark MLlibの主な特徴:</h4>
            <ul>
                <li>分散データ処理と機械学習</li>
                <li>大規模データセットを処理するためのスケーラビリティ</li>
                <li>Sparkエコシステムとの統合（例：Spark SQL、Spark Streaming）</li>
            </ul>

            <div class="code-block">from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

# Sparkセッションを初期化
spark = SparkSession.builder.appName("MLlibExample").getOrCreate()

# データセットをロード
data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

# ロジスティック回帰モデルをトレーニング
lr = LogisticRegression(maxIter=10, regParam=0.01)
model = lr.fit(data)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>TensorFlow:</strong> TensorFlowは分散可能なディープラーニングフレームワークですが、Spark MLlibのような汎用的な分散機械学習用に設計されていません。</li>
                <li><strong>Scikit-learn:</strong> Scikit-learnは人気のある機械学習ライブラリですが、分散されていません。シングルノード実行用に設計されており、クラスタ全体での大規模データ処理はできません。</li>
                <li><strong>XGBoost:</strong> XGBoostはスケーラブルで効率的な勾配ブースティングの実装ですが、本質的に分散されていません。大規模データセットを効率的に処理できますが、Spark MLlibと同じ分散コンピューティング機能は提供していません。</li>
                <li><strong>上記すべて:</strong> Apache Spark MLlibのみが分散機械学習フレームワークであるため、これは不正解です。</li>
            </ul>
        `
    },
    {
        number: 12,
        domain: "ML workflows",
        question: "包括的なデータ分析のためにSpark DataFrameの要約統計を取得するにはどうすればよいですか？",
    keyPoint: "describe()/summary()/dbutils.data.summarize()は要約統計を取得する方法",
        choices: [
            "spark_dataframe.describe()",
            "spark_dataframe.summary()",
            "dbutils.data.summarize(spark_dataframe)",
            "上記すべて"
        ],
        correctIndex: 3,
        explanation: `
            <p>Apache Sparkには、包括的なデータ分析のためにSpark DataFrameの要約統計を取得する複数の方法があります。各メソッドには独自のユースケースがあります。</p>

            <h4>1. spark_dataframe.describe()</h4>
            <ul>
                <li>数値列のみの基本的な統計要約（count、mean、stddev、min、max）を提供</li>
                <li>パーセンタイル（中央値、四分位数など）は含まれない</li>
            </ul>

            <h4>2. spark_dataframe.summary()</h4>
            <ul>
                <li>describe()よりも詳細な要約を提供（25%、50%、75%のパーセンタイルを含む）</li>
                <li>数値列とカテゴリカル列の両方で動作</li>
            </ul>

            <h4>3. dbutils.data.summarize(spark_dataframe)（Databricks固有）</h4>
            <ul>
                <li>探索的データ分析（EDA）のためのDatabricksユーティリティ</li>
                <li>Databricksノートブックでインタラクティブな要約統計を提供</li>
                <li>視覚化、分布、データインサイトをサポート</li>
            </ul>

            <p>すべてのオプションが異なるレベルの要約統計を提供するため、最適な答えは「上記すべて」です。</p>
        `
    },
    {
        number: 13,
        domain: "ML workflows",
        question: "機械学習エンジニアがMLパイプラインをスケーリングしようとしています。トレーニングデータ全体を各コアにブロードキャストした後、各コアは一度に1つのモデルをトレーニングできます。チューニングプロセスがまだ遅いため、並列度を4コアから8コアに増やす予定です。ただし、クラスタの総メモリは増やせません。どの条件下で並列度を4から8に上げるとチューニングプロセスが高速化されますか？",
    keyPoint: "全データが各コアに収まる場合のみ、並列度増加で高速化する",
        choices: [
            "データが縦長の形状の場合",
            "データが横長の形状の場合",
            "モデルを並列化できない場合",
            "チューニングプロセスがランダム化されている場合",
            "全データが各コアに収まる場合"
        ],
        correctIndex: 4,
        explanation: `
            <p>分散ML訓練では、各コアが割り当てられたワークロードを処理するのに十分なメモリを持っている場合にのみ、並列処理が効率的に機能します。クラスタの総メモリを増やせないため、コアを増やす（4から8へ）ことでパフォーマンスが向上するのは、各コアがフルデータセットを保存・処理するのに十分なメモリを持っている場合のみです。</p>

            <h4>重要なポイント:</h4>
            <ul>
                <li>データセットが各コアに収まる場合、各コアは独立して別々のモデルをトレーニングできる</li>
                <li>データセットがコアあたりの利用可能メモリを超える場合、スワッピング、ディスクI/O、メモリオーバーフローがトレーニングを遅くする</li>
            </ul>

            <div class="code-block">from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("ParallelTuning").getOrCreate()
df = spark.read.csv("data.csv", header=True, inferSchema=True)

lr = LogisticRegression(featuresCol="features", labelCol="label")
paramGrid = ParamGridBuilder() \\
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \\
    .build()

crossval = CrossValidator(
    estimator=lr,
    estimatorParamMaps=paramGrid,
    evaluator=BinaryClassificationEvaluator(),
    numFolds=3,
    parallelism=8  # 並列度を8に設定
)

model = crossval.fit(df)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>データが縦長の形状の場合:</strong> 長い/狭いデータセット（多くの行、少ない列）は必ずしもボトルネックではありません。実際の問題は、各コアがデータセットを処理するのに十分なメモリを持っているかどうかです。</li>
                <li><strong>データが横長の形状の場合:</strong> 幅広いデータセット（多くの列、少ない行）は、コアあたりより多くのメモリを必要とします。メモリが限られている場合、並列度を増やすとメモリオーバーヘッドのためにパフォーマンスが悪化する可能性があります。</li>
                <li><strong>モデルを並列化できない場合:</strong> モデルを並列化できない場合、コアを追加してもパフォーマンスは向上しません。</li>
                <li><strong>チューニングプロセスがランダム化されている場合:</strong> ランダム化されたチューニング（例：ランダム検索）は、コアを追加することでより良いパフォーマンスを保証しません。</li>
            </ul>
        `
    },
    {
        number: 14,
        domain: "Scaling ML Models",
        question: "データサイエンティストがSpark DataFrame 'spark_df'を扱っています。'discount'列の値が0未満の行のみを保持する新しいSpark DataFrameを生成したいと考えています。この目的を達成するコードセグメントはどれですか？",
    keyPoint: "filter(col('discount') < 0)を使ってdiscount<0の行を抽出する",
        choices: [
            "spark_df.find(spark_df(\"discount\") < 0)",
            "spark_df.loc[spark_df(\"discount\") < 0]",
            "spark_df.loc[spark_df(\"discount\") < 0,:]",
            "SELECT * FROM spark_df WHERE discount < 0",
            "spark_df.filter(col(\"discount\") < 0)"
        ],
        correctIndex: 4,
        explanation: `
            <p>Spark DataFrameで条件に基づいて行をフィルタリングする正しい方法は、filter()メソッド（またはそのエイリアスwhere()）を使用することです。filter()メソッドは列式を入力として受け取り、col("discount") < 0が条件を指定する正しい方法です。</p>

            <div class="code-block">from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("FilterExample").getOrCreate()

# サンプルデータ
data = [(1, 0.1), (2, -0.5), (3, 0.0), (4, -0.2)]
columns = ["id", "discount"]
spark_df = spark.createDataFrame(data, columns)

# discount < 0の行をフィルタリング
filtered_df = spark_df.filter(col("discount") < 0)
filtered_df.show()</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>spark_df.find():</strong> find()メソッドはSpark DataFrame APIに存在しません。このコードはエラーになります。</li>
                <li><strong>spark_df.loc[]:</strong> locアクセサはpandasで使用され、Spark DataFramesでは使用されません。このコードはエラーになります。</li>
                <li><strong>spark_df.loc[,:]:</strong> 上記と同様に、locはSpark DataFramesには適用されません。</li>
                <li><strong>SELECT * FROM spark_df WHERE discount < 0:</strong> これはSQL構文であり、DataFrameを一時ビューとして登録してSpark SQLを使用しない限り、Spark DataFrameで直接使用できません。ただし、これはSpark DataFrameで行をフィルタリングする最も直接的な方法ではありません。</li>
            </ul>
        `
    },
    {
        number: 15,
        domain: "SparkTrials",
        question: "データサイエンティストがscikit-learnモデルのハイパーパラメータを並行して効率的に調整しようとしています。Hyperoptライブラリを活用することにしました。Hyperoptライブラリ内のどのツールがハイパーパラメータを並列で最適化する機能を提供しますか？",
    keyPoint: "SparkTrialsをtrialsに渡すとSpark上で並列最適化が可能",
        choices: [
            "fmin",
            "Search Space",
            "hp.quniform",
            "SparkTrials",
            "Trials"
        ],
        correctIndex: 3,
        explanation: `
            <p>HyperoptライブラリのSparkTrialsクラスは、Apache Sparkを使用した分散コンピューティング環境で並列ハイパーパラメータ最適化を可能にするために特別に設計されています。これにより、ハイパーパラメータの組み合わせの評価をクラスタ全体に分散させることができ、チューニングプロセスを大幅に高速化できます。</p>

            <h4>動作方法:</h4>
            <ol>
                <li>検索空間と目的関数を定義</li>
                <li>SparkTrialsを使用してハイパーパラメータの組み合わせの評価を並列化</li>
                <li>SparkTrialsをfmin関数に渡して分散最適化を実行</li>
            </ol>

            <div class="code-block">from hyperopt import fmin, tpe, hp, SparkTrials

# 目的関数を定義
def objective(params):
    # モデルをトレーニングして損失を返す
    return loss

# 検索空間を定義
space = {
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
    'max_depth': hp.quniform('max_depth', 3, 10, 1)
}

# 並列最適化にSparkTrialsを使用
spark_trials = SparkTrials()

# ハイパーパラメータ最適化を実行
best_hyperparams = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=spark_trials
)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>fmin:</strong> fmin関数はハイパーパラメータ最適化を実行するために使用されますが、本質的に並列実行を可能にするわけではありません。並列性を有効にするには、trials引数（SparkTrialsなど）に依存します。</li>
                <li><strong>Search Space:</strong> 検索空間は探索するハイパーパラメータの範囲を定義しますが、並列実行を制御しません。</li>
                <li><strong>hp.quniform:</strong> これは検索空間で量子化された一様分布を定義するために使用される関数です。並列実行を有効にしません。</li>
                <li><strong>Trials:</strong> Trialsクラスは単一マシンでの最適化に使用されますが、並列実行は提供しません。</li>
            </ul>
        `
    }
    ,
    {
        number: 16,
        domain: "Feature Store",
        question: "Feature Storeテーブルを'feature_set'で作成・登録するためにFeature Store Client 'feature_client'を使用する正しいコードスニペットはどれですか？",
    keyPoint: "create_table(name, primary_keys, df=feature_set, schema=...)で登録する",
        choices: [
            "feature_client.create_table(name='new_feature_table', primary_keys='customer_id', df=feature_set, schema=feature_set.schema, description='Client features')",
            "feature_client.create_table(name='new_feature_table', primary_keys='customer_id', schema=feature_set.schema, description='Client features')",
            "feature_client.create_table(function='generate_features', schema=feature_set.schema, description='Client features')",
            "feature_set.write.mode('feature').saveAsTable('new_feature_table')"
        ],
        correctIndex: 0,
        explanation: `
            <p>Feature Store Clientの<code>create_table</code>メソッドに<code>df</code>引数としてSpark DataFrameを渡すことで、テーブルを作成し同時にデータを登録できます。必要な引数には、テーブル名（name）、主キー（primary_keys）、データフレーム（df）、スキーマ（schema）、説明（description）などがあります。</p>

            <div class="code-block"># 正しい例
feature_client.create_table(
    name='new_feature_table',
    primary_keys='customer_id',
    df=feature_set,
    schema=feature_set.schema,
    description='Client features'
)
</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>dfを指定しないcreate_table:</strong> <code>df</code>がないとテーブルは作成されてもデータは登録されません。</li>
                <li><strong>function引数を渡す形式:</strong> function引数はテーブル作成用の正しい署名ではありません。</li>
                <li><strong>feature_set.write...を使う形式:</strong> Feature Store用に特化した書き込みではなく、Feature Store Client APIを使うのが推奨されます。</li>
            </ul>
        `
    },
    {
        number: 17,
        domain: "Spark ML",
        question: "Pandasで書かれた特徴量エンジニアリングのノートブックを、大規模データに対して最小限のリファクタでスケールさせたいときに最適なツールはどれですか？",
    keyPoint: "pandas API on SparkでPandasコードを分散実行できる",
        choices: [
            "Feature Store",
            "PySpark DataFrame API",
            "Spark SQL",
            "Scala Dataset API",
            "pandas API on Spark"
        ],
        correctIndex: 4,
        explanation: `
            <p>Pandasのコードを大きなデータセットに対応させる最短ルートは、pandas API on Spark（旧Koalas）を使うことです。既存のPandasコードをほとんど書き換えずにSpark上で分散実行できるため、最小限のリファクタでスケールできます。</p>

            <div class="code-block">import pyspark.pandas as ps

# pandasライクなAPIで大規模データを処理
psdf = ps.read_csv("large_dataset.csv")
psdf['new_col'] = psdf['existing_col'] * 2
spark_df = psdf.to_spark()</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>Feature Store:</strong> フィーチャ管理用で直接のリファクタ削減ツールではない</li>
                <li><strong>PySpark DataFrame API:</strong> 効率的だが大幅な書き換えが必要</li>
                <li><strong>Spark SQL / Scala Dataset API:</strong> 言語やAPIの変更が必要でリファクタ工数が大きい</li>
            </ul>
        `
    },
    {
        number: 18,
        domain: "Pandas API on Spark",
        question: "pandas-on-Spark DataFrameをPySpark DataFrameに変換する際、特定列の型変換をastypeで行うにはどうすればよいですか？",
    keyPoint: "psdf['列名'].astype('型')で特定列をキャストする",
        choices: [
            "pandas-on-Sparkでは型変換はサポートされない",
            "DataFrame全体に対してastypeを呼び出す必要がある",
            "型変換は自動的に行われる",
            "psdf['column_name'].astype('desired_type')で特定列をcastする"
        ],
        correctIndex: 3,
        explanation: `
            <p>pandas-on-Spark（pyspark.pandas）では、特定の列に対して<code>astype</code>を使って型変換が可能です。必要な列に対して個別に<code>psdf['col'] = psdf['col'].astype('int')</code>のように適用します。</p>

            <div class="code-block">import pyspark.pandas as ps

pandas_df = ps.DataFrame({'A': [1.0, 2.0], 'B': ['4.4', '5.5']})
pandas_df['A'] = pandas_df['A'].astype('int')
pandas_df['B'] = pandas_df['B'].astype('float')</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>サポートされない:</strong> 実際にはサポートされている</li>
                <li><strong>全体に適用:</strong> 列単位で指定するのが一般的で効率的</li>
                <li><strong>自動変換:</strong> 明示的な変換が必要な場合が多い</li>
            </ul>
        `
    },
    {
        number: 19,
        domain: "Spark ML",
        question: "DatabricksのmapInPandas()の主なユースケースは何ですか？",
    keyPoint: "mapInPandasは各パーティションに関数を適用するために使う",
        choices: [
            "複数モデルを並列で実行する",
            "DataFrameの各パーティションに関数を適用する",
            "グループ化したデータに関数を適用する",
            "2つのDataFrameのco-groupedデータに関数を適用する"
        ],
        correctIndex: 1,
        explanation: `
            <p><code>mapInPandas()</code>は、Spark DataFrameの各パーティションに対してPython関数（pandas DataFrameを受け取る）を適用するためのAPIです。パーティション単位でpandas操作を適用し、複雑な変換を分散実行できます。</p>

            <div class="code-block">from pyspark.sql import SparkSession
import pandas as pd

def transform_partition(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf['value'] = pdf['value'] * 2
    return pdf

result_df = spark_df.mapInPandas(transform_partition, schema=spark_df.schema)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>複数モデルの並列実行:</strong> 直接の目的ではない</li>
                <li><strong>グループ化:</strong> groupBy後のapplyInPandasが適切</li>
                <li><strong>co-grouped:</strong> cogroupやjoinが適切</li>
            </ul>
        `
    },
    {
        number: 20,
        domain: "Feature Store",
        question: "ローカルPython環境にdatabricks-feature-storeクライアントをインストールするにはどうすればよいですか？",
    keyPoint: "pip install databricks-feature-storeでインストールする",
        choices: [
            "pip install databricks-feature-store",
            "%pip install databricks-feature-store",
            "conda install databricks-feature-store",
            "spark install databricks-feature-store"
        ],
        correctIndex: 0,
        explanation: `
            <p>databricks-feature-storeクライアントは通常PyPIから入手でき、ローカル環境では<code>pip install databricks-feature-store</code>を使ってインストールします。ノートブック内では<code>%pip</code>マジックが使えますが、ローカル環境の一般的な手順はpipです。</p>

            <div class="code-block"># ローカル環境でのインストール
pip install databricks-feature-store</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>%pip:</strong> ノートブックマジックであり、通常のシェルコマンドではない</li>
                <li><strong>conda:</strong> パッケージがcondaで提供されていない場合が多い</li>
                <li><strong>spark install:</strong> そのようなコマンドは存在しない</li>
            </ul>
        `
    },
    {
        number: 21,
        domain: "Pandas API on Spark",
        question: "3つのモデルを使う新しいソリューションが、以前の単一モデルソリューションより推論時に非効率になるのはどのような場合ですか？",
    keyPoint: "各レコードで全モデルが予測を行うと推論が遅くなる",
        choices: [
            "新しいソリューションのモデルの平均サイズが元のモデルより大きい場合",
            "各レコードに対して新しい各モデルが予測を計算する必要がある場合",
            "新しいモデルの平均レイテンシが元のモデルより大きい場合",
            "予測ごとにどのモデルを使うかをif-elseで決める場合"
        ],
        correctIndex: 1,
        explanation: `
            <p>各レコードで3つのモデルすべてが予測を計算する必要がある場合、各入力に対して複数回の推論処理が発生するため、総合的な推論時間が増加します。個々のモデルレイテンシが同等でも、モデル数分だけ遅くなります。</p>

            <div class="code-block">def ensemble_predict(input):
    pred1 = model1.predict(input)
    pred2 = model2.predict(input)
    pred3 = model3.predict(input)
    return (pred1 + pred2 + pred3) / 3
</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>モデルサイズ:</strong> ストレージやメモリには影響するが、必ずしも推論時間を決定しない</li>
                <li><strong>平均レイテンシが大きい:</strong> 問題文ではレイテンシは同等であるとされている</li>
                <li><strong>if-elseで選択する:</strong> 条件選択なら1モデルのみ実行されるため効率的</li>
            </ul>
        `
    }
    ,
    {
        number: 22,
        domain: "Spark ML",
        question: "Pandas API on SparkでApache Arrowがある主な利点は何ですか？",
    keyPoint: "ArrowはJVMとPython間の高速なデータ転送を可能にする",
        choices: [
            "ArrowはJVMとPythonプロセス間の効率的なデータ転送を可能にする",
            "ArrowはSpark SQLクエリを自動的に最適化する",
            "Arrowは非カラム型データフォーマットの使用を可能にする",
            "ArrowはDataFrame間の結合を高速化する"
        ],
        correctIndex: 0,
        explanation: `
            <p>Apache Arrowは、pandas API on Spark（pyspark.pandas）で重要な役割を果たし、JVM（Sparkエンジン）とPythonプロセス間でゼロコピーに近い高速なデータ交換を可能にします。これによりシリアライズ/デシリアライズのオーバーヘッドを削減できます。</p>

            <div class="code-block">from pyspark.sql import SparkSession
# Arrowを有効にする例
spark = SparkSession.builder.config("spark.sql.execution.arrow.pyspark.enabled", "true").getOrCreate()</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>Spark SQLの自動最適化:</strong> これはCatalystが担当し、Arrowはデータ転送に特化している</li>
                <li><strong>非カラム型フォーマットの使用:</strong> Arrowはカラムナフォーマットであり、行指向ではない</li>
                <li><strong>結合の高速化:</strong> 結合処理はSparkのオプティマイザが担う</li>
            </ul>
        `
    },
    {
        number: 23,
        domain: "Hyperopt",
        question: "Hyperoptでhp.choice()を使用した場合、Hyperoptは何を返し、実際のパラメータ値はどのように取得しますか？",
    keyPoint: "hp.choiceは選択肢のインデックスを返し、space_evalで実値を復元する",
        choices: [
            "Hyperoptは実際のパラメータ値を返す",
            "Hyperoptはchoiceリストのインデックスを返し、hyperopt.space_eval()で実値を取得する",
            "Hyperoptはパラメータ値の辞書を返す",
            "Hyperoptはchoiceリストのインデックスを返し、実値は取得できない"
        ],
        correctIndex: 1,
        explanation: `
            <p>hp.choice()を使うと、Hyperoptは選択肢リスト内の選択された要素の<b>インデックス</b>を返します。実際の値に戻すには<code>hyperopt.space_eval()</code>を使用してインデックスを実値にマッピングします。</p>

            <div class="code-block">from hyperopt import hp, fmin, tpe, Trials, space_eval

space = {'model': hp.choice('model', ['svm','rf','xgboost'])}
trials = Trials()
best = fmin(fn=lambda p: 0.5, space=space, algo=tpe.suggest, max_evals=1, trials=trials)
actual = space_eval(space, best)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>実際の値を返す:</strong> hp.choiceは内部ではインデックスを返す</li>
                <li><strong>辞書を返す:</strong> 返される辞書の中のchoiceはインデックスである</li>
                <li><strong>実値は取得できない:</strong> space_evalでマッピングが可能である</li>
            </ul>
        `
    },
    {
        number: 24,
        domain: "Databricks ML",
        question: "Databricksでノートブックのバージョン管理を実装する推奨アプローチは何ですか？",
    keyPoint: "ノートブックの組み込みバージョン管理機能を有効化する",
        choices: [
            "Databricks Jobsで定期的にスナップショットを作成する",
            "ノートブックをエクスポートして外部のバージョン管理システムにコミットする",
            "MLflow Trackingでノートブックのバージョンを自動でログする",
            "Databricksノートブックの組み込みのバージョン管理機能を有効にする"
        ],
        correctIndex: 3,
        explanation: `
            <p>Databricksはノートブックのリビジョン履歴やGit連携など、組み込みのバージョン管理機能を提供しています。これを有効にすることで、変更追跡や以前のバージョンへのロールバックが容易になります。</p>

            <div class="code-block"># Databricks Reposを使ったGit連携の例
databricks repos create --url https://github.com/user/project --provider gitHub --path /Repos/project</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>Jobsでスナップショット:</strong> Jobsはスケジューリング用でバージョン管理ではない</li>
                <li><strong>外部でのエクスポート:</strong> 手動作業が増え、Git連携機能の方が効率的</li>
                <li><strong>MLflowでログ:</strong> MLflowは実験トラッキング用でノートブック編集履歴管理ではない</li>
            </ul>
        `
    },
    {
        number: 25,
        domain: "ML workflows",
        question: "CrossValidatorをPipeline内に配置すべき状況はどれですか？",
    keyPoint: "前処理によるデータリーケージを防ぐためPipeline内に配置する",
        choices: [
            "パイプラインにestimatorやtransformerが含まれる場合",
            "パイプライン内の前処理でデータリーケージが発生する恐れがある場合",
            "パイプライン内でrefitを行いたい場合",
            "モデルを並列で訓練したい場合"
        ],
        correctIndex: 1,
        explanation: `
            <p>CrossValidatorをPipeline内に置く主な理由は、前処理（スケーリング、欠損値補完など）が交差検証の各分割で適切に適用され、データリーケージ（検証データが学習プロセスに影響すること）を防ぐためです。</p>

            <div class="code-block">from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
lr = LogisticRegression(featuresCol="scaled_features")
pipeline = Pipeline(stages=[scaler, lr])
</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>estimator/transformerが含まれる:</strong> それ自体は理由にならない</li>
                <li><strong>refit:</strong> refitは一般的な手順であり主な理由ではない</li>
                <li><strong>並列訓練:</strong> 並列化はSparkの機能であり置き場所とは無関係</li>
            </ul>
        `
    }
    ,
    {
        number: 26,
        domain: "Databricks ML",
        question: "MLflowクライアントで実験から最新の実行（run）を取得した後、その実行のメトリクスにアクセスするにはどうすればよいですか？",
    keyPoint: "runs[0].data.metricsで最新runのmetrics辞書を取得する",
        choices: [
            "metrics = runs[0].data.metrics",
            "metrics = runs[0].get_metrics()",
            "metrics = runs[0].fetch_metrics()",
            "metrics = runs[0].metrics.data"
        ],
        correctIndex: 0,
        explanation: `
            <p>MLflowのrunオブジェクトからメトリクスを取得する正しい方法は、<code>runs[0].data.metrics</code>を参照することです。<code>data</code>属性にはログされたmetrics, params, tagsが格納されています。</p>

            <div class="code-block">from mlflow.tracking import MlflowClient
client = MlflowClient()
runs = client.search_runs(experiment_id, order_by=["attributes.start_time desc"], max_results=1)
metrics = runs[0].data.metrics
print(metrics)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>get_metrics / fetch_metrics:</strong> そのようなメソッドはrunオブジェクトには存在しない</li>
                <li><strong>metrics.data:</strong> metricsは辞書であり、さらにdata属性を持たない</li>
            </ul>
        `
    },
    {
        number: 27,
        domain: "Databricks ML",
        question: "チームメンバーがノートブックの重要なセルを誤って削除した場合、どのようにそのセルを復元できますか？",
    keyPoint: "ノートブックのリビジョン履歴から以前のバージョンに戻す",
        choices: [
            "ノートブックのバージョン履歴を使って以前のバージョンに戻す",
            "削除したチームメンバーにセルを再実行してもらう",
            "ノートブックをエクスポートして削除されたコードを探し、再インポートする",
            "Databricksのゴミ箱機能でセルを復元する"
        ],
        correctIndex: 0,
        explanation: `
            <p>Databricksノートブックにはリビジョン履歴があり、過去のバージョンを確認して復元できます。削除されたセルが含まれるバージョンを選択して復元するのが推奨手順です。</p>

            <div class="code-block"># ノートブックのRevision Historyを開き、以前のバージョンを選択してRestoreします</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>チームメンバーに再実行してもらう:</strong> セルが失われた場合は再現できないことが多い</li>
                <li><strong>エクスポートして再インポート:</strong> 手間がかかり非効率</li>
                <li><strong>ゴミ箱:</strong> ゴミ箱はノートブック単位の復元を対象で、セル単位の復元はリビジョン履歴を使う</li>
            </ul>
        `
    },
    {
        number: 28,
        domain: "Databricks ML",
        question: "Databricks Jobsで外部イベントやトリガーに基づいてタスクをオーケストレーションできる機能は何ですか？",
    keyPoint: "Event-driven schedulingで外部イベントをトリガーにできる",
        choices: [
            "Event-driven scheduling",
            "Time-based scheduling",
            "Dependency-based scheduling",
            "Task-based scheduling"
        ],
        correctIndex: 0,
        explanation: `
            <p>Databricks JobsのEvent-driven schedulingは、S3へのアップロードやメッセージキューの受信など外部イベントをトリガーにジョブを起動できます。リアルタイムや準リアルタイムのパイプラインで有用です。</p>

            <div class="code-block"># 例: 新しいファイルがアップロードされたときにジョブをトリガーする設定（クラウドイベント連携を使用）
/* Cloud provider specific configuration */</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>Time-based:</strong> 時刻ベースのスケジューリングでイベント駆動ではない</li>
                <li><strong>Dependency-based:</strong> ジョブ内のタスク依存関係の管理で外部イベントではない</li>
                <li><strong>Task-based:</strong> タスク単位の実行管理でイベント駆動とは異なる</li>
            </ul>
        `
    },
    {
        number: 29,
        domain: "Hyperopt",
        question: "分散Spark MLモデルのベイズ的ハイパーパラメータチューニングを有効にするツールはどれですか？",
    keyPoint: "HyperoptはSparkTrialsと組合せ分散ベイズ最適化を可能にする",
        choices: [
            "Hyperopt",
            "Autoscaling clusters",
            "Feature Store",
            "MLflow Experiment Tracking",
            "AutoML"
        ],
        correctIndex: 0,
        explanation: `
            <p>HyperoptはTree-structured Parzen Estimator（TPE）などのベイズ最適化手法を提供し、SparkTrialsと組み合わせることでSparkクラスタ上で分散ハイパーパラメータ最適化を行えます。</p>

            <div class="code-block">from hyperopt import fmin, tpe, hp, SparkTrials
spark_trials = SparkTrials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=spark_trials)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>Autoscaling:</strong> リソース管理であり最適化手法ではない</li>
                <li><strong>Feature Store / MLflow / AutoML:</strong> それぞれ目的が異なり分散ベイズ最適化のツールではない</li>
            </ul>
        `
    },
    {
        number: 30,
        domain: "Hyperopt",
        question: "SparkTrialsクラスはHyperoptでどのような役割を果たし、いつ使用すべきですか？",
    keyPoint: "SparkTrialsは試行をSparkワーカーに分配してチューニングを加速する",
        choices: [
            "分散モデルのハイパーパラメータ空間を定義する",
            "試行をSparkワーカーに分配して単一マシンのチューニングを加速する",
            "チューニング結果をMLflowにログする",
            "MLlibやHorovodのような分散MLアルゴリズムを実行する"
        ],
        correctIndex: 1,
        explanation: `
            <p>SparkTrialsはHyperoptのために設計され、試行(trials)をSparkクラスタ上のワーカーに分配して並列評価を行います。これにより大規模なハイパーパラメータ探索を高速化できます。</p>

            <div class="code-block">from hyperopt import SparkTrials
spark_trials = SparkTrials()</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>空間定義:</strong> ハイパーパラメータ空間はhp関数で定義する</li>
                <li><strong>MLflowログ:</strong> MLflowは別途統合する必要がある</li>
                <li><strong>分散MLアルゴリズム実行:</strong> SparkTrialsは最適化の分配であり、MLアルゴリズムそのものを実行するものではない</li>
            </ul>
        `
    }
    ,
    {
        number: 31,
        domain: "Scaling ML Models",
        question: "反復処理を必要とする分散機械学習モデルで、中間データをメモリにキャッシュして高速化するSparkの機能は何ですか？",
    keyPoint: "Spark RDD Persistenceで中間データをメモリにキャッシュする",
        choices: [
            "Spark SQL",
            "Spark MLlib",
            "Spark GraphX",
            "Spark RDD Persistence"
        ],
        correctIndex: 3,
        explanation: `
            <p>反復処理（例えば反復型アルゴリズム）では、中間データを再計算するコストを避けるためにRDDをメモリにキャッシュ（persist/cache）することが重要です。SparkのRDD Persistenceはこの目的に使われます。</p>

            <div class="code-block">from pyspark import SparkContext
from pyspark.storagelevel import StorageLevel
sc = SparkContext()
data = sc.parallelize(range(1000000))
data.persist(StorageLevel.MEMORY_ONLY)
# 反復処理
for i in range(10):
    result = data.map(lambda x: x * i).reduce(lambda a, b: a + b)
data.unpersist()</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>Spark SQL/MLlib/GraphX:</strong> それぞれ異なる用途のコンポーネントであり、直接のキャッシュ機能はRDD Persistenceが担当する</li>
            </ul>
        `
    },
    {
        number: 32,
        domain: "Spark ML Basics",
        question: "共有クラスタのすべてのノートブックでPythonライブラリ 'newpackage' を利用可能にする最適な方法はどれですか？",
    keyPoint: "クラスタのinitスクリプトで /databricks/python/bin/pip install を実行する",
        choices: [
            "クラスタをDatabricks Runtime for Machine Learningに変更する",
            "Sparkセッションのruntime-version変数を'ml'に設定する",
            "任意のノートブックで一度だけ %pip install newpackage を実行する",
            "クラスタのbash initスクリプトに /databricks/python/bin/pip install newpackage を追加する"
        ],
        correctIndex: 3,
        explanation: `
            <p>共有クラスタ上のすべてのノートブックで恒久的に利用可能にするには、クラスタのinitスクリプトでパッケージをインストールするのが推奨されます。Initスクリプトはクラスタ起動時に実行され、全ユーザー・全ノートブックに対してパッケージが利用可能になります。</p>

            <div class="code-block">#!/bin/bash
/databricks/python/bin/pip install newpackage</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>Runtime変更:</strong> ML Runtimeは一般的なライブラリを含むがカスタムパッケージの恒久的インストールにはinitスクリプトが適切</li>
                <li><strong>%pip:</strong> セッション単位のインストールでクラスタ再起動時に失われる</li>
            </ul>
        `
    },
    {
        number: 33,
        domain: "Databricks ML",
        question: "Databricks AutoMLで回帰モデルの性能評価にデフォルトで使用される指標はどれですか？",
    keyPoint: "Databricks AutoMLの回帰のデフォルト指標はR-squared（R2）",
        choices: [
            "Mean squared error (MSE)",
            "Mean absolute error (MAE)",
            "R-squared (R2)",
            "Root mean squared error (RMSE)"
        ],
        correctIndex: 2,
        explanation: `
            <p>Databricks AutoMLでは、回帰問題のデフォルト評価指標としてR-squared（R2）が使用されます。R2は説明変数が目的変数の分散をどれだけ説明できるかを示します。</p>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>MSE/MAE/RMSE:</strong> いずれも回帰評価で使われるが、Databricks AutoMLのデフォルトはR2である</li>
            </ul>
        `
    },
    {
        number: 34,
        domain: "Databricks ML",
        question: "NLP向けの前処理で、テキストのトークン化とストップワード除去にSpark MLlibのどの機能を使うべきですか？",
    keyPoint: "TokenizerとStopWordsRemoverでトークン化→ストップワード除去を行う",
        choices: [
            "StringIndexer",
            "Tokenizer",
            "TokenizerとStopWordsRemover",
            "CountVectorizer"
        ],
        correctIndex: 2,
        explanation: `
            <p>テキスト前処理ではまずTokenization（Tokenizer）で単語に分割し、次にStopWordsRemoverで不要な一般語を除去します。これらを組み合わせることでスケーラブルなNLP前処理が実現できます。</p>

            <div class="code-block">from pyspark.ml.feature import Tokenizer, StopWordsRemover
data = spark.createDataFrame([(0, "I love Databricks and machine learning")], ["id","text"])
tokenizer = Tokenizer(inputCol="text", outputCol="words")
tokenized = tokenizer.transform(data)
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
filtered = remover.transform(tokenized)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>StringIndexer:</strong> ラベルのインデックス化用でトークン化ではない</li>
                <li><strong>CountVectorizer:</strong> 特徴量化のためで、トークン化・除去の後に使う</li>
            </ul>
        `
    },
    {
        number: 35,
        domain: "Hyperopt",
        question: "Hyperoptのfmin()関数の目的と必須引数は何ですか？",
    keyPoint: "fminは目的関数と探索空間を受け探索を実行する関数",
        choices: [
            "チューニング結果をMLflowにログする",
            "ハイパーパラメータ空間を定義する",
            "ハイパーパラメータ探索を実行し最適解を探す",
            "単一マシンのモデル計算を並列化する"
        ],
        correctIndex: 2,
        explanation: `
            <p>fmin()はHyperoptのコア関数で、与えられた目的関数と探索空間を用いてハイパーパラメータ探索を実行します。主要な引数には <code>fn</code>（目的関数）、<code>space</code>（探索空間）、<code>algo</code>、<code>max_evals</code>、<code>trials</code> があります。</p>

            <div class="code-block">from hyperopt import fmin, tpe, hp, Trials
def objective(params):
    return loss
space = {'lr': hp.uniform('lr', 0.001, 0.1)}
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>MLflowログ:</strong> fmin自体はログ機能を持たず、別途統合が必要</li>
                <li><strong>空間定義:</strong> 空間はhp関数で定義する</li>
                <li><strong>並列化:</strong> 並列化はtrials引数（例：SparkTrials）が担当する</li>
            </ul>
        `
    },
    {
        number: 36,
        domain: "ML workflows",
        question: "MLflowで一連の実行が完了した後、RMSEが最も低い実行を特定するために次のコードを使いたい。mlflow._________(experiment_id, order_by = [\"metrics.rmse\"])['run_id'][0]。空欄に当てはまる関数は何ですか？",
    keyPoint: "mlflow.search_runsでorder_by=['metrics.rmse']を指定して最良runを取得する",
        choices: [
            "client",
            "search_runs",
            "experiment",
            "identify_run",
            "show_runs"
        ],
        correctIndex: 1,
        explanation: `
            <p>MLflowで実行リストを取得してソートするには<code>mlflow.search_runs()</code>を使います。<code>order_by=['metrics.rmse']</code>として最小RMSE順に並べ、先頭のrun_idを参照することで最良の実行を得られます。</p>

            <div class="code-block">best_run_id = mlflow.search_runs(experiment_id, order_by=["metrics.rmse"])['run_id'][0]</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>client/experiment/identify_run/show_runs:</strong> 該当する関数/オブジェクト名ではない</li>
            </ul>
        `
    },
    {
        number: 37,
        domain: "Data Preparation",
        question: "複数の特徴量に欠損値が含まれるデータセットを扱う際、Databricks MLlibで欠損値を効果的に処理する手法はどれですか？",
    keyPoint: "欠損値はImputerで平均/中央値/最頻値などで補完する",
        choices: [
            "Data Imputation",
            "Feature Scaling",
            "Outlier Detection",
            "Feature Selection"
        ],
        correctIndex: 0,
        explanation: `
            <p>欠損値を扱う代表的な手法はデータインピュテーション（Data Imputation）です。Databricks MLlibは列ごとに平均、中央値、最頻値などで補完するImputerを提供しています。</p>

            <div class="code-block">from pyspark.ml.feature import Imputer
# サンプルDataFrame
data = [(1, 10.0), (2, None), (3, 30.0), (4, 40.0)]
columns = ["id", "value"]
df = spark.createDataFrame(data, columns)
# Imputerで欠損値を補完
imputer = Imputer(inputCols=["value"], outputCols=["value_imputed"]).setStrategy("mean")
imputed_df = imputer.fit(df).transform(df)
imputed_df.show()</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>Feature Scaling:</strong> 正規化・標準化などであり欠損値処理ではない</li>
                <li><strong>Outlier Detection:</strong> 外れ値検出は欠損値の処理手法ではない</li>
                <li><strong>Feature Selection:</strong> 特徴選択は重要度に基づく削減であり欠損値処理ではない</li>
            </ul>
        `
    },
    {
        number: 38,
        domain: "ML workflow",
        question: "Spark MLで数値列の欠損値を中央値で補完しようとしたが、コードが期待どおりに動作しない理由は何ですか？",
    keyPoint: "Imputerはまずfit()でImputerModelを生成してからtransformを呼ぶ必要がある",
        choices: [
            "中央値での補完は不可能である",
            "学習用と評価用を同時に補完しない",
            "inputColsとoutputColsは完全に一致する必要がある",
            "ImputerをデータにfitしてImputerModelを作成していない"
        ],
        correctIndex: 3,
        explanation: `
            <p>Imputerを使用する際は、まず<code>fit()</code>でImputerModelを作成して各列の中央値（または平均）を計算する必要があります。fitをスキップして直接transformを呼ぶと補完は行われません。</p>

            <div class="code-block">from pyspark.ml.feature import Imputer
input_columns = ["col1","col2","col3"]
output_columns = ["col1_imputed","col2_imputed","col3_imputed"]
my_imputer = Imputer(strategy="median", inputCols=input_columns, outputCols=output_columns)
# fitでImputerModelを作成
imputer_model = my_imputer.fit(features_df)
# transformで補完を適用
imputed_df = imputer_model.transform(features_df)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>中央値での補完は不可能である:</strong> Imputerはmean/medianをサポートしている</li>
                <li><strong>学習用と評価用を同時に補完しない:</strong> 同じImputerModelを用いれば両方に適用できるが、これは今回の不具合の原因ではない</li>
                <li><strong>inputColsとoutputColsは完全に一致する必要がある:</strong> 出力列名は新しい列名でも構わない</li>
            </ul>
        `
    },
    {
        number: 39,
        domain: "Scaling ML Models",
        question: "非常に大規模なデータセットに対する線形回帰問題をSpark MLはどのように扱いますか？",
    keyPoint: "大規模線形回帰は行列反転の代わりに勾配降下法を使って最適化する",
        choices: [
            "Brute Force Algorithm",
            "Matrix decomposition",
            "Singular value decomposition",
            "Least square method",
            "Gradient descent"
        ],
        correctIndex: 4,
        explanation: `
            <p>大規模データに対する線形回帰では、巨大な行列を反転する従来の最小二乗法は現実的ではないため、Spark MLは勾配降下法などの反復最適化手法を使います。これによりメモリに載らないデータでも学習が可能です。</p>

            <div class="code-block">from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol="features", labelCol="label", solver="gd")  # 勾配降下法を指定
model = lr.fit(df)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>行列分解/最小二乗:</strong> 大規模データでの行列反転は非現実的</li>
                <li><strong>SVD:</strong> 主に次元削減に用いられる手法であり、回帰のスケーラブルな解法ではない</li>
            </ul>
        `
    },
    {
        number: 40,
        domain: "Orchestrating Multi-task ML Workflows",
        question: "登録済みモデルの新しいバージョンをテスト合格後にProductionステージに移行するために使用されるMLflow Client APIの操作はどれですか？",
    keyPoint: "client.transition_model_version_stageでモデルバージョンをProductionに移行する",
        choices: [
            "Client.update_model_stage",
            "client.transition_model_version_stage",
            "client.transition_model_version",
            "client.update_model_version"
        ],
        correctIndex: 1,
        explanation: `
            <p>MLflow Client APIでは、<code>transition_model_version_stage</code>を用いてモデルバージョンを特定のステージ（例: Staging、Production）に移行します。これによりModel Registry内でのステージ管理が行えます。</p>

            <div class="code-block">from mlflow.tracking import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(name="my_model", version=1, stage="Production")</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>Client.update_model_stage / transition_model_version / update_model_version:</strong> これらは正しいAPI名ではない、またはステージ遷移のためのメソッドではない</li>
            </ul>
        `
    },
    {
        number: 41,
        domain: "SparkTrials",
        question: "SparkTrialsを使う際、MLflowの実行管理に関して推奨される方法は何ですか、そしてその理由は何ですか？",
    keyPoint: "fmin()呼び出しをwith mlflow.start_run()でラップしてメイン実行を分ける",
        choices: [
            "with mlflow.start_run()を使うと競合を避けるため避ける",
            "複数のfmin()を単一のMLflow実行で行う",
            "fmin()の呼び出しをwith mlflow.start_run()でラップして別のメイン実行にする",
            "各トライアルごとに別個のMLflow実行を作る"
        ],
        correctIndex: 2,
        explanation: `
            <p>SparkTrialsとHyperoptを用いる際は、<code>fmin()</code>の呼び出しを<code>with mlflow.start_run()</code>でラップすることが推奨されます。これにより各fmin実行が個別のメイン実行として扱われ、各トライアルはその下のネストされた実行としてログされます。</p>

            <div class="code-block">import mlflow
from hyperopt import fmin, tpe, hp, SparkTrials
with mlflow.start_run():
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=SparkTrials())</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>単一実行で複数fmin:</strong> 実験の管理が混乱するため推奨されない</li>
                <li><strong>各トライアルごとに個別実行:</strong> SparkTrialsはトライアルをネスト実行として自動でログするため冗長で非効率</li>
            </ul>
        `
    },
    {
        number: 42,
        domain: "Databricks ML",
        question: "FeatureStoreClientをfsとしてインスタンス化した後、primary_keysに指定する正しい形式はどれですか？",
    keyPoint: "primary_keysはリストで指定する（例: [\"index\"]）",
        choices: [
            "[\"index\"]",
            "\"index\"",
            "(\"index\")",
            "index",
            "None of the above"
        ],
        correctIndex: 0,
        explanation: `
            <p>Databricks Feature Storeでは、<code>primary_keys</code>は行を一意に識別するカラム名のリストとして渡す必要があります。単一の主キーでもリスト（例: <code>["index"]</code>）で指定します。</p>

            <div class="code-block">from databricks.feature_store import FeatureStoreClient
fs = FeatureStoreClient()
fs.create_table(
    name="airbnb_features",
    primary_keys=["index"],
    schema=airbnb_df.schema,
    description="All Errors are captured in this table"
)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>"index" / ("index") / index:</strong> 文字列やタプル、未定義変数になり得るため正しくない</li>
                <li><strong>None of the above:</strong> リスト形式が正しい</li>
            </ul>
        `
    },
    {
        number: 43,
        domain: "Hyperopt",
        question: "SparkTrialsでparallelismを設定する方法と、そのトレードオフは何ですか？",
    keyPoint: "parallelismは手動設定で速度と適応性のトレードオフ",
        choices: [
            "Spark実行子の数に基づき自動設定される",
            "クラスタの同時実行タスク数に合わせて設定される",
            "ハイパーパラメータ空間の試行数で決まる",
            "オプション引数で設定し速度と適応性のトレードオフがある"
        ],
        correctIndex: 3,
        explanation: `
            <p>SparkTrialsのparallelismはユーザーが明示的に指定するオプション引数で、同時に実行するトライアル数を決めます。parallelismを高くすると速度は上がるが適応性（過去の結果を踏まえた探索効率）は下がることがあります。</p>

            <div class="code-block">from hyperopt import SparkTrials
trials = SparkTrials(parallelism=4)  # 4つのトライアルを同時実行</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li>実行子数やクラスタ設定で自動決定されるわけではない</li>
                <li>試行の総数（max_evals）はparallelismとは別の概念である</li>
            </ul>
        `
    },
    {
        number: 44,
        domain: "Feature Store",
        question: "sklearnの決定木をSpark MLに移植中、maxBinsがカテゴリの値数以上である必要があるとエラーが出た。なぜSpark MLはmaxBinsをカテゴリのユニーク数以上に要求するのか？",
    keyPoint: "カテゴリごとに少なくとも1つのビンが必要だから",
        choices: [
            "Spark MLは単一ノード実装より多くの分割候補を必要とする",
            "各カテゴリ特徴量の各カテゴリに少なくとも1つのビンが必要だから",
            "Spark MLは分割でカテゴリ特徴のみをテストする",
            "Spark MLは分割で数値特徴のみをテストする"
        ],
        correctIndex: 1,
        explanation: `
            <p>Spark MLの決定木ではカテゴリ特徴量をビンに分けて扱います。<code>maxBins</code>は利用可能なビン数を制御し、各カテゴリが少なくとも1つのビンを持てるようにする必要があるため、ユニークなカテゴリ数以上を指定しなければなりません。</p>

            <div class="code-block">from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(featuresCol="categoryIndex", labelCol="label", maxBins=3)  # num categories <= maxBins</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li>maxBinsは分割候補数ではなくカテゴリのビニングに関係する</li>
                <li>Spark MLはカテゴリ／数値の両方をサポートする</li>
            </ul>
        `
    }
    ,
    {
        number: 45,
        domain: "Feature Store",
        question: "Spark MLlibで回帰の予測(prediction)と正解(label)があるDataFrameから平均絶対誤差(MAE)を計算する正しいコードはどれですか？",
    keyPoint: "RegressionEvaluator(metricName='mae')を使ってMAEを算出する",
        choices: [
            "mae_evaluator = RegressionEvaluator(predictionCol=\"prediction\", labelCol=\"label\", metricName=\"mae\")\nmae = mae_evaluator.evaluate(regression_preds_df)",
            "mae_evaluator = MulticlassClassificationEvaluator(predictionCol=\"prediction\", labelCol=\"label\", metricName=\"mae\")\nmae = mae_evaluator.evaluate(regression_preds_df)",
            "mae_evaluator = BinaryClassificationEvaluator(predictionCol=\"prediction\", labelCol=\"label\", metricName=\"mae\")\nmae = mae_evaluator.evaluate(regression_preds_df)",
            "mae_evaluator = RegressionSummarizer(predictionCol=\"prediction\", labelCol=\"label\", metricName=\"mae\")\nmae = mae_evaluator.evaluate(regression_preds_df)"
        ],
        correctIndex: 0,
        explanation: `
            <p>回帰モデルの評価指標としてMAEを計算するには<code>RegressionEvaluator</code>を用い、<code>metricName="mae"</code>を指定します。</p>

            <div class="code-block">from pyspark.ml.evaluation import RegressionEvaluator
mae_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="mae")
mae = mae_evaluator.evaluate(regression_preds_df)
print(mae)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li>Multiclass/Binary evaluatorは分類用でありMAE計算には不適切</li>
                <li>RegressionSummarizerはこの用途の標準的なAPIではない</li>
            </ul>
        `
    }
    ,
    {
        number: 46,
        domain: "Spark ML",
        question: "複雑で非線形な判別境界を持つ分類問題に対して、Spark MLで適切なアルゴリズムはどれですか？",
    keyPoint: "非線形分離はDecision Treeが適している",
        choices: [
            "Linear Regression",
            "Decision Trees",
            "Naive Bayes",
            "Support Vector Machines"
        ],
        correctIndex: 1,
        explanation: `
            <p>複雑で非線形な決定境界を扱うにはDecision Treeが有効です。ツリーベースのモデルは階層的な分割で非線形領域を表現できます。</p>

            <div class="code-block">from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
model = dt.fit(train_df)
predictions = model.transform(test_df)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li>Linear Regressionは回帰用であり線形モデルに限定される</li>
                <li>Naive Bayesは特徴独立性の仮定があり複雑境界には弱い</li>
                <li>SVMはSpark MLで一般的にサポートされていない（カーネルSVM等）</li>
            </ul>
        `
    }
    ,
    {
        number: 47,
        domain: "Scaling ML Models",
        question: "大規模機械学習プロジェクトで反復アルゴリズムの高速化のため、中間データをメモリに保持してディスクI/Oを削減する手法は何ですか？",
        keyPoint: "中間データをメモリにキャッシュする（cache()/persist()）",
        choices: [
            "Data Shuffling",
            "In-Memory Computation",
            "Disk Caching",
            "Data Replication"
        ],
        correctIndex: 1,
        explanation: `
            <p>In-Memory Computationは、中間データをメモリに格納することでSparkの性能を最適化する重要な手法です。データを再計算したりディスクから繰り返し読み込んだりする代わりに、メモリに保持することで高速化します。</p>

            <h4>重要性:</h4>
            <ul>
                <li>多くの機械学習アルゴリズム（勾配降下法、KMeans、決定木など）はデータに対して複数回のパスを必要とする</li>
                <li>In-Memory Computationなしでは、Sparkは変換を元データセットから再計算するか、ディスクから読み込む必要があり、パフォーマンスが低下する</li>
                <li>データをメモリに永続化することで、ディスクI/Oと再計算のオーバーヘッドを大幅に削減できる</li>
            </ul>

            <div class="code-block">from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("InMemoryComputation").getOrCreate()
df = spark.read.csv("large_dataset.csv", header=True, inferSchema=True)

# DataFrameをメモリにキャッシュ
df.cache()  # メモリに格納して高速アクセス

# 複数の反復計算を実行
df.groupBy("category").count().show()
df.groupBy("category").avg("value").show()</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>Data Shuffling:</strong> シャッフルはパーティション間でデータを再配分するコストの高い操作であり、中間データをメモリに格納するものではない</li>
                <li><strong>Disk Caching:</strong> ディスクキャッシングはデータをディスクに保存するためメモリよりも遅く、ディスクI/Oが発生する</li>
                <li><strong>Data Replication:</strong> レプリケーションはデータ損失を防ぐために使用されるが、反復計算の高速化には直接寄与しない</li>
            </ul>
        `
    }
    ,
    {
            number: 48,
            domain: "Scaling ML Models",
            question: "Spark MLで2クラスの決定木分類器について、predictionとactualの列を持つDataFrameから精度(accuracy)を計算する適切なコードはどれですか？",
            keyPoint: "MulticlassClassificationEvaluatorでaccuracyを評価する",
            choices: [
                "accuracy = RegressionEvaluator(predictionCol=\"prediction\", labelCol=\"actual\", metricName=\"accuracy\")",
                "classification_evaluator = MulticlassClassificationEvaluator(predictionCol=\"prediction\", labelCol=\"actual\", metricName=\"accuracy\")\naccuracy = classification_evaluator.evaluate(preds_df)",
                "classification_evaluator = BinaryClassificationEvaluator(predictionCol=\"prediction\", labelCol=\"actual\", metricName=\"accuracy\")\naccuracy = classification_evaluator.evaluate(preds_df)",
                "accuracy = Summarizer(predictionCol=\"prediction\", labelCol=\"actual\", metricName=\"accuracy\")"
            ],
            correctIndex: 1,
            explanation: `
                <p>2クラス分類の精度を計算する一般的な方法は、<code>MulticlassClassificationEvaluator</code>を用いて<code>metricName="accuracy"</code>を指定し評価することです。BinaryClassificationEvaluatorはAUCなどの指標向けであり、精度を扱う際はMulticlassClassificationEvaluatorが適切です。</p>

                <div class="code-block">from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="actual", metricName="accuracy")
    accuracy = evaluator.evaluate(preds_df)
    print(accuracy)</div>

                <h4>他の選択肢が不正解な理由:</h4>
                <ul>
                    <li>RegressionEvaluatorは回帰指標用でありaccuracyは対象外</li>
                    <li>BinaryClassificationEvaluatorは主にAUC等のバイナリ指標に使う</li>
                    <li>Summarizerは精度計算の標準APIではない</li>
                </ul>
            `
        },
        {
            number: 49,
            domain: "Cluster Creation and Management",
            question: "大規模な計算能力が必要な機械学習プロジェクトのためにDatabricksクラスタを設定する際、最も適したクラスタタイプはどれですか？",
            keyPoint: "計算負荷が大きければマルチノードクラスタを使う",
            choices: [
                "Single-node cluster",
                "Standard cluster",
                "Multi-node cluster",
                "Task-specific cluster"
            ],
            correctIndex: 2,
            explanation: `
                <p>大量計算を要するMLワークロードでは、複数ノードに処理を分散できるマルチノードクラスタが最適です。複数のexecutorで並列処理でき、オートスケールやGPUノードの利用などにより性能を向上できます。</p>

                <div class="code-block"># Databricks UIでMulti-nodeクラスタを作成し、ワーカー数やインスタンスタイプを設定します
    # 例: ワーカー数を増やし、必要であればGPUインスタンスを選択する</div>

                <h4>他の選択肢が不正解な理由:</h4>
                <ul>
                    <li>Single-nodeは小規模データやプロトタイプ向け</li>
                    <li>Standardは一般カテゴリで、明示的にMulti-nodeを選ぶのが適切</li>
                    <li>Task-specificは短時間ジョブ向けで継続的なMLトレーニングには向かない</li>
                </ul>
            `
        },
        {
            number: 50,
            domain: "AutoML",
            question: "AutoMLのfeature_store_lookupsパラメータの目的は何ですか？",
            keyPoint: "Feature Storeの特徴量をデータに結合して拡張するために使う",
            choices: [
                "AutoMLの実行時間を制御する",
                "AutoMLで除外するアルゴリズムを指定する",
                "ワークスペース内のディレクトリパスを指定する",
                "Feature Storeの特徴量をデータ拡張のために表現する"
            ],
            correctIndex: 3,
            explanation: `
                <p><code>feature_store_lookups</code>はDatabricks Feature Storeのテーブルから特徴量を取得して元データに結合（データ拡張）するために使用されます。これにより手動での結合作業を自動化し、再現性の高い特徴量活用が可能になります。</p>

                <div class="code-block">from databricks.feature_store import FeatureLookup
    feature_store_lookups = [
        FeatureLookup(table_name="customer_features", lookup_key="customer_id", feature_names=["age","income"])
    ]
    databricks.automl.classify(dataset=df, target_col="churn_label", feature_store_lookups=feature_store_lookups)</div>

                <h4>他の選択肢が不正解な理由:</h4>
                <ul>
                    <li>実行時間はtimeout_minutesで制御する</li>
                    <li>除外アルゴリズムはexcluded_algorithmsで指定する</li>
                    <li>ワークスペースパスはexperiment_dir等を使う</li>
                </ul>
            `
        },
        {
            number: 51,
            domain: "AutoML",
            question: "マルチシリーズの時系列予測で、複数の時系列を識別するために使用すべきパラメータはどれですか？",
            keyPoint: "identity_colで時系列ごとに識別子を指定する",
            choices: [
                "identity_col",
                "output_database",
                "time_col",
                "frequency"
            ],
            correctIndex: 0,
            explanation: `
                <p>マルチシリーズ予測では、各時系列を区別するために<code>identity_col</code>を指定します。例えば店舗ごとの売上を予測する場合、<code>identity_col="store_id"</code>のように各系列を識別します。</p>

                <div class="code-block">databricks.automl.forecast(dataset=df, target_col="sales", time_col="date", identity_col="store_id")</div>

                <h4>他の選択肢が不正解な理由:</h4>
                <ul>
                    <li><strong>time_col:</strong> 時刻列を指定するが系列の識別はしない</li>
                    <li><strong>frequency:</strong> 時系列の頻度指定であり識別子ではない</li>
                </ul>
            `
        },
        {
            number: 52,
            domain: "ML workflows",
            question: "3-fold CVと指定のハイパーパラメータグリッド（Hyper1: [4,6,7], Hyper2: [5,10]）を使ったとき、合計で何個のモデルが訓練されるか？",
            keyPoint: "組合せ数(3x2)=6にfold数3を掛け、合計18モデル",
            choices: [
                "2",
                "6",
                "12",
                "18",
                "24"
            ],
            correctIndex: 3,
            explanation: `
                <p>ハイパーパラメータの組合せは3×2=6通りで、各組合せに対して3分割交差検証を行うため、6×3=18個のモデルが訓練されます。</p>

                <div class="code-block"># combinations = 3 * 2 = 6
    # total models = combinations * folds = 6 * 3 = 18</div>

                <h4>補足:</h4>
                <p>ここでの「simultaneously」は総数を指し、並列実行数とは別の概念です。</p>
            `
        },
        {
            number: 53,
            domain: "Databricks ML",
            question: "Databricks Runtime for Machine Learningでの効率的なクラスタ構成が可能にすることは何ですか？",
            keyPoint: "リソースを最適利用してコスト効率よく性能を出す",
            choices: [
                "Quick data preprocessing",
                "Optimal resource utilization",
                "Streamlined model deployment",
                "Real-time data visualization"
            ],
            correctIndex: 1,
            explanation: `
                <p>効率的なクラスタ構成は計算リソースの最適利用（オートスケール、適切なインスタンスタイプの選択、メモリ/CPUのバランス）を可能にし、性能向上とコスト削減を両立します。</p>

                <div class="code-block"># 例: Worker typeを適切に選び、Autoscalingを有効化してコスト効率を改善</div>

                <h4>他の選択肢が不正解な理由:</h4>
                <ul>
                    <li>前処理やデプロイ、可視化は他の設定やツールに依存する</li>
                </ul>
            `
        },
        {
            number: 54,
            domain: "Databricks ML",
            question: "既存のモデルバージョンのメタデータ（説明やタグ）を更新したい場合に使うMLflow操作はどれですか？",
            keyPoint: "client.update_model_versionでモデルバージョンのメタデータを更新する",
            choices: [
                "mlflow.update_model_metadata",
                "mlflow.register_model",
                "mlflow.update_model_version",
                "mlflow.edit_model_version"
            ],
            correctIndex: 2,
            explanation: `
                <p>MLflow Model Registryでモデルバージョンの説明やタグなどのメタデータを更新するには<code>update_model_version()</code>（MlflowClientのclient.update_model_version）を使います。</p>

                <div class="code-block">from mlflow.tracking import MlflowClient
    client = MlflowClient()
    client.update_model_version(name="my_model", version=2, description="Updated model description")
    client.set_model_version_tag(name="my_model", version=2, key="dataset_version", value="v2.1")</div>

                <h4>他の選択肢が不正解な理由:</h4>
                <ul>
                    <li>update_model_metadata/edit_model_versionは存在しない関数名</li>
                    <li>register_modelは新規登録で既存の更新ではない</li>
                </ul>
            `
        }
    ,
    {
        number: 55,
        domain: "AutoML",
        question: "分類タスクのAutoML実行中に、PrecisionやRecallなどのメトリクス計算のために正のクラスを明示的に定義する必要がある場合、どのパラメータを使用すべきですか？",
        keyPoint: "pos_labelで正クラスを明示的に指定する",
        choices: [
            "primary_metric",
            "pos_label",
            "max_trials",
            "time_col"
        ],
        correctIndex: 1,
        explanation: `
            <p>Databricks AutoMLの分類タスクでは、<code>pos_label</code>パラメータを使用して、Precision、Recall、F1スコアなどのメトリクスを計算する際の正のクラスを明示的に定義します。</p>

            <h4>なぜ必要か:</h4>
            <ul>
                <li>デフォルトでは、AutoMLは大きい値を正のクラスと仮定する（例: バイナリ分類{0,1}では1）</li>
                <li>不均衡なラベルや異なる正のクラスが必要な場合は、<code>pos_label</code>を手動で指定して正確なメトリクス計算を保証する</li>
            </ul>

            <div class="code-block">import databricks.automl

databricks.automl.classify(
    dataset=df,
    target_col="churn",
    pos_label=1,  # '1'を正のクラスとして明示的に設定
    primary_metric="f1"
)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>primary_metric:</strong> 主要評価メトリクスを選択するが、正のクラスは定義しない</li>
                <li><strong>max_trials:</strong> AutoML実験の数を制御するが、分類ラベルの定義には影響しない</li>
                <li><strong>time_col:</strong> 時系列予測専用で分類には使用しない</li>
            </ul>
        `
    }
    ,
    {
        number: 56,
        domain: "Scaling ML Models",
        question: "分散コンピューティング環境において、データパーティショニングの主な目的は何ですか？",
        keyPoint: "データを複数ノードに分散して並列処理を実現する",
        choices: [
            "ストレージコストの削減",
            "データセキュリティの強化",
            "ノード間でのデータ分散",
            "圧縮の有効化"
        ],
        correctIndex: 2,
        explanation: `
            <p>分散コンピューティング環境では、データパーティショニングは主に<strong>データを複数ノードに分散</strong>して並列処理を可能にし、クエリ性能を向上させ、計算ワークロードを効率的にバランスさせるために使用されます。</p>

            <h4>重要性:</h4>
            <ul>
                <li>パーティショニングは大規模データセットを小さなチャンク（パーティション）に分割し、クラスタ内の複数ノードが並列処理できるようにする</li>
                <li>Apache Spark、Databricks、Hadoopなどのビッグデータフレームワークで不可欠</li>
            </ul>

            <div class="code-block">df = spark.read.csv("large_dataset.csv", header=True, inferSchema=True)

# DataFrameを10パーティションに再分割して並列処理
df = df.repartition(10)

# パーティション数を確認
print(df.rdd.getNumPartitions())</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>ストレージコスト削減:</strong> パーティショニングは直接ストレージコストを削減しない。圧縮や最適化ファイル形式（Parquet、ORC）がストレージに役立つ</li>
                <li><strong>セキュリティ強化:</strong> パーティショニングは並列処理用でセキュリティ用ではない。セキュリティは暗号化やアクセス制御で管理</li>
                <li><strong>圧縮の有効化:</strong> 圧縮はファイルサイズを削減するがノード間のデータ分散は行わない</li>
            </ul>
        `
    }
    ,
    {
        number: 57,
        domain: "Spark ML",
        question: "分類モデル評価のためのメトリクスガイドラインで、どのような状況でF1スコアが精度（accuracy）より好まれるべきですか？",
        keyPoint: "クラス不均衡があり偽陰性を減らすことが重要な場合",
        choices: [
            "目的変数が3つ以上のカテゴリを持つ場合",
            "実際の正例と負例の数が等しい場合",
            "正と負のクラス間に大きな不均衡があり偽陰性を最小化することが重要な場合",
            "目的変数がちょうど2つのクラスで構成される場合",
            "真陽性と真陰性を正しく識別することがビジネス上同等に重要な場合"
        ],
        correctIndex: 2,
        explanation: `
            <p>F1スコアは精度とリコールの調和平均であり、次のシナリオで特に有用です：</p>

            <h4>F1スコアが推奨される状況:</h4>
            <ul>
                <li><strong>クラス不均衡が存在:</strong> 不均衡データセット（例: 95%負クラス、5%正クラス）では、精度は誤解を招く。マジョリティクラスを予測するモデルは高精度に見えるが、マイノリティクラスを捉えられない</li>
                <li><strong>偽陰性がコスト高:</strong> 詐欺検出や病気診断のようなケースでは、正例を見逃す（偽陰性）ことが深刻な結果を招く。F1スコアはリコールを重視して偽陰性を最小化する</li>
            </ul>

            <div class="code-block">from sklearn.metrics import f1_score, accuracy_score

# 不均衡データでの評価
y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # 90% negative, 10% positive
y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # すべて負と予測

print("Accuracy:", accuracy_score(y_true, y_pred))  # 0.9 (高い！)
print("F1 Score:", f1_score(y_true, y_pred))  # 0.0 (正クラスを捉えていない)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>3つ以上のカテゴリ:</strong> F1スコアは主にバイナリ分類用。マルチクラス問題にはF1-micro/macroを使用</li>
                <li><strong>正負が等しい場合:</strong> バランスが取れた場合は精度で十分。F1スコアは不均衡時に力を発揮</li>
                <li><strong>2クラスである:</strong> これは事実だが、F1を選ぶ理由にはならない。重要なのは不均衡やコストの非対称性</li>
                <li><strong>真陽性と真陰性が同等:</strong> 精度は既に真陽性/陰性をバランスする。F1は偽陰性/陽性が非対称な場合に適切</li>
            </ul>
        `
    }
    ,
    {
        number: 58,
        domain: "Databricks ML",
        question: "共有Databricksクラスタでチーム全員がサードパーティPythonライブラリ（etl_utils）を使用できるようにする推奨方法は何ですか？",
        keyPoint: "クラスタ初期化スクリプトでライブラリをインストールする",
        choices: [
            "Databricks Runtime for Data Engineeringを使用するようクラスタを編集する",
            "クラスタ設定でPYTHONPATH変数にetl_utilsのパスを含める",
            "クラスタに接続された任意のノートブックで%pip install etl_utilsを一度実行する",
            "クラスタの初期化スクリプトでdbutils.library.installPyPI('etl_utils')コマンドを使用する",
            "etl_utilsライブラリをクラスタで利用可能にする方法はない"
        ],
        correctIndex: 3,
        explanation: `
            <p>共有Databricksクラスタでサードパーティライブラリをすべてのユーザーに利用可能にする推奨方法は、<strong>クラスタの初期化スクリプト</strong>にライブラリインストールコマンドを追加することです。</p>

            <h4>初期化スクリプトの例:</h4>
            <div class="code-block">#!/bin/bash
/databricks/python/bin/pip install etl_utils</div>

            <h4>なぜこれが機能するか:</h4>
            <ul>
                <li>初期化スクリプトはクラスタ起動時に毎回実行され、ライブラリがグローバルにインストールされる</li>
                <li>クラスタに接続されたすべてのユーザーが手動インストールなしでetl_utilsにアクセスできる</li>
            </ul>

            <div class="code-block"># ノートブックでの初期化スクリプト例
dbutils.library.installPyPI("etl_utils")
dbutils.library.restartPython()  # オプション: 変更を適用するためPythonを再起動</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>Runtime for Data Engineering:</strong> ランタイム環境を変更するだけでカスタムライブラリはインストールされない</li>
                <li><strong>PYTHONPATHを設定:</strong> モジュール検索パス用でパッケージインストールには使えない</li>
                <li><strong>%pip install:</strong> 現在のセッションのみにインストールされ、他のユーザーには見えない</li>
                <li><strong>方法がない:</strong> 不正解。初期化スクリプトやUIのライブラリタブでグローバルインストール可能</li>
            </ul>
        `
    }
    ,
    {
        number: 59,
        domain: "ML workflows",
        question: "単一のTrain-Test Splitが交差検証（Cross-Validation）よりも優れているシナリオはどれですか？",
        keyPoint: "計算時間とリソースが制限されている場合",
        choices: [
            "モデル性能を最大化することが目標の場合",
            "モデルの安定性と汎化性を確保することが目標の場合",
            "計算時間とリソースが制限されている場合",
            "データセットが不均衡な場合"
        ],
        correctIndex: 2,
        explanation: `
            <p>単一のTrain-Test Splitは、<strong>計算時間とリソースが制約されている</strong>シナリオで交差検証より望ましいです。</p>

            <h4>理由:</h4>
            <ul>
                <li><strong>速度と効率:</strong> Train-Test Splitは1回の訓練/テストサイクルのみで評価するため、高速でリソース消費が少ない</li>
                <li><strong>交差検証（k-fold）:</strong> モデルをk回訓練する必要があり（各フォールドごとに1回）、大規模データや複雑なモデルでは計算コストが高い</li>
            </ul>

            <h4>使用例:</h4>
            <ul>
                <li>探索的分析: モデルの実現可能性を素早く検証</li>
                <li>大規模データセット: データが豊富な場合、単一分割でも信頼性の高い性能推定が可能</li>
                <li>リソース制約: ハードウェアが限られている（小規模クラスタ）や期限が厳しい場合</li>
            </ul>

            <div class="code-block">from sklearn.model_selection import train_test_split

# 単一分割（高速）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>性能最大化:</strong> 交差検証の方がより多くのデータを訓練に使用し、堅牢な性能推定を提供するため優れている</li>
                <li><strong>安定性と汎化性:</strong> 交差検証は複数フォールドで汎化性を評価し過学習を減らすため優れている</li>
                <li><strong>不均衡データセット:</strong> 層化交差検証（StratifiedKFold）の方が各フォールドでクラス分布を維持できるため適切</li>
            </ul>
        `
    }
    ,
    {
        number: 60,
        domain: "Feature Store",
        question: "Unity Catalogで新しいカタログを作成するにはどうすればよいですか？",
        keyPoint: "CREATE CATALOG mlでカタログを作成する",
        choices: [
            "spark.sql('CREATE CATALOG IF NOT EXISTS ml')を実行する",
            "spark.sql('CREATE CATALOG ml')を実行する",
            "spark.sql('USE CATALOG ml')を実行する",
            "spark.sql('CREATE SCHEMA ml')を実行する"
        ],
        correctIndex: 1,
        explanation: `
            <p>Unity Catalogで新しいカタログを作成するには、<code>CREATE CATALOG</code> SQLコマンドを使用する必要があります。このコマンドは、データ資産を整理するための新しいトップレベルカタログを初期化します。</p>

            <div class="code-block"># カタログを作成
spark.sql("CREATE CATALOG ml")

# カタログ内にスキーマを作成
spark.sql("CREATE SCHEMA ml.default")

# 確認
spark.sql("SHOW CATALOGS").show()</div>

            <h4>重要なポイント:</h4>
            <ul>
                <li>カタログはUnity Catalogのトップレベル名前空間（例: ml）</li>
                <li>スキーマ（データベース）はカタログ内に存在（例: ml.default）</li>
                <li>管理者権限またはCREATE CATALOGパーミッションが必要</li>
            </ul>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>CREATE CATALOG IF NOT EXISTS:</strong> この構文は有効だが、問題は基本的な作成コマンドを求めている。IF NOT EXISTS句はオプション</li>
                <li><strong>USE CATALOG:</strong> 既存のカタログに切り替えるコマンドで、作成はしない</li>
                <li><strong>CREATE SCHEMA:</strong> スキーマ（データベース）はカタログ内に作成される。カタログレベルではない</li>
            </ul>
        `
    }
    ,
    {
        number: 61,
        domain: "Databricks ML",
        question: "Databricks MLlibで訓練・評価したモデルを本番環境でリアルタイム予測にデプロイする際、どのステップを踏むべきですか？",
        keyPoint: "MLflowでモデルをパッケージ化しREST APIとしてデプロイする",
        choices: [
            "モデルをシリアル化ファイルとしてエクスポートし別サーバーにデプロイする",
            "MLflowを使用してモデルをパッケージ化しREST APIエンドポイントとしてデプロイする",
            "モデルをDeltaテーブルに保存しクエリで予測を取得する",
            "Databricks Jobをスケジュールしてモデルを定期的に実行する"
        ],
        correctIndex: 1,
        explanation: `
            <p>Databricks MLlibモデルを本番環境にデプロイする最も堅牢でスケーラブルなアプローチは、<strong>MLflowを使用</strong>してモデルをパッケージ化し、REST APIエンドポイントとしてデプロイすることです。</p>

            <h4>手順:</h4>
            <ol>
                <li><strong>モデルをパッケージ化:</strong> mlflow.spark.log_model()で訓練済みモデルをログ</li>
                <li><strong>REST APIとしてデプロイ:</strong> MLflow Model ServingまたはAzure ML/AWS SageMakerにデプロイしてリアルタイム推論を実現</li>
            </ol>

            <div class="code-block">import mlflow
from pyspark.ml import PipelineModel

# モデルをログ
model = PipelineModel.load("path/to/model")
mlflow.spark.log_model(model, "spark-model")

# デプロイ（Databricks Model Serving UIなど経由）
# REST APIエンドポイントが自動的に作成される</div>

            <h4>MLflowデプロイの主な利点:</h4>
            <ul>
                <li><strong>依存関係管理:</strong> Python/Sparkバージョンを自動キャプチャ</li>
                <li><strong>スケーラビリティ:</strong> 組み込みのロードバランシング（Databricks Model Serving経由）</li>
                <li><strong>モニタリング:</strong> レイテンシ、エラー、使用状況メトリクスを追跡</li>
            </ul>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>シリアル化ファイルで別サーバー:</strong> 手動デプロイはエラーが発生しやすく（依存関係の不一致、スケーリング問題）、MLflowの監視機能やスケーラビリティがない</li>
                <li><strong>Deltaテーブルに保存:</strong> Deltaテーブルはデータストレージ用でモデルサービング用ではない。リアルタイム予測には低レイテンシAPIが必要</li>
                <li><strong>Jobをスケジュール:</strong> Jobはバッチ処理用でリアルタイム推論ではない</li>
            </ul>
        `
    }
    ,
    {
        number: 62,
        domain: "Data Preparation",
        question: "モデル効率を改善するための特徴選択プロジェクトで、Databricks MLlibがサポートする、データセットから最も重要な特徴を識別する手法はどれですか？",
        keyPoint: "Feature Importance Rankingで重要な特徴を特定する",
        choices: [
            "Principal Component Analysis",
            "Recursive Feature Elimination",
            "Feature Importance Ranking",
            "Feature Scaling"
        ],
        correctIndex: 2,
        explanation: `
            <p>Databricks MLlibの<strong>Feature Importance Ranking</strong>は、データセット内で最も影響力のある特徴を特定するための手法であり、ツリーベースモデル（Random Forest、Gradient-Boosted Treesなど）で特徴の貢献度を定量化します。</p>

            <h4>動作方法:</h4>
            <ol>
                <li>ツリーベースモデルを訓練（例: RandomForestClassifier）</li>
                <li>特徴重要度を抽出:</li>
            </ol>

            <div class="code-block">from pyspark.ml.classification import RandomForestClassifier

# モデル訓練
rf = RandomForestClassifier(featuresCol="features", labelCol="label")
model = rf.fit(train_data)

# 特徴重要度を取得
importances = model.featureImportances
print("Feature Rankings:", importances)

# 上位特徴を選択（スコアに基づいて高→低）</div>

            <h4>特徴選択に推奨される理由:</h4>
            <ul>
                <li><strong>モデルに基づく洞察:</strong> モデル性能に直接結びついている</li>
                <li><strong>スケーラブル:</strong> 分散Spark DataFrameで動作</li>
                <li><strong>解釈可能:</strong> スコアは相対的重要度を示す（例: 0.8=高影響、0.1=低影響）</li>
            </ul>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>PCA:</strong> 次元削減だが新しい合成特徴を作成（元の特徴の解釈可能性を失う）。ユースケースは特徴抽出で選択ではない</li>
                <li><strong>Recursive Feature Elimination:</strong> MLlibではネイティブサポートされていない（カスタム実装が必要）。大規模データセットでは計算コストが高い</li>
                <li><strong>Feature Scaling:</strong> 特徴範囲を正規化（StandardScalerなど）するが、特徴をランク付けや選択はしない</li>
            </ul>
        `
    }
    ,
    {
        number: 63,
        domain: "Databricks ML",
        question: "本番環境でMLflowを使用してモデルサービングを実装する際、モデルをDockerコンテナとしてデプロイしREST APIとして公開するために使用できるMLflowコマンドは何ですか？",
        keyPoint: "mlflow.models.build_dockerでDockerイメージを作成する",
        choices: [
            "mlflow.build_docker_image",
            "mlflow.dockerize_model",
            "mlflow.create_container",
            "mlflow.serve_model",
            "mlflow.models.build_docker"
        ],
        correctIndex: 4,
        explanation: `
            <p>MLflowモデルをREST APIを持つDockerコンテナとしてデプロイするには、<code>mlflow.models.build_docker</code>関数を使用します。このコマンドは：</p>

            <ul>
                <li>モデルとその依存関係をDockerイメージにパッケージ化</li>
                <li>リアルタイム予測のためのREST APIエンドポイントを公開</li>
            </ul>

            <h4>デプロイ手順:</h4>

            <div class="code-block"># 1. 訓練中にモデルをログ
import mlflow
mlflow.sklearn.log_model(model, "model")

# 2. Dockerイメージをビルド
mlflow.models.build_docker(
    model_uri="runs:/<RUN_ID>/model",
    name="mlflow-docker-model",
    env_manager="conda"
)

# 3. コンテナを実行
# docker run -p 5000:8080 mlflow-docker-model
# モデルは http://localhost:5000/invocations でアクセス可能</div>

            <h4>MLflow Dockerデプロイの主な利点:</h4>
            <ul>
                <li><strong>一貫性:</strong> モデル+環境をカプセル化</li>
                <li><strong>スケーラビリティ:</strong> Kubernetes、AWS ECSなどにデプロイ可能</li>
                <li><strong>低レイテンシ:</strong> リアルタイム推論用に最適化</li>
            </ul>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>mlflow.build_docker_image:</strong> 存在しない関数名。正しいコマンドはmlflow.models.build_docker</li>
                <li><strong>mlflow.dockerize_model:</strong> 有効なMLflow関数ではない</li>
                <li><strong>mlflow.create_container:</strong> MLflowに存在しない</li>
                <li><strong>mlflow.serve_model:</strong> ローカルRESTサーバーを起動するがDockerイメージは作成しない</li>
            </ul>
        `
    }
    ,
    {
        number: 64,
        domain: "Pandas API on Spark",
        question: "pandas API on Sparkで変換する際、PySparkのDecimalType(38, 18)にマッチするPythonデータ型はどれですか？",
        keyPoint: "decimal.DecimalがDecimalType(38, 18)にマッチする",
        choices: [
            "float",
            "int",
            "bytes",
            "decimal.Decimal"
        ],
        correctIndex: 3,
        explanation: `
            <p>pandas API on Spark（Koalas）を使用する際、PySparkの<code>DecimalType(38, 18)</code>（高精度10進数型）はPythonの<code>decimal.Decimal</code>にマッピングされ、精度を保持します。これにより、floatで一般的な浮動小数点丸め誤差を回避できます。</p>

            <h4>重要性:</h4>
            <ul>
                <li><strong>DecimalType(38, 18):</strong> 合計38桁の数値を格納し、そのうち18桁が小数点以下。金融/科学データで精度が重要な場合に使用</li>
                <li><strong>decimal.Decimal:</strong> Pythonの任意精度10進数型。正確な10進数表現を保証（floatのようなバイナリ浮動小数点ではない）</li>
            </ul>

            <div class="code-block">from pyspark.sql import SparkSession
import decimal

# DecimalTypeを持つSpark DataFrameを作成
spark = SparkSession.builder.getOrCreate()
df_spark = spark.createDataFrame(
    [(decimal.Decimal("12345678901234567890.123456789012345678"),)],
    schema="value DECIMAL(38, 18)"
)

# pandas-on-Spark DataFrameに変換
df_koalas = df_spark.to_koalas()

# Pythonの型を確認
print(type(df_koalas["value"].iloc[0]))  # Output: <class 'decimal.Decimal'></div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>float:</strong> DecimalType値の精度を失う（丸め誤差）。PySparkはFloatType/DoubleTypeにfloatを使用し、DecimalTypeには使用しない</li>
                <li><strong>int:</strong> 小数部を表現できない。PySparkのIntegerType/LongTypeに使用</li>
                <li><strong>bytes:</strong> 数値型とは無関係。PySparkのBinaryTypeがbytesにマッピング</li>
            </ul>

            <h4>重要なポイント:</h4>
            <p>pandas API on Sparkで正確な10進精度を得るには、PySparkのDecimalTypeに対してdecimal.Decimalを使用します。金融/科学データで丸め誤差を防ぐためにfloatを避けましょう。</p>
        `
    }
    ,
    {
        number: 65,
        domain: "Spark ML Basics",
        question: "Spark MLで訓練した機械学習モデルの結果を解釈する際、情報に基づいたビジネス上の意思決定を行うために最も重要なことは何ですか？",
        keyPoint: "出力とパラメータを解釈してビジネスアクションにつなげる",
        choices: [
            "訓練時間",
            "モデルの複雑さ",
            "出力とパラメータの解釈",
            "特徴量の数"
        ],
        correctIndex: 2,
        explanation: `
            <p>情報に基づいたビジネス上の意思決定を行うには、<strong>モデルの出力とパラメータを解釈する</strong>ことが最も重要です。これには以下が含まれます：</p>

            <h4>解釈の重要な側面:</h4>
            <ol>
                <li><strong>予測の理解:</strong> モデルの予測がビジネスにとって何を意味するか（例: 詐欺検出モデルの「確率スコア」はリスクレベルに変換される）</li>
                <li><strong>パラメータ/特徴の分析:</strong> どの特徴が予測を駆動しているか（例: ツリーベースモデルのfeatureImportances）</li>
                <li><strong>実行可能なインサイト:</strong> モデル結果をビジネスアクションに変換（例: 「低エンゲージメントの顧客をターゲットにして解約を減らす」）</li>
            </ol>

            <div class="code-block"># Spark MLで特徴重要度を抽出
model = pipeline.fit(train_data)
importances = model.stages[-1].featureImportances  # ツリーベースモデル用

# ビジネスコンテキストで検証
# 例: 'credit_utilization'が高いと本当にデフォルトを予測するか？</div>

            <h4>ビジネスへの影響例:</h4>
            <p>銀行がローン承認モデルのfeatureImportancesを使用して「債務対収入比率」が最上位の要因であることを発見。これに基づいて融資ポリシーを調整します。</p>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>訓練時間:</strong> エンジニアリング効率には関連するが、ビジネス上の意思決定に直接影響しない</li>
                <li><strong>モデルの複雑さ:</strong> 保守性に影響するが、予測理由を説明しない</li>
                <li><strong>特徴量の数:</strong> 性能には重要だが、意思決定には解釈の方が重要</li>
            </ul>

            <h4>重要なポイント:</h4>
            <p>ビジネス上の意思決定では、解釈 > メトリクス。以下に焦点を当てましょう：</p>
            <ul>
                <li>モデルが何を予測するか</li>
                <li>なぜその予測をするのか</li>
                <li>それに基づいてどう行動するか</li>
            </ul>
        `
    }
];

// Make questions available globally
if (typeof window !== 'undefined') {
    window.questions = questions;
}
