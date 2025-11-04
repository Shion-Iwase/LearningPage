// DBMLA 試験問題データ2

const questions2 = [
    {
        number: 1,
        domain: "Spark ML Algorithms",
        question: "不正取引を識別するデータセットを扱う機械学習プロジェクトにおいて、稀で異常なインスタンスを識別する異常検知タスクに適したSpark MLアルゴリズムはどれですか？",
        keyPoint: "Isolation Forestは異常検知に最適化されている",
        choices: [
            "Linear Regression",
            "Decision Trees",
            "Naive Bayes",
            "Isolation Forest"
        ],
        correctIndex: 3,
        explanation: `
            <p>Isolation Forestは、異常検知に特化して設計されたSpark MLアルゴリズムです。稀または異常なインスタンス（例: 不正取引）を識別するために、正常な動作をプロファイリングするのではなく、データ内の異常を分離することに優れています。</p>

            <h4>Isolation Forestの主な特徴:</h4>
            <ul>
                <li><strong>異常検知:</strong> 不均衡データセット（少数の不正ケース vs 多数の正常取引）で効果的。ランダムパーティショニングを使用して正常点より速く異常を分離</li>
                <li><strong>スケーラビリティ:</strong> 高次元データ（不正検知で一般的）に対して効率的</li>
                <li><strong>Spark ML統合:</strong> pyspark.ml.feature.IsolationForestで利用可能</li>
            </ul>

            <div class="code-block">from pyspark.ml.feature import IsolationForest

# Isolation Forestを初期化
isolation_forest = IsolationForest(
    featuresCol="features",
    predictionCol="prediction",
    contamination=0.05  # 異常の割合（5%）
)

# 訓練と予測
model = isolation_forest.fit(train_data)
predictions = model.transform(test_data)

# 異常は1としてマーク（正常: 0）
predictions.filter(predictions.prediction == 1).show()</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>Linear Regression:</strong> 連続値を予測するもので異常検知ではない</li>
                <li><strong>Decision Trees:</strong> 分類/回帰に使用され、異常検知に最適化されていない</li>
                <li><strong>Naive Bayes:</strong> 確率的分類器で、不均衡な異常検知には効果的でない</li>
            </ul>
        `
    }
    ,
    {
        number: 2,
        domain: "Databricks ML",
        question: "MLflow実行を実行したノートブックを表示するために使用できる方法は何ですか？",
        keyPoint: "MLflow実験ページの「Source」リンクでノートブックを開く",
        choices: [
            "MLflow実行ページでmodel.pklアーティファクトを開く",
            "MLflow実験ページで実行に対応する「Models」リンクをクリックする",
            "MLflow実行ページでMLmodelアーティファクトを開く",
            "MLflow実験ページで実行に対応する「Start Time」リンクをクリックする",
            "MLflow実験ページで実行の行にある「Source」リンクをクリックする"
        ],
        correctIndex: 4,
        explanation: `
            <p>MLflow実行を実行したノートブック（またはスクリプト）を表示するには:</p>

            <ol>
                <li>DatabricksのMLflow実験ページに移動</li>
                <li>実行テーブルで特定の実行を見つける</li>
                <li>実行の行にある<strong>「Source」リンク</strong>をクリック。これにより実行を開始したノートブック/ジョブにリダイレクトされる</li>
            </ol>

            <h4>なぜこれが機能するか:</h4>
            <ul>
                <li>MLflowの「Source」フィールドは実行の起点（例: ノートブックパス、ジョブID）を追跡する</li>
                <li>クリックすると、使用された正確なノートブックバージョン（Gitにリンクされている場合はコミット履歴を含む）が開かれる</li>
            </ul>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>model.pklアーティファクト:</strong> シリアル化されたモデルを保存するが、ノートブックではない</li>
                <li><strong>「Models」リンク:</strong> Model Registryにリダイレクトされ、ソースコードではない</li>
                <li><strong>MLmodelアーティファクト:</strong> モデルメタデータ（例: フレーバー、依存関係）を含むが、ノートブックではない</li>
                <li><strong>「Start Time」リンク:</strong> 実行時間/タイムスタンプを表示するが、ソースではない</li>
            </ul>

            <h4>重要性:</h4>
            <ul>
                <li><strong>再現性:</strong> 使用された正確なコードを検証</li>
                <li><strong>デバッグ:</strong> エラーを起点まで追跡</li>
            </ul>
        `
    }
    ,
    {
        number: 3,
        domain: "Spark ML",
        question: "100万レコードのSpark DataFrameでシングルノードモデルの推論をスケールする際、Iteratorを使用する利点は何ですか？",
        keyPoint: "Iteratorでモデルは各executorで1回だけロードされる",
        choices: [
            "モデルが単一executorに制限されデータ分散が防止される",
            "データが単一executorに制限されモデルの複数回ロードが防止される",
            "推論プロセス中にデータが複数executorに分散される",
            "IteratorをI/Oとして含めることに利点はない",
            "推論プロセス中にモデルはバッチごとではなくexecutorごとに1回だけロードされる"
        ],
        correctIndex: 4,
        explanation: `
            <p>Pandas UDF（@pandas_udf）で<strong>Iterator</strong>を使用すると、モデルがデータのバッチごとではなくexecutorごとに1回だけロードされるため、効率が向上します。</p>

            <h4>なぜ重要か:</h4>
            <ul>
                <li>Iteratorなしでは、Sparkが処理するデータのバッチごとにモデルが再ロードされ、高い計算オーバーヘッドが発生</li>
                <li>Iteratorありでは、executorごとにモデルが1回ロードされ、関数はモデルを再ロードせずに複数バッチを処理</li>
            </ul>

            <div class="code-block">from typing import Iterator
import pandas as pd
import mlflow
from pyspark.sql.functions import pandas_udf

@pandas_udf("double")
def predict(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.Series]:
    # モデルをexecutorごとに1回だけロード
    model = mlflow.sklearn.load_model("models:/my_model/Production")

    # 複数バッチを処理（モデル再ロードなし）
    for batch in iterator:
        yield pd.Series(model.predict(batch))

# Spark DataFrameに適用
predictions = df.select(predict("features").alias("prediction"))</div>

            <h4>Iteratorが性能を最適化する仕組み:</h4>
            <ul>
                <li><code>Iterator[pd.DataFrame]</code>はバッチごとに関数を個別に呼び出す代わりに、executorごとに複数バッチを処理</li>
                <li>モデルロードのI/Oと計算コストが削減される</li>
            </ul>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>単一executorに制限:</strong> データは複数Spark executorに分散されたまま。Iteratorはexecutorごとのモデルロードを最適化するが、データ分散には影響しない</li>
                <li><strong>データが単一executorに:</strong> データは依然として複数executorで処理される</li>
                <li><strong>データ分散:</strong> これはSparkでデフォルトで発生。Iteratorはモデルロードのみに影響</li>
                <li><strong>利点なし:</strong> 不正確。Iteratorは不要なモデル再ロードを削減し性能を大幅改善</li>
            </ul>
        `
    }
    ,
    {
        number: 4,
        domain: "AutoML",
        question: "時系列データセットで予測のために時間を表す列を指定する場合、どのパラメータを使用すべきですか？",
        keyPoint: "time_colで時間列を指定する",
        choices: [
            "time_col",
            "target_col",
            "max_trials",
            "exclude_cols"
        ],
        correctIndex: 0,
        explanation: `
            <p>Databricks AutoMLや他のMLフレームワークで時系列予測を扱う場合、<code>time_col</code>パラメータを使用してデータセット内の時間要素を表す列（例: 日付やタイムスタンプ）を指定します。これにより、モデルがデータの時間構造を正しく解釈できます。</p>

            <div class="code-block">from databricks import automl

# 予測用の時間列を指定
summary = automl.forecast(
    dataset=df,
    time_col="date",  # 時間列を指定
    target_col="sales",
    horizon=7,
    frequency="D"
)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>target_col:</strong> 予測する列（例: 売上）を指定するが、時間列ではない</li>
                <li><strong>max_trials:</strong> ハイパーパラメータチューニングのトライアル数を制御し、時系列とは無関係</li>
                <li><strong>exclude_cols:</strong> 訓練時に無視する列をリストするが、時間を識別するためではない</li>
            </ul>

            <h4>重要性:</h4>
            <p>このパラメータは以下に重要です:</p>
            <ul>
                <li>時系列順にデータを訓練/検証セットに分割</li>
                <li>予測のための将来の時点を生成（horizon）</li>
            </ul>
        `
    }
    ,
    {
        number: 5,
        domain: "Spark ML Algorithms",
        question: "不規則な時間間隔を含む時系列予測プロジェクトで、欠損時点を効果的に処理するために有用なDatabricksの機能やライブラリは何ですか？",
        keyPoint: "Databricks Time Series Libraryで欠損値を処理する",
        choices: [
            "MLlib CrossValidator",
            "Databricks Delta",
            "Databricks Time Series Library",
            "MLflow Tracking"
        ],
        correctIndex: 2,
        explanation: `
            <p>不規則な間隔と欠損時点を持つ時系列予測で欠損値を効果的に処理するには、<strong>Databricks Time Series Library</strong>が最適です。</p>

            <h4>なぜこのライブラリが最適か:</h4>
            <ul>
                <li><strong>欠損データ処理:</strong> 線形補間、前方埋め、統計的手法（ARIMA、ETSなど）などの様々な補完技術を使用して欠損値を補完可能</li>
                <li><strong>不規則間隔の処理:</strong> ギャップや不規則な間隔を効率的に処理し、不均一なデータでも正確な予測を提供</li>
                <li><strong>Sparkとの統合:</strong> Spark DataFrameとシームレスに統合し、時系列データの処理と操作が容易</li>
                <li><strong>包括的機能:</strong> 予測、異常検知、トレンド分析などの機能を含む</li>
            </ul>

            <div class="code-block"># Databricks Time Series Libraryの使用例
from databricks.feature_store import time_series

# 欠損値の補完
df_filled = time_series.fill_missing(
    df,
    time_col="timestamp",
    value_col="sales",
    method="linear"  # 線形補間
)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>MLlib CrossValidator:</strong> モデル性能評価に有用だが、欠損データ問題には直接対処しない</li>
                <li><strong>Databricks Delta:</strong> データレイクのストレージ形式で、主に信頼性とバージョン管理に焦点。Time Series Libraryのように欠損値を処理できない</li>
                <li><strong>MLflow Tracking:</strong> 主にML実験のログと管理用で、データ操作や欠損値処理には特化していない</li>
            </ul>
        `
    }
    ,
    {
        number: 6,
        domain: "Feature Store",
        question: "Databricks Runtimeでdatabricks-feature-engineeringクライアントをインストールする方法は？",
        keyPoint: "%pip install コマンドでパッケージをインストールする",
        choices: [
            "pip install databricks-feature-engineering",
            "%pip install databricks-feature-engineering",
            "conda install databricks-feature-engineering",
            "spark install databricks-feature-engineering"
        ],
        correctIndex: 1,
        explanation: `
            <p>Databricks Runtimeでdatabricks-feature-engineeringクライアントをインストールするには、ノートブックセルで<code>%pip</code>マジックコマンドを使用します。これにより、クラスタのPython環境にパッケージがインストールされ、クラスタに接続しているすべてのユーザーが利用できるようになります。</p>

            <h4>手順:</h4>
            <ol>
                <li>Databricksノートブックを開く</li>
                <li>以下を実行:
                    <div class="code-block">%pip install databricks-feature-engineering</div>
                </li>
                <li>プロンプトが表示されたらPythonカーネルを再起動（または<code>dbutils.library.restartPython()</code>を実行）</li>
            </ol>

            <h4>なぜこれが機能するか:</h4>
            <ul>
                <li><strong>%pip:</strong> Databricksノートブックでのパッケージインストールの推奨方法</li>
                <li><strong>クラスタ全体で利用可能:</strong> 現在のクラスタセッションでパッケージをインストール</li>
            </ul>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>pip install:</strong> ローカルPython環境では機能するが、Databricksノートブックでは機能しない（%pipを使用すべき）</li>
                <li><strong>conda install:</strong> CondaはDatabricks Runtimeのデフォルトパッケージマネージャーではない</li>
                <li><strong>spark install:</strong> 無効なコマンド（Sparkはパッケージインストールを処理しない）</li>
            </ul>

            <p><strong>注意:</strong> 本番環境では以下も検討:</p>
            <ul>
                <li>クラスタ初期化スクリプト（永続的なインストール用）</li>
                <li>Libraries UI（クラスタ設定下）</li>
            </ul>
        `
    }
    ,
    {
        number: 7,
        domain: "Pandas API on Spark",
        question: "Pandas API on Sparkでplot.barやplot.pieなどのtop-nベースのプロットの視覚的制限を制御するオプションは？",
        keyPoint: "plotting.max_rowsでプロット行数を制限する",
        choices: [
            "plotting.max_rows",
            "compute.default_index_type",
            "compute.shortcut_limit",
            "compute.ops_on_diff_frames"
        ],
        correctIndex: 0,
        explanation: `
            <p><code>plotting.max_rows</code>オプションは、Pandas API on Sparkで生成されるtop-nベースのプロットに視覚的に表示される最大行数を制御します。</p>

            <h4>対象プロット:</h4>
            <ul>
                <li>plot.bar()</li>
                <li>plot.pie()</li>
                <li>行のサブセットを選択する他のプロット</li>
            </ul>

            <h4>効果:</h4>
            <p>特に大規模データセットを扱う場合、バー、スライス、その他の視覚要素の数を制限することで、プロットの視覚的複雑性を管理するのに役立ちます。</p>

            <div class="code-block">import pandas as pd

# top-nプロットの最大視覚制限を設定
pd.set_option("plotting.max_rows", 20)  # プロットで最大20行を表示

# DataFrameを作成
df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]})

# 棒グラフをプロット（top 20行のバーを表示）
df.plot.bar()</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>compute.default_index_type:</strong> 新しいDataFrameのデフォルトインデックスタイプを制御するが、プロット動作ではない</li>
                <li><strong>compute.shortcut_limit:</strong> 特定の操作でドライバに収集できる最大行数を制御するが、プロットに直接影響しない</li>
                <li><strong>compute.ops_on_diff_frames:</strong> 異なるインデックスを持つDataFrameでの操作の処理方法を決定するが、プロットとは無関係</li>
            </ul>

            <h4>重要ポイント:</h4>
            <ul>
                <li>プロットの雑然さを減らし解釈しやすくする</li>
                <li>プロット用に処理されるデータを制限することで大規模データセットの性能を改善</li>
                <li>現在の値は<code>pd.get_option("plotting.max_rows")</code>で確認可能</li>
            </ul>
        `
    }
    ,
    {
        number: 8,
        domain: "Spark ML",
        question: "Pandas API on SparkがJVMとPythonプロセス間でデータを効率的に転送するために使用するインメモリ列形式は何ですか？",
        keyPoint: "Apache Arrowでゼロコピーデータ転送を実現する",
        choices: [
            "Parquet",
            "ORC",
            "Avro",
            "Apache Arrow"
        ],
        correctIndex: 3,
        explanation: `
            <p>Pandas API on Spark（Koalas）は、JVM（Spark）とPython（pandas）プロセス間でデータを効率的に転送するために、インメモリ列形式として<strong>Apache Arrow</strong>を使用します。</p>

            <h4>Apache Arrowの利点:</h4>
            <ul>
                <li><strong>ゼロコピーデータ共有:</strong>
                    JVMとPython間で直接メモリアクセスを可能にし、シリアライゼーション/デシリアライゼーションのオーバーヘッドを排除
                </li>
                <li><strong>列形式:</strong>
                    ベクトル化操作により分析ワークロード（Sparkなど）に最適化
                </li>
                <li><strong>シームレスな統合:</strong>
                    Pandas DataFrame ↔ Spark DataFrameの変換がArrowによって高速化
                </li>
            </ul>

            <div class="code-block">import databricks.koalas as ks

# Spark DataFrame → Koalas（内部でArrowを使用）
kdf = spark_df.to_koalas()

# Koalas → Pandas（Arrowバック）
pdf = kdf.to_pandas()</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>Parquet/ORC:</strong> ディスクベースの列形式（ストレージ用であり、インメモリ転送ではない）</li>
                <li><strong>Avro:</strong> 行ベースのシリアライゼーション形式で、JVM-Python相互運用性に最適化されていない</li>
            </ul>

            <h4>重要性:</h4>
            <p>Apache Arrowは以下の基盤となっています:</p>
            <ul>
                <li>SparkとPandas間の高速データ転送</li>
                <li>Pandas API on Sparkでのゼロコピー操作</li>
            </ul>
        `
    }
    ,
    {
        number: 9,
        domain: "Databricks ML",
        question: "MLflowでハイパーパラメータチューニングを実施する際、親実行と子実行を階層的に整理するために採用すべき方法は何ですか？",
        keyPoint: "nested=Trueで子実行を親実行の下にネストする",
        choices: [
            "各子実行を親実行と同じexperiment IDで開始する",
            "各ハイパーパラメータ値の組み合わせの子実行開始時にnested=Trueを指定する",
            "親実行のインデントされたコードブロック内でmlflow.start_run()を使用して各子実行を開始する",
            "Databricks Autologgingを有効にする",
            "チューニングプロセスの親実行開始時にnested=Trueを指定する"
        ],
        correctIndex: 1,
        explanation: `
            <p>MLflow実行を親子階層に整理するには（例: ハイパーパラメータチューニング用の1つの親実行と各ハイパーパラメータ組み合わせ用の子実行）、子実行開始時に<code>nested=True</code>フラグを使用します。</p>

            <div class="code-block">import mlflow

# 親実行を開始
with mlflow.start_run() as parent_run:
    # 親実行ロジック（例: チューニングセットアップ）

    # 親コンテキスト内で子実行を開始
    for params in hyperparameter_combinations:
        with mlflow.start_run(nested=True) as child_run:  # 重要: nested=True
            # 各子実行でハイパーパラメータとメトリクスをログ
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy)</div>

            <h4>なぜこれが機能するか:</h4>
            <ul>
                <li><strong>nested=True:</strong> 子実行がアクティブな親実行の下にネストされることを明示的に宣言</li>
                <li><strong>UI整理:</strong> MLflow実験ページで子実行が親の下にグループ化されて表示される</li>
            </ul>

            <h4>MLflow UIでの表示例:</h4>
            <div class="code-block">Parent Run (Tuning)
├── Child Run (Params 1)
├── Child Run (Params 2)
└── Child Run (Params 3)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>同じexperiment ID:</strong> 階層を作成せず、実行はフラットな兄弟関係になる</li>
                <li><strong>nested=Trueなしのインデント:</strong> nested=Trueが指定されない限り実行は兄弟関係</li>
                <li><strong>Databricks Autologging:</strong> ログを自動化するが、実行階層は制御しない</li>
                <li><strong>親実行でnested=True:</strong> nested=Trueは子実行にのみ有効（親実行はネストできない）</li>
            </ul>
        `
    }
    ,
    {
        number: 10,
        domain: "Hyperopt & Sparktail",
        question: "HyperoptでSparkTrialsがチューニングタスクをどのように分散し、ドライバノードとワーカーノードの役割は何ですか？",
        keyPoint: "トライアルはドライバで生成、ワーカーで評価される",
        choices: [
            "SparkTrialsはワーカーノードでトライアルを生成し、ドライバノードで評価する",
            "各トライアルはドライバノードでSparkジョブとして生成され、ワーカーノードで評価される",
            "SparkTrialsは集中型アプローチを使用し、全トライアルがドライバノードで評価される",
            "トライアルはワーカーノード上で独立して生成・評価される"
        ],
        correctIndex: 1,
        explanation: `
            <p>HyperoptでSparkTrialsがチューニングタスクを分散する仕組み:</p>

            <h4>1. ドライバノード:</h4>
            <ul>
                <li><strong>トライアル生成:</strong>
                    中央コーディネーターとして、Hyperoptの検索アルゴリズムを使用して新しいハイパーパラメータ構成（トライアル）を反復的に生成。
                    各トライアルに対して、単一タスクのSparkジョブを作成
                </li>
                <li><strong>ジョブ配布:</strong>
                    これらのSparkジョブを利用可能なワーカーノードに配布
                </li>
            </ul>

            <h4>2. ワーカーノード:</h4>
            <ul>
                <li><strong>トライアル評価:</strong>
                    各ワーカーノードはトライアルに対応するSparkジョブを受信。
                    ジョブ内のタスクを実行:
                    <ul>
                        <li>必要なデータとモデルコードをロード</li>
                        <li>指定されたハイパーパラメータ構成を使用してモデルを適合</li>
                        <li>検証セットでモデルの性能を評価</li>
                    </ul>
                </li>
                <li><strong>結果返却:</strong>
                    評価結果（損失値、ハイパーパラメータなど）をドライバノードに送信
                </li>
            </ul>

            <h4>重要ポイント:</h4>
            <ul>
                <li><strong>集中型トライアル生成:</strong> ドライバノードが新しいトライアルの生成を担当し、検索プロセスの調整と一貫性を確保</li>
                <li><strong>分散評価:</strong> 計算集約的なトライアル評価タスクはワーカーノード間で並列化され、クラスタの計算リソースを活用</li>
                <li><strong>非同期実行:</strong> トライアルは非同期で評価され、複数のトライアルが異なるワーカーノードで同時に実行可能でチューニングを加速</li>
                <li><strong>通信とフィードバック:</strong> ドライバノードがワーカーノードから結果を収集・集約し、この情報を利用して後続トライアルの生成を導き、有望なハイパーパラメータ領域へ検索を誘導</li>
            </ul>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>a):</strong> トライアル生成はワーカーノードではなくドライバノードで発生</li>
                <li><strong>c):</strong> 評価はドライバノードではなくワーカーノードに分散される</li>
                <li><strong>d):</strong> トライアル生成はワーカーノード上で独立ではなくドライバノードで集中管理</li>
            </ul>
        `
    }
    ,
    {
        number: 11,
        domain: "ML Workflows",
        question: "感染症の分類モデルで、可能な限り多くの症例を識別することが目標の場合、どの評価指標を使用すべきですか？",
        keyPoint: "Recallで陽性ケースの検出を最大化する",
        choices: [
            "Accuracy",
            "Precision",
            "Recall",
            "RMSE",
            "Area under the ROC curve"
        ],
        correctIndex: 2,
        explanation: `
            <p>可能な限り多くのケースを識別する（つまり、できるだけ多くの真の感染を捉える）という目標のためには、<strong>Recall</strong>（感度またはTrue Positive Rateとも呼ばれる）が最も適切な指標です。</p>

            <h4>なぜRecallか？</h4>
            <p>Recallは、モデルが正しく識別した実際の陽性（感染患者）の割合を測定します:</p>
            <ul>
                <li><strong>高いRecall = 少ない偽陰性（見逃された感染）</strong></li>
                <li><strong>医療では重要:</strong> 感染の見逃し（偽陰性）は、誤警報（偽陽性）よりはるかに悪い</li>
            </ul>

            <h4>例:</h4>
            <div class="code-block">100人の感染患者がいる場合:
- Recall = 90% → 90人の感染患者を正しくフラグ（10人を見逃す）
- Recall = 50% → 50人の感染患者を見逃す（医療リスクとして許容できない）</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>Accuracy:</strong>
                    不均衡データセットでは誤解を招く（例: 95%の患者が健康な場合、「常に健康」モデルは95%の精度を達成するが、感染を0件検出）
                </li>
                <li><strong>Precision:</strong>
                    フラグされたケースのうち正しいものの割合を測定（偽陽性の削減に焦点）。
                    ここでは適さない。優先事項は、いくつかの誤警報があったとしてもすべての感染を捕捉すること
                </li>
                <li><strong>RMSE:</strong>
                    回帰用であり、分類ではない
                </li>
                <li><strong>Area under the ROC curve (AUC-ROC):</strong>
                    しきい値全体でのモデル性能を評価するが、すべての陽性を捕捉するために直接最適化しない
                </li>
            </ul>

            <h4>実用的なヒント:</h4>
            <ul>
                <li>不均衡データセットには適合-再現率曲線（ROCではなく）を使用</li>
                <li>Recallを上げるために低い決定しきい値を設定（例: 30%の確率でも「感染」と分類）</li>
            </ul>
        `
    }
    ,
    {
        number: 12,
        domain: "Spark ML Basics",
        question: "Databricksで複数のノートブックを使用する共同プロジェクトにおいて、チームメンバーの変更が互いに上書きされないようにするには？",
        keyPoint: "Gitバージョン管理で変更競合を防ぐ",
        choices: [
            "ノートブック設定で「Auto-Save」を有効にする",
            "Databricksワークスペース内でGitを使用したバージョン管理を実装する",
            "共通のユーザー名とパスワードを使用してノートブックアクセスを共有する",
            "複数ユーザーによる同時編集を防ぐために「Lock」機能を使用する"
        ],
        correctIndex: 1,
        explanation: `
            <p>競合を防ぎ、変更を上書きせずに共同作業を行うには、Databricksノートブックと<strong>Gitバージョン管理</strong>を統合すべきです。</p>

            <h4>Gitバージョン管理の利点:</h4>
            <ul>
                <li><strong>ブランチング:</strong> 各チームメンバーが独自のブランチで作業</li>
                <li><strong>マージリクエスト:</strong> mainブランチにマージする前に変更をレビュー</li>
                <li><strong>履歴追跡:</strong> 必要に応じて以前のバージョンにロールバック</li>
            </ul>

            <h4>DatabricksでGitをセットアップする手順:</h4>
            <ol>
                <li><strong>Gitに接続:</strong>
                    User Settings → Git Integrationに移動し、GitHub/GitLab/Bitbucketアカウントをリンク
                </li>
                <li><strong>リポジトリをクローン:</strong>
                    Databricks UIを使用してワークスペースにリポジトリをクローン
                </li>
                <li><strong>変更をコミット・プッシュ:</strong>
                    説明的なメッセージで定期的に変更をコミットし、リモートリポジトリにプッシュ
                </li>
            </ol>

            <div class="code-block"># Gitワークフローの例
1. ユーザーAがfeatureブランチを作成 → ノートブックを編集 → コミット/プッシュ
2. ユーザーBが変更をレビュー → マージリクエストを承認</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>「Auto-Save」:</strong>
                    編集を自動保存するが、複数ユーザーが同じノートブックを編集する場合の競合を防がない
                </li>
                <li><strong>共有ユーザー名/パスワード:</strong>
                    非常に安全でなく、競合解決を提供しない
                </li>
                <li><strong>「Lock」機能:</strong>
                    Databricksにはノートブック用の組み込みロック機能がない
                </li>
            </ul>

            <h4>ベストプラクティス:</h4>
            <ul>
                <li>レビューなしに共有ノートブックを直接編集しない</li>
                <li>誰が何をいつ変更したかの完全な監査証跡を維持</li>
            </ul>
        `
    }
    ,
    {
        number: 13,
        domain: "Hyperopt & Sparktail",
        question: "GPU対応クラスタでSparkTrialsを使用する際、並列性をどのように設定すべきですか？",
        keyPoint: "GPU数に基づいて並列性を設定しタスク競合を回避する",
        choices: [
            "GPUクラスタは最大並列性を使用するため、特定の設定は不要",
            "GPUクラスタは最適な並列性のためにノードごとに複数のexecutorスレッドを使用する",
            "GPU対応インスタンスの数に基づいて並列性を設定し、Sparkタスク間の競合を回避する",
            "GPUクラスタはSparkTrials設定に基づいて並列性を自動調整する"
        ],
        correctIndex: 2,
        explanation: `
            <p>GPU対応クラスタでSparkTrialsを使用する際の重要な考慮事項:</p>

            <h4>Executorスレッド設定:</h4>
            <ul>
                <li><strong>CPU vs GPU:</strong>
                    CPUクラスタは通常、CPU使用率を最大化するためにワーカーノードごとに複数のexecutorスレッドを使用。
                    GPUクラスタは、同じGPUにアクセスしようとするSparkタスク間の競合を防ぐため、ノードごとに1つのexecutorスレッドのみを使用することが多い
                </li>
            </ul>

            <h4>並列性の調整:</h4>
            <ul>
                <li><strong>最大並列性の削減:</strong>
                    GPUクラスタのノードごと単一executorスレッド設定は、CPUクラスタと比較して達成可能な最大並列性が低いことを意味
                </li>
                <li><strong>手動設定:</strong>
                    GPU対応インスタンスの数に合わせて並列性を手動で設定し、リソースを効果的に活用して競合を回避する必要がある
                </li>
            </ul>

            <div class="code-block"># 4つのGPU対応ノードを持つクラスタを想定
spark = SparkSession.builder.appName("my_app").config(
    "spark.executor.instances", "4",  # executor数（GPUごとに1つ）
    "spark.executor.cores", "1",      # executorごとのコア数（GPUクラスタでは通常1）
    "spark.default.parallelism", "4"  # 並列タスク数（GPU数に応じて設定）
).getOrCreate()</div>

            <h4>追加の考慮事項:</h4>
            <ul>
                <li><strong>インスタンスタイプ:</strong> ワークロードの要件（メモリ、計算能力など）に合うGPUインスタンスタイプを選択</li>
                <li><strong>GPU互換性:</strong> ライブラリとフレームワークがGPU加速と互換性があることを確認</li>
                <li><strong>リソース監視:</strong> GPU使用率を監視して潜在的なボトルネックを特定し、必要に応じて設定を調整</li>
            </ul>
        `
    }
    ,
    {
        number: 14,
        domain: "Spark ML Basics",
        question: "データサイエンティストのチームが異なる機械学習実験のパフォーマンスを追跡・比較したい場合、どのMLflowコンポーネントが役立ちますか？",
        keyPoint: "MLflow Tracking Serverで実験を追跡・比較する",
        choices: [
            "MLflow Tracking Server",
            "MLlib CrossValidator",
            "MLflow REST API",
            "Databricks Jobs"
        ],
        correctIndex: 0,
        explanation: `
            <p><strong>MLflow Tracking Server</strong>は、機械学習実験を追跡、ログ、比較するために設計されたコアコンポーネントです。</p>

            <h4>主な機能:</h4>
            <ul>
                <li><strong>パラメータ、メトリクス、アーティファクトのログ:</strong>
                    ハイパーパラメータ、評価メトリクス（例: accuracy、RMSE）、出力ファイル（例: モデル、プロット）を記録
                </li>
                <li><strong>実験の比較:</strong>
                    統一されたUIで実行結果を表示。平行座標プロットとメトリクストレンドを含む
                </li>
                <li><strong>コラボレーション:</strong>
                    全チームメンバーが実験結果に共有アクセス
                </li>
            </ul>

            <div class="code-block">import mlflow

# 実験を開始
mlflow.set_experiment("fraud_detection")

# パラメータとメトリクスをログ
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    # モデルをログ
    mlflow.sklearn.log_model(model, "model")</div>

            <h4>主要機能:</h4>
            <ul>
                <li>✅ 集中UI: 実行を視覚的に比較</li>
                <li>✅ 再現性: すべての依存関係とコードバージョンをキャプチャ</li>
            </ul>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>MLlib CrossValidator:</strong>
                    Spark MLでのハイパーパラメータチューニングに使用され、実験追跡ではない
                </li>
                <li><strong>MLflow REST API:</strong>
                    Tracking Serverへのプログラマティックアクセスを許可するが、比較のための主要ツールではない
                </li>
                <li><strong>Databricks Jobs:</strong>
                    ワークフローをスケジュール・実行するが、実験を追跡・比較しない
                </li>
            </ul>

            <h4>重要性:</h4>
            <p>共同ML実験追跡には、MLflow Tracking Serverを使用して:</p>
            <ul>
                <li>実行をログし視覚化</li>
                <li>モデルを包括的に比較</li>
                <li>チーム全体で結果を共有</li>
            </ul>
        `
    }
    ,
    {
        number: 15,
        domain: "Spark ML",
        question: "Databricksで、estimatorの.fit()メソッドに必要なスカラー値の列をベクトル型の列に変換するために使用されるコンポーネントは？",
        keyPoint: "VectorAssemblerでスカラー列をベクトルに変換する",
        choices: [
            "VectorScaler",
            "VectorConverter",
            "VectorAssembler",
            "VectorTransformer"
        ],
        correctIndex: 2,
        explanation: `
            <p>Databricks（およびApache Spark MLlib）では、<strong>VectorAssembler</strong>がスカラー列（例: 数値特徴）を単一のベクトル列に変換するために使用されます。これはほとんどのML estimator（例: LinearRegression、RandomForestClassifier）の.fit()メソッドに必要です。</p>

            <h4>動作原理:</h4>
            <ul>
                <li><strong>入力:</strong> 複数のスカラー列（例: age、income、score）</li>
                <li><strong>出力:</strong> 単一のベクトル列（例: features）。各ベクトルは行のスカラー値を結合</li>
            </ul>

            <div class="code-block">from pyspark.ml.feature import VectorAssembler

# スカラー列を持つサンプルDataFrame
df = spark.createDataFrame([
    (25, 50000, 3.5),
    (30, 80000, 4.2)
], ["age", "income", "score"])

# 列をベクトルに結合
assembler = VectorAssembler(
    inputCols=["age", "income", "score"],
    outputCol="features"
)

# DataFrameを変換
df_vector = assembler.transform(df)
df_vector.show()</div>

            <h4>出力:</h4>
            <div class="code-block">+---+------+-----+------------------+
|age|income|score|          features|
+---+------+-----+------------------+
| 25| 50000|  3.5|[25.0,50000.0,3.5]|
| 30| 80000|  4.2|[30.0,80000.0,4.2]|
+---+------+-----+------------------+</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>VectorScaler:</strong>
                    ベクトル列をスケーリング（例: 正規化）するが、スカラーからベクトルを作成しない
                </li>
                <li><strong>VectorConverter:</strong>
                    Spark MLlibには存在しない
                </li>
                <li><strong>VectorTransformer:</strong>
                    一般的な用語であり、特定のSparkクラスではない
                </li>
            </ul>

            <h4>重要性:</h4>
            <p>VectorAssemblerを使用して:</p>
            <ul>
                <li>✅ スカラー列をMLモデル用のベクトル形式に変換</li>
                <li>✅ .fit()のようなestimatorのためのデータを準備</li>
            </ul>
        `
    }
    ,
    {
        number: 16,
        domain: "Spark ML",
        question: "PySparkの代わりにPandas API on Sparkを使用する潜在的な欠点は何ですか？",
        keyPoint: "内部フレーム変換により計算時間が増加する",
        choices: [
            "分散コンピューティングのサポートが制限されている",
            "非効率的なデータ構造",
            "内部フレーム変換による計算時間の増加",
            "PySparkと比較して機能が制限されている"
        ],
        correctIndex: 2,
        explanation: `
            <p>PySparkの代わりにPandas API on Spark（Koalas）を使用する主な欠点は、SparkとPandas形式間の<strong>内部フレーム変換のオーバーヘッド</strong>です。</p>

            <h4>変換オーバーヘッド:</h4>
            <ul>
                <li>Pandas API on Sparkは、Pandas風の構文を内部でSpark操作に変換して動作</li>
                <li>各操作で、Spark DataFrame（分散）とPandas DataFrame（シングルノード）間の変換が必要になる場合があり、レイテンシーが追加される</li>
            </ul>

            <h4>パフォーマンスへの影響:</h4>
            <ul>
                <li><strong>小〜中規模データセット:</strong> Pandas構文の利便性がコストを上回る</li>
                <li><strong>大規模データセット:</strong> 頻繁な変換がネイティブPySparkと比較して計算を遅くする可能性がある</li>
            </ul>

            <div class="code-block">import databricks.koalas as ks

# Pandas API on Spark（隠れた変換）
kdf = ks.DataFrame(...)
result = kdf.groupby("col1").mean()  # Sparkに変換、その後Pandas風形式に戻す</div>

            <h4>他の選択肢があまり関連しない理由:</h4>
            <ul>
                <li><strong>分散コンピューティングのサポートが制限:</strong>
                    不正確。Pandas API on SparkはSparkの分散コンピューティングを完全に活用
                </li>
                <li><strong>非効率的なデータ構造:</strong>
                    部分的に真だが、主な問題は構造自体ではなく変換
                </li>
                <li><strong>機能が制限:</strong>
                    一部のPySpark機能が欠けている可能性があるが、APIはほとんどの一般的なユースケースをカバー
                </li>
            </ul>

            <h4>使い分け:</h4>
            <p><strong>Pandas API on Sparkを使用:</strong></p>
            <ul>
                <li>✅ ビッグデータでPandas風構文を使用</li>
                <li>⚠️ トレードオフ: 使いやすさのために変換オーバーヘッドを受け入れる</li>
            </ul>

            <p><strong>最大性能にはネイティブPySparkを使用:</strong></p>
            <ul>
                <li>非常に大規模なデータセットを扱う場合</li>
                <li>Spark最適化の低レベル制御が必要な場合</li>
            </ul>
        `
    }
    ,
    {
        number: 17,
        domain: "ML Workflows",
        question: "MLflow Model Registryで既存のモデルに新しいバージョンを登録する場合、registered_model_nameパラメータは何を意味しますか？",
        keyPoint: "既存モデルの新バージョンを自動登録する",
        choices: [
            "mlflow.register_modelの呼び出しの必要性を排除する",
            "MLflow Model Registryに新しいモデルを記録する",
            "MLflow実験でログされたモデルの名前を表す",
            "MLflow Model Registryで既存モデルの新バージョンを登録する",
            "MLflow実験でのRunの名前を示す"
        ],
        correctIndex: 3,
        explanation: `
            <p><code>mlflow.sklearn.log_model()</code>で<code>registered_model_name=model_name</code>が指定され、モデル（model_name）がMLflow Model Registryに既に存在する場合:</p>

            <ul>
                <li>既存の<code>model_name</code>の下に新しいバージョンが自動的に作成される</li>
                <li>モデルは現在のMLflow実行でアーティファクトとしてログされる</li>
                <li><code>mlflow.register_model()</code>への追加呼び出しは不要</li>
            </ul>

            <h4>主要な動作:</h4>
            <ul>
                <li>✅ <strong>バージョニング:</strong> 各実行は新しいバージョンを作成（例: Version 2、Version 3）</li>
                <li>✅ <strong>重複なし:</strong> 重複するモデルエントリの作成を回避。既存のmodel_nameを使用</li>
            </ul>

            <div class="code-block">import mlflow

# "model_name"の下に新しいバージョンをログ・登録
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    registered_model_name="model_name"  # 既存モデルにリンク
)

# MLflow UIでの結果:
# model_name → Version 2（Version 1が既に存在する場合）</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>「mlflow.register_modelの呼び出しを排除」:</strong>
                    真だが、これは副作用であり、registered_model_nameの主要な目的ではない
                </li>
                <li><strong>「新しいモデルを記録」:</strong>
                    model_nameが存在しない場合のみ真。ここでは既存モデルにバージョンを追加
                </li>
                <li><strong>「実験でのモデル名」:</strong>
                    実験は実行を追跡するが、registered_model_nameはModel Registryに紐付く
                </li>
                <li><strong>「Runの名前」:</strong>
                    実行名は別途設定（例: <code>mlflow.start_run(run_name="...")</code>）
                </li>
            </ul>

            <h4>重要性:</h4>
            <p><code>registered_model_name</code>を使用して:</p>
            <ul>
                <li>✅ Model Registryの既存名でモデルを自動登録</li>
                <li>✅ 手動ステップなしでバージョニングを有効化</li>
            </ul>
        `
    }
    ,
    {
        number: 18,
        domain: "Pandas API on Spark",
        question: "Pandas API on Sparkの目的は何ですか？",
        keyPoint: "Pandasの機能をビッグデータに拡張する",
        choices: [
            "データ分析タスクでPySparkを置き換える",
            "Pythonにスケーラブルなデータ構造を提供する",
            "Pandasの機能をビッグデータに拡張する",
            "Apache Spark用の新しいPythonパッケージを導入する"
        ],
        correctIndex: 2,
        explanation: `
            <p>Pandas API on Spark（Koalasとも呼ばれる）は、PandasとPySpark間のギャップを埋めるように設計されており、ユーザーは:</p>

            <ul>
                <li>Sparkによる分散コンピューティングが必要な大規模データセットでPandas風の構文を使用できる</li>
                <li>コードを書き換えずにPandasワークフローをビッグデータにスケール可能</li>
            </ul>

            <h4>主要機能:</h4>
            <ul>
                <li>✅ <strong>使い慣れたインターフェース:</strong>
                    <code>df.groupby()</code>、<code>df.pivot()</code>、<code>df.plot()</code>などのメソッドがPandasと同じように動作
                </li>
                <li>✅ <strong>分散実行:</strong>
                    内部では、操作がSparkクラスタ全体で並列化される
                </li>
                <li>✅ <strong>シームレスな統合:</strong>
                    PandasとSpark DataFrames間で変換可能（例: <code>to_pandas()</code>、<code>to_spark()</code>）
                </li>
            </ul>

            <div class="code-block">import databricks.koalas as ks

# Koalas DataFrameを作成（ビッグデータにスケール）
kdf = ks.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

# Pandas風の操作を使用
result = kdf.groupby("A").sum()  # Sparkで実行</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>「PySparkを置き換える」:</strong>
                    Koalasは補完するものであり、置き換えない。低レベル制御にはPySparkが依然として必要
                </li>
                <li><strong>「スケーラブルなデータ構造」:</strong>
                    部分的に真だが、主要目標は新しい構造の導入ではなくPandas互換性
                </li>
                <li><strong>「新しいPythonパッケージ」:</strong>
                    KoalasはSparkに基づいて構築されているが、SparkのネイティブAPIの置き換えではなくPandasユーザーに焦点
                </li>
            </ul>

            <h4>使用場面:</h4>
            <ul>
                <li>✅ ビッグデータでPandas構文を活用</li>
                <li>✅ シングルノードから分散へスケール時のコード書き換えを回避</li>
            </ul>
        `
    }
    ,
    {
        number: 19,
        domain: "ML Workflows",
        question: "8つの評価を8つの計算ノードで実行するハイパーパラメータ最適化で精度の一貫した向上が見られない場合、どの変更が精度向上に役立ちますか？",
        keyPoint: "ベイズ最適化など賢いアルゴリズムに切り替える",
        choices: [
            "計算ノード数を評価数の半分以下に調整する",
            "チューニングプロセスを支援する反復最適化アルゴリズムを切り替える",
            "計算ノード数を評価数の2倍以上に調整する",
            "計算ノードと評価の両方の数を大幅に減らす",
            "計算ノードと評価の両方の数を大幅に増やす"
        ],
        correctIndex: 1,
        explanation: `
            <p>核となる問題は、反復最適化アルゴリズム（例: ランダムサーチ、グリッドサーチ）がハイパーパラメータサーチを効果的に精度向上に導いていないことです。一貫した改善がないことは、現在のアルゴリズムが過去の評価から学習していないことを示唆しています。</p>

            <h4>なぜアルゴリズムを切り替えるか？</h4>
            <p><strong>現在の問題:</strong></p>
            <ul>
                <li>アルゴリズムがハイパーパラメータを独立に評価（ノード間で知識共有なし）</li>
                <li>例: ランダムサーチは最適領域を見逃す可能性がある</li>
            </ul>

            <p><strong>より良い代替案:</strong></p>
            <ul>
                <li><strong>ベイズ最適化（例: Hyperopt）:</strong>
                    過去の評価を使用して有望なハイパーパラメータに焦点を当てる。
                    探索（新しい組み合わせを試す）と活用（良いものを改良）のバランスをとる
                </li>
                <li><strong>TPE（Tree-structured Parzen Estimator）:</strong>
                    過去の結果に基づいて良いハイパーパラメータの確率をモデル化
                </li>
            </ul>

            <div class="code-block">from hyperopt import fmin, tpe, hp, SparkTrials

# 検索空間と目的関数を定義
space = {
    'learning_rate': hp.loguniform('lr', -5, 0),
    'max_depth': hp.choice('depth', range(1, 10))
}

def objective(params):
    # モデルを訓練・評価
    accuracy = train_and_evaluate(params)
    return -accuracy  # Hyperoptは最小化するため精度を負にする

# TPEでよりスマートな検索を使用
best = fmin(objective, space, algo=tpe.suggest, max_evals=8, trials=SparkTrials())</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>計算ノードの調整（少ない/多い）:</strong>
                    アルゴリズムが評価から学習できないという問題に対処しない。
                    より多くのノードは非効率的な検索を並列化するだけ
                </li>
                <li><strong>小/大ノード/評価数:</strong>
                    速度に影響する可能性があるが、検索効率は改善しない
                </li>
            </ul>

            <h4>重要性:</h4>
            <p>ハイパーパラメータチューニング中に精度を向上させるには:</p>
            <ul>
                <li>✅ よりスマートなアルゴリズム（例: ベイズ最適化、TPE）に切り替える</li>
                <li>✅ 計算ノードはそのまま（8評価に8ノードで問題なし）</li>
            </ul>
        `
    }
    ,
    {
        number: 20,
        domain: "Databricks ML",
        question: "多数の特徴を持つ大規模データセットを含むSparkジョブを最適化する際、メモリ使用量を最小化するための高度な最適化技術は？",
        keyPoint: "小さいDataFrameをbroadcastして結合を最適化する",
        choices: [
            "cacheメソッドで中間DataFrameをメモリに永続化する",
            "結合操作で小さいDataFrameにbroadcastヒントを使用する",
            "Spark executorメモリのサイズを増やす",
            "Sparkクラスタ設定で自動スキーマ推論を有効にする"
        ],
        correctIndex: 1,
        explanation: `
            <p>大規模データセットを持つSparkジョブを最適化し、メモリ使用量を最小化するには、1つのDataFrameがメモリに収まるほど小さい場合の効率的な結合のための<strong>broadcastヒント</strong>が強力な技術です。</p>

            <h4>動作原理:</h4>
            <p><code>broadcast</code>は、小さいDataFrameをすべてのワーカーノードに送信することをSparkに強制し、大きいDataFrameのコストのかかるシャッフルを回避します。</p>

            <div class="code-block">from pyspark.sql.functions import broadcast

# broadcastで効率的な結合
large_df.join(broadcast(small_df), "join_key")</div>

            <h4>利点:</h4>
            <ul>
                <li><strong>シャッフルオーバーヘッドを削減:</strong> 小さいDataFrameのネットワークトラフィックを排除</li>
                <li><strong>メモリを節約:</strong> ワーカーは小さいDataFrameを（ノードごとに）1回ロードし、大きなデータを再分散しない</li>
            </ul>

            <h4>使用時期:</h4>
            <p>小さいDataFrameがexecutorメモリに完全に収まる必要がある（通常<100MBだが、クラスタ設定に依存）</p>

            <h4>他の選択肢が最適でない理由:</h4>
            <ul>
                <li><strong>cache():</strong>
                    中間DataFrameの永続化は役立つが、結合の非効率性には対処しない。
                    キャッシュするデータが多すぎるとOOMのリスク
                </li>
                <li><strong>Executorメモリを増やす:</strong>
                    応急処置的解決策。最適化されていない結合のような非効率的な操作を修正しない
                </li>
                <li><strong>自動スキーマ推論:</strong>
                    手動スキーマ定義を減らすが、メモリ/性能に直接影響しない
                </li>
            </ul>

            <h4>ベストプラクティス:</h4>
            <div class="code-block"># broadcastで最適な結合
result = large_df.join(broadcast(small_lookup_df), "id")

# 避けるべき（シャッフルを引き起こす）
result = large_df.join(small_lookup_df, "id")  # broadcastヒントなし</div>

            <p>メモリ効率的なSparkジョブのために:</p>
            <ul>
                <li>✅ 結合で小さいDataFrameをbroadcast</li>
                <li>✅ 以下と組み合わせる:
                    <ul>
                        <li>大きいDataFrameのパーティショニング</li>
                        <li>再利用されるDataFrameに対して選択的に<code>cache()</code>を使用</li>
                    </ul>
                </li>
            </ul>
        `
    }
    ,
    {
        number: 21,
        domain: "Feature Store",
        question: "fs.score_batch()でバッチ予測を実行する際、コードが正常に予測を実行する条件は何ですか？",
        keyPoint: "必要な全特徴がSpark DataFrameに存在する必要がある",
        choices: [
            "このコードはどの状況でも予測を達成しない",
            "model_uriのモデルがorder_idのみを特徴として使用する場合",
            "Feature Storeの特徴セットがmodel_uriのモデルと共に登録された場合",
            "model_uriのモデルが使用するすべての特徴が1つのFeature Storeテーブルに収容されている場合",
            "model_uriのモデルが使用するすべての特徴がPySparkセッション内のSpark DataFrameに存在する場合"
        ],
        correctIndex: 4,
        explanation: `
            <p>Databricksが提供する<code>fs.score_batch()</code>メソッドは、指定されたモデルでDataFrameをバッチモードでスコアリングするために使用されます。このメソッドは、提供されたDataFrameがモデルが予測を生成するために必要なすべての特徴を保持していることを前提としています。</p>

            <h4>正解の理由:</h4>
            <p>ここでは、DataFrameは<code>batch_df</code>で、<code>order_id</code>という1つの特徴のみを含んでいます。したがって、コードブロックは、<code>model_uri</code>のモデルが使用するすべての特徴がPySparkセッション内のSpark DataFrameで利用可能な場合にのみ、望ましい予測を実行します。</p>

            <div class="code-block"># 正しい使用例: 必要な全特徴がDataFrameに存在
predictions = fs.score_batch(
    model_uri="models:/my_model/Production",
    df=batch_df  # order_idと他の必要な特徴を含む
)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>A: 「コードは決して機能しない」:</strong>
                    このオプションは、コードが決して機能しないことを示唆するが、特定の条件下では機能可能なため、最良の選択ではない
                </li>
                <li><strong>B: 「モデルがorder_idのみを使用」:</strong>
                    このオプションは、モデルがorder_idのみを特徴として使用すると仮定するが、正確な予測には追加の特徴が必要な可能性があるため、制限的すぎる
                </li>
                <li><strong>C: 「Feature Store特徴セットが登録された」:</strong>
                    バッチスコアの使用は、Feature Storeで利用可能な特徴セットを使用してモデルが生成されることを意味し、これは自動的に登録される
                </li>
                <li><strong>D: 「1つのFeature Storeテーブルに全特徴」:</strong>
                    すべての必要な特徴が1つのFeature Storeテーブルにある場合は機能する可能性があるが、必須ではない。特徴は異なるテーブルまたは外部ソースから取得可能
                </li>
            </ul>

            <h4>重要なポイント:</h4>
            <ul>
                <li><code>batch_df</code>にはモデルが必要とするすべての特徴が含まれている必要がある</li>
                <li>特徴は単一のテーブルまたは複数のソースから来ることができる</li>
                <li>コードが機能するための鍵は、必要なすべての特徴がDataFrameに存在することである</li>
            </ul>
        `
    }
    ,
    {
        number: 22,
        domain: "Scaling ML Models",
        question: "分散コンピューティング環境で地理空間データを処理する際、効率的なインデックス作成と検索を可能にする技術は何ですか？",
        keyPoint: "Spatial Indexingで地理空間データを効率的に検索する",
        choices: [
            "Spatial Partitioning",
            "Geospatial Clustering",
            "Spatial Indexing",
            "Geospatial Replication"
        ],
        correctIndex: 2,
        explanation: `
            <p><strong>Spatial Indexing（空間インデックス）</strong>は、分散コンピューティング環境（例: Apache Spark）で地理空間データの効率的なインデックス作成と検索を可能にする技術です。空間データ（点、ポリゴンなど）を構造化されたインデックスに整理し、迅速なクエリと分析を可能にします。</p>

            <h4>Spatial Indexingの主要機能:</h4>
            <ul>
                <li><strong>最適化されたクエリ:</strong>
                    範囲検索（「10km以内のすべての点を見つける」）や最近傍検索などの操作を加速
                </li>
                <li><strong>分散フレンドリー:</strong>
                    インデックス（例: R-tree、QuadTree、GeoHash）はクラスタ全体でパーティション化可能
                </li>
                <li><strong>Sparkでのサポート:</strong>
                    Sedona（旧GeoSpark）のようなライブラリがPySpark用の空間インデックスを実装
                </li>
            </ul>

            <div class="code-block">from sedona.core.SpatialRDD import PointRDD
from sedona.core.enums import IndexType

# 点から空間RDDを作成
spatial_rdd = PointRDD(sc, "path/to/points.csv")

# R-treeインデックスを構築
spatial_rdd.buildIndex(IndexType.RTREE, True)  # True = メモリにインデックスを保存

# 効率的な空間クエリ
result = spatial_rdd.rangeQuery(query_polygon)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>Spatial Partitioning:</strong>
                    データを地理的に分割するが、高速検索を可能にしない（インデックスが行う）
                </li>
                <li><strong>Geospatial Clustering:</strong>
                    類似データポイントをグループ化（例: DBSCAN）するが、検索を最適化しない
                </li>
                <li><strong>Geospatial Replication:</strong>
                    冗長性のためにデータをコピーするが、インデックス/クエリとは無関係
                </li>
            </ul>

            <h4>重要性:</h4>
            <p>分散システムでの効率的な地理空間分析には:</p>
            <ul>
                <li>✅ 空間インデックス（R-tree、QuadTree、GeoHash）を使用</li>
                <li>✅ Spark上でSedonaやGeoPandasなどのライブラリを活用</li>
            </ul>
        `
    }
    ,
    {
        number: 23,
        domain: "Hyperopt & Sparktail",
        question: "SparkTrialsでワーカーからログを記録する際、目的関数で実行を明示的に管理する必要がありますか？",
        keyPoint: "SparkTrialsが実行管理を自動で処理する",
        choices: [
            "はい、実行を明示的に管理する必要がある",
            "いいえ、SparkTrialsが実行管理を自動的に処理する",
            "目的関数でMLlibまたはHorovodを使用する場合のみ",
            "目的関数の複雑さに依存する"
        ],
        correctIndex: 1,
        explanation: `
            <p>Hyperoptで分散ハイパーパラメータチューニングにSparkTrialsを使用する場合、<strong>実行管理は完全に自動化</strong>されています。</p>

            <h4>自動ログ記録:</h4>
            <ul>
                <li>SparkTrialsは、ワーカーノード上の各トライアル（ハイパーパラメータ評価）に対してMLflow実行を自動的に作成</li>
                <li>目的関数で手動で<code>mlflow.start_run()</code>を呼び出す必要はない</li>
            </ul>

            <h4>ワーカーレベルの追跡:</h4>
            <ul>
                <li>各トライアルのメトリクス、パラメータ、アーティファクトは、親Hyperopt実行にリンクされたネストされた実行の下にログされる</li>
            </ul>

            <div class="code-block">from hyperopt import fmin, tpe, SparkTrials

def objective(params):
    # 手動のMLflow実行は不要！
    accuracy = train_and_evaluate(params)
    return {"loss": -accuracy, "status": "ok"}  # SparkTrialsによって自動ログ

spark_trials = SparkTrials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=50, trials=spark_trials)</div>

            <h4>集中UI:</h4>
            <p>すべてのネストされた実行は、簡単な比較のためにMLflow UIの親実行の下に表示されます。</p>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>「明示的に管理する必要がある」:</strong>
                    プロセスを過度に複雑にする。SparkTrialsが処理する
                </li>
                <li><strong>「MLlib/Horovodの場合のみ」:</strong>
                    無関係。SparkTrialsの自動化はどの目的関数でも機能する
                </li>
                <li><strong>「複雑さに依存」:</strong>
                    誤り。複雑さに関係なく自動化は一貫している
                </li>
            </ul>

            <h4>重要性:</h4>
            <p>SparkTrialsでは:</p>
            <ul>
                <li>✅ 目的関数で手動の実行管理は不要</li>
                <li>✅ すべてのメトリクス/アーティファクトは自動ログされる</li>
            </ul>

            <p><strong>注意:</strong> カスタムログ記録（例: 追加のアーティファクト）の場合は、<code>start_run()</code>なしで目的関数内で<code>mlflow.log_*</code>を使用</p>
        `
    }
    ,
    {
        number: 24,
        domain: "Pandas API on Spark",
        question: "pandas-on-SparkでDataFrame.transform()とDataFrame.apply()の主な違いは何ですか？",
        keyPoint: "transformは同じ長さ、applyは任意の長さを返せる",
        choices: [
            "transformは関数が入力と同じ長さを返すことを要求し、applyは任意の長さを許可する",
            "applyは関数が入力と同じ長さを返すことを要求し、transformは任意の長さを許可する",
            "transformとapplyの両方が関数が入力と同じ長さを返すことを要求する",
            "transformとapplyの両方が出力に任意の長さを許可する"
        ],
        correctIndex: 0,
        explanation: `
            <h4>主な違い:</h4>

            <p><strong>transform():</strong></p>
            <ul>
                <li>pandas-on-Spark DataFrameに要素ごとに関数を適用</li>
                <li><strong>関数は入力と同じ長さのSeriesまたはDataFrameを返す必要がある</strong></li>
                <li>元のDataFrameのインデックスと列ラベルを保持</li>
            </ul>

            <p><strong>apply():</strong></p>
            <ul>
                <li>pandas-on-Spark DataFrameの指定された軸（行または列）に沿って関数を適用</li>
                <li><strong>関数はスカラー、Series、DataFrameを含む任意の長さの値を返すことができる</strong></li>
                <li>関数の出力に応じて、インデックスまたは列ラベルが保持されない場合がある</li>
            </ul>

            <h4>使い分け:</h4>
            <ul>
                <li><strong>transform():</strong> 各要素または行を同じ長さの対応する出力に直接変換し、DataFrameの構造を維持する関数を適用する場合に使用</li>
                <li><strong>apply():</strong> 出力長により柔軟性が必要な場合、またはDataFrameの形状や構造を変更する可能性のある集計、削減、操作を実行する場合に使用</li>
            </ul>

            <div class="code-block">import pandas as pd

# サンプルDataFrame
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

# transform()の例（同じ長さの出力）
def double_values(x):
    return x * 2

df_transformed = df.transform(double_values)
# 出力: 列AとBが2倍になったDataFrame

# apply()の例（任意の長さの出力）
def sum_row(row):
    return row.sum()

df_applied = df.apply(sum_row, axis=1)
# 出力: 各行の合計を含むSeries</div>

            <h4>重要ポイント:</h4>
            <ul>
                <li>DataFrameの構造を維持する要素ごとの操作には<code>transform()</code>を選択</li>
                <li>DataFrameの形状を変更する可能性のあるより柔軟な操作には<code>apply()</code>を選択</li>
                <li>適切なメソッドを選択する際は、期待される出力長に注意</li>
            </ul>
        `
    }
    ,
    {
        number: 25,
        domain: "Databricks ML",
        question: "Databricks環境でSparkジョブの性能を最適化し、変換中のデータシャッフルを削減するために考慮すべきことは？",
        keyPoint: "repartitionメソッドでパーティション数を制御する",
        choices: [
            "DataFrameのパーティション数を増やす",
            "repartitionメソッドを使用してパーティション数を制御する",
            "Sparkクラスタのワーカーノード数を減らす",
            "Databricksクラスタ設定で自動最適化を有効にする"
        ],
        correctIndex: 1,
        explanation: `
            <p>データシャッフル（Sparkでコストのかかる操作）を削減するには、チームは以下に基づいて戦略的にDataFrameを再パーティション化すべきです:</p>

            <h4>パーティションサイズ:</h4>
            <ul>
                <li>パーティションあたり約100-200MBを目標（小さすぎる多数または少なすぎる大きなパーティションを避ける）</li>
                <li><code>df.repartition(n)</code>を使用してデータを均等に分散</li>
            </ul>

            <h4>パーティションキー:</h4>
            <ul>
                <li>結合/group-byキーで再パーティション化してシャッフルを最小化:
                    <div class="code-block">df.repartition("customer_id")  # customer_idでの結合のシャッフルを最小化</div>
                </li>
            </ul>

            <h4>なぜこれが機能するか:</h4>
            <ul>
                <li>関連データを同じワーカーに共存させ、ノード間転送を削減</li>
                <li>並列性のバランスをとる（歪んだパーティションを避ける）</li>
            </ul>

            <div class="code-block"># シャッフルを避けるために結合前に最適化
df1 = df1.repartition("join_key")
df2 = df2.repartition("join_key")
result = df1.join(df2, "join_key")  # シャッフルなし！</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>パーティションをむやみに増やす:</strong>
                    より多くの小さなタスクを引き起こし、オーバーヘッドが増加する可能性
                </li>
                <li><strong>ワーカーノードを減らす:</strong>
                    並列性を減らすが、シャッフルの非効率性には対処しない
                </li>
                <li><strong>自動最適化:</strong>
                    役立つ（例: Delta Lake自動コンパクション）が、手動パーティショニングの代わりにはならない
                </li>
            </ul>

            <h4>重要性:</h4>
            <p>シャッフルを最小化するには:</p>
            <ul>
                <li>✅ 結合/group-byキーで再パーティション化</li>
                <li>✅ パーティションサイズを監視（<code>df.rdd.getNumPartitions()</code> + <code>df.rdd.glom().map(len).collect()</code>）</li>
            </ul>
        `
    }
    ,
    {
        number: 26,
        domain: "ML Workflows",
        question: "Spark MLでImputerを使用して欠損値を中央値で補完する際、コードが正しく動作しない理由は何ですか？",
        keyPoint: "fit()でImputerModelを作成する必要がある",
        choices: [
            "中央値を使用した補完は不可能である",
            "訓練とテストデータセットを同時に補完しない",
            "inputColsとoutputColsは完全に一致する必要がある",
            "データに適合させてImputerModelを作成するfitステップが欠けている",
            "transformの代わりにfitメソッドを呼び出す必要がある"
        ],
        correctIndex: 3,
        explanation: `
            <p>コードが不正確な理由は、<code>fit()</code>ステップをスキップしているためです。このステップは以下のために必要です:</p>

            <ul>
                <li>指定された各列の中央値を計算</li>
                <li>これらの中央値を保存するImputerModelを作成</li>
            </ul>

            <h4>修正されたコード:</h4>
            <div class="code-block">from pyspark.ml.feature import Imputer

# 入力/出力列を定義（事前定義されていると仮定）
input_columns = ["col1", "col2"]  # 欠損値を持つ数値列
output_columns = ["col1_imputed", "col2_imputed"]  # 出力列名

# Imputerを初期化
my_imputer = Imputer(
    strategy="median",
    inputCols=input_columns,
    outputCols=output_columns
)

# 中央値を計算してImputerModelを作成
imputer_model = my_imputer.fit(features_df)  # 重要なステップ！

# 欠損値を補完するために変換
imputed_df = imputer_model.transform(features_df)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>「中央値を使用した補完は不可能」:</strong>
                    不正確。ImputerはString"median"、"mean"、"mode"をサポート
                </li>
                <li><strong>「訓練/テストセットを同時に補完しない」:</strong>
                    無関係。問題は欠けている<code>fit()</code>ステップであり、データセット分割ではない
                </li>
                <li><strong>「inputCols/outputColsが完全に一致する必要」:</strong>
                    一致する必要はない。outputColsは補完された列の名前
                </li>
                <li><strong>「transformの代わりにfit()を呼び出す」:</strong>
                    誤解を招く。両方が必要（最初に<code>fit()</code>、次に<code>transform()</code>）
                </li>
            </ul>

            <h4>重要性:</h4>
            <p>Spark ML補完には:</p>
            <ul>
                <li>✅ <code>fit()</code>は統計（例: 中央値）を計算し、モデルを作成</li>
                <li>✅ <code>transform()</code>は適合されたモデルを使用して補完を適用</li>
            </ul>

            <p><strong>注意:</strong> inputColsが数値（例: DoubleType、IntegerType）であることを常に確認</p>
        `
    }
    ,
    {
        number: 27,
        domain: "Pandas API on Spark",
        question: "pandas DataFrameコードをpandas API on Sparkに移行する際、空白を埋めるための正しいインポート文は？",
        keyPoint: "import pyspark.pandas as psで正しくインポートする",
        choices: [
            "import pandas as ps",
            "import databricks.pandas as ps",
            "import pyspark.pandas as ps",
            "import pandas.spark as ps",
            "import databricks.pyspark as ps"
        ],
        correctIndex: 2,
        explanation: `
            <p>pandas API on Spark（旧Koalas）を使用するには、正しいインポート文は:</p>

            <div class="code-block">import pyspark.pandas as ps  # 公式パッケージ名</div>

            <p>このエイリアス（ps）を使用すると、Sparkの分散コンピューティングを活用しながらPandas風の構文を使用できます。</p>

            <h4>なぜこれが機能するか:</h4>
            <ul>
                <li><strong>ps.read_parquet():</strong>
                    データをpandas-on-Spark DataFrameに読み込む（pandasまたはPySpark DataFrameではない）
                </li>
                <li><strong>df["category"].value_counts():</strong>
                    内部で分散カウント操作を実行
                </li>
            </ul>

            <div class="code-block">import pyspark.pandas as ps

# データを読み込む（分散）
df = ps.read_parquet("/path/to/data.parquet")

# Pandas風の操作（Sparkで実行）
counts = df["category"].value_counts()</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>import pandas as ps:</strong>
                    バニラpandas（シングルノード）を使用し、Sparkではない
                </li>
                <li><strong>import databricks.pandas as ps:</strong>
                    不正確なパッケージ名（そのようなモジュールは存在しない）
                </li>
                <li><strong>import pandas.spark as ps:</strong>
                    存在しない。pandasにはネイティブSpark統合がない
                </li>
                <li><strong>import databricks.pyspark as ps:</strong>
                    不正確。pysparkはネイティブSpark APIであり、pandas-on-Sparkではない
                </li>
            </ul>

            <h4>重要性:</h4>
            <p>pandas API on Sparkには:</p>
            <ul>
                <li>✅ <code>import pyspark.pandas as ps</code>を使用</li>
                <li>✅ 操作はpandasを模倣するがビッグデータにスケール</li>
            </ul>

            <p><strong>注意:</strong> パッケージがインストールされていることを確認（<code>%pip install pyspark-pandas</code>）</p>
        `
    }
    ,
    {
        number: 28,
        domain: "AutoML",
        question: "分類用のAutoML実行を設定する際、実行の期間を制御するために使用すべきパラメータは？",
        keyPoint: "timeout_minutesで実行時間を制限する",
        choices: [
            "max_trials",
            "timeout_minutes",
            "exclude_cols",
            "pos_label"
        ],
        correctIndex: 1,
        explanation: `
            <p>DatabricksでAutoML実行の期間を制御するには、<code>timeout_minutes</code>パラメータを使用します。これは、AutoML実験全体の時間制限（分単位）を設定し、その後、すべてのトライアルが完了していなくても実行が停止します。</p>

            <div class="code-block">from databricks import automl

# AutoML実行に60分のタイムアウトを設定
summary = automl.classify(
    dataset=df,
    target_col="label",
    timeout_minutes=60,  # 1時間後に停止
    exclude_cols=["id"]  # オプション: 無視する列
)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>max_trials:</strong>
                    トライアル（ハイパーパラメータの組み合わせ）の数を制限するが、実行時間ではない
                </li>
                <li><strong>exclude_cols:</strong>
                    訓練から除外する列を指定するが、時間制限とは無関係
                </li>
                <li><strong>pos_label:</strong>
                    メトリクス（例: precision/recall）の正クラスラベルを定義するが、実行時間ではない
                </li>
            </ul>

            <h4>重要性:</h4>
            <p>時間制限付きAutoML実行には:</p>
            <ul>
                <li>✅ <code>timeout_minutes</code>を使用して期間を制限</li>
                <li>✅ <code>max_trials</code>と組み合わせて探索と時間のバランスをとる</li>
            </ul>

            <p><strong>注意:</strong> Databricks AutoMLのデフォルトタイムアウトは120分（2時間）</p>
        `
    }
    ,
    {
        number: 29,
        domain: "ML Workflows",
        question: "Spark MLで文字列のカテゴリ属性をワンホットエンコードする際、エラーを修正するために必要な変更は何ですか？",
        keyPoint: "StringIndexerで文字列を数値インデックスに変換する",
        choices: [
            "列はinput_columnsと同じ名前で返す必要がある",
            "OneHotEncoderでmethodパラメータを指定する必要がある",
            "ワンホットエンコードを実行する前にStringIndexerを使用する必要がある",
            "fitを含む行を削除する必要がある"
        ],
        correctIndex: 2,
        explanation: `
            <p>Spark MLでは、カテゴリ文字列列は、OneHotEncoderを適用する前に、まず<strong>StringIndexer</strong>を使用して数値インデックスに変換する必要があります。</p>

            <h4>OneHotEncoderの要件:</h4>
            <ul>
                <li>SparkのOneHotEncoderは数値列（通常StringIndexerによって生成）でのみ動作</li>
                <li>生の文字列値（例: "red"、"blue"）を直接エンコードできない</li>
            </ul>

            <h4>正しいワークフロー:</h4>
            <div class="code-block">from pyspark.ml.feature import StringIndexer, OneHotEncoder

# ステップ1: 文字列を数値インデックスに変換
indexers = [
    StringIndexer(inputCol=col, outputCol=f"{col}_index")
    for col in input_columns
]

# ステップ2: インデックスをワンホットエンコード
encoder = OneHotEncoder(
    inputCols=[f"{col}_index" for col in input_columns],
    outputCols=output_columns
)

# ステップを連鎖するパイプライン
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=indexers + [encoder])
encoded_df = pipeline.fit(features_df).transform(features_df)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>「列はinput_columnsと同じ名前」:</strong>
                    不正確。outputColsは異なることができる（例: "color_index" → "color_encoded"）
                </li>
                <li><strong>「methodパラメータを指定」:</strong>
                    Spark MLのOneHotEncoderにはmethodパラメータがない（scikit-learnと異なる）
                </li>
                <li><strong>「fitを削除」:</strong>
                    <code>fit()</code>はカテゴリを学習してエンコーダモデルを作成するために必要
                </li>
            </ul>

            <h4>重要性:</h4>
            <p>Spark MLでのワンホットエンコードには:</p>
            <ul>
                <li>✅ 最初にStringIndexerを使用して文字列をインデックスに変換</li>
                <li>✅ 次にインデックス化された列にOneHotEncoderを適用</li>
            </ul>

            <p><strong>注意:</strong> 大きなカーディナリティの場合、<code>OneHotEncoder(dropLast=True)</code>を使用して次元爆発を回避</p>
        `
    }
    ,
    {
        number: 30,
        domain: "Spark ML",
        question: "Spark MLパイプラインでクラスタの再設定後にトレーニングセットの行数が変わる問題を解決するには？",
        keyPoint: "分割データセットを永続化して一貫性を保つ",
        choices: [
            "入力データセットの手動パーティショニングを実装する",
            "分割データセットを永続的に保存する",
            "クラスタ設定を手動で調整する",
            "データ分割プロセスでレートを規定する",
            "一貫したトレーニングとテストセットを保証する戦略は存在しない"
        ],
        correctIndex: 1,
        explanation: `
            <p>パイプライン実行全体で一貫したトレーニングとテストセット（クラスタ変更に関係なく）を確保するには、アナリストは以下を行うべきです:</p>

            <h4>推奨戦略:</h4>
            <ol>
                <li>データセットを1回分割（例: <code>randomSplit()</code>を使用）</li>
                <li>分割を永続化（例: Delta Lakeまたはメモリ/ストレージにキャッシュに保存）</li>
                <li>後続のすべての実行で同じ分割を再利用</li>
            </ol>

            <div class="code-block">from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# ステップ1: 分割して永続化
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
train_df.write.mode("overwrite").parquet("/path/to/train")
test_df.write.mode("overwrite").parquet("/path/to/test")

# ステップ2: 将来の実行で再ロード
train_df = spark.read.parquet("/path/to/train")
test_df = spark.read.parquet("/path/to/test")</div>

            <h4>なぜこれが機能するか:</h4>
            <ul>
                <li><strong>ランダム性を回避:</strong> 再パーティショニングやクラスタサイズ変更によるSparkの分割再計算を防ぐ</li>
                <li><strong>再現性:</strong> 実行全体で同じトレーニング/テストデータを保証</li>
            </ul>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>手動パーティショニング:</strong>
                    <code>randomSplit()</code>のランダム性には対処しない
                </li>
                <li><strong>クラスタ設定の調整:</strong>
                    無関係。分割はクラスタサイズではなくデータパーティショニングに依存
                </li>
                <li><strong>「分割でレートを規定」:</strong>
                    有効なSpark MLの概念ではない
                </li>
                <li><strong>「戦略は存在しない」:</strong>
                    不正確。永続化は一貫性を保証
                </li>
            </ul>

            <h4>重要性:</h4>
            <p>Spark MLでの決定論的分割には:</p>
            <ul>
                <li>✅ 分割を永続化（Delta/Parquet/cache）</li>
                <li>✅ <code>randomSplit()</code>で固定シードを使用</li>
            </ul>

            <p><strong>注意:</strong> キャッシング（<code>train_df.cache()</code>）は機能するがセッション間で揮発性。ディスクストレージ（Delta/Parquet）がより信頼性が高い</p>
        `
    }
    ,
    {
        number: 31,
        domain: "Pandas API on Spark",
        question: "Pandas API on SparkはApache Sparkのどのバージョンから利用可能になりましたか？",
        keyPoint: "Apache Spark 3.2で正式統合された",
        choices: [
            "Apache Spark 2.4",
            "Apache Spark 3.0",
            "Apache Spark 3.2",
            "Apache Spark 4.0"
        ],
        correctIndex: 2,
        explanation: `
            <p>Pandas API on Spark（旧Koalas）は、2021年10月にリリースされた<strong>Apache Spark 3.2</strong>で公式に統合されました。この統合により、ユーザーはpandas風のコードを記述すると自動的に分散Sparkクラスタにスケールできるようになりました。</p>

            <h4>主要マイルストーン:</h4>
            <ul>
                <li><strong>Spark 3.2:</strong> Pandas API on Sparkが<code>pyspark.pandas</code>モジュールの下で公式Spark配布の一部になった</li>
                <li><strong>Spark 3.2以前:</strong> ユーザーは別個のkoalasパッケージをインストールする必要があった</li>
            </ul>

            <div class="code-block">import pyspark.pandas as ps  # Spark 3.2+が必要

# pandas風の構文でデータを読み込む（Sparkで実行）
df = ps.read_csv("data.csv")
df.groupby("category").sum()</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>Spark 2.4:</strong>
                    Pandas API on Sparkより前。ユーザーはkoalasのようなサードパーティツールに依存していた
                </li>
                <li><strong>Spark 3.0:</strong>
                    多くの改善を導入したがPandas APIは含まれていなかった
                </li>
                <li><strong>Spark 4.0:</strong>
                    2024年時点ではまだ存在しない
                </li>
            </ul>

            <h4>重要性:</h4>
            <p>SparkでのPandas風構文には:</p>
            <ul>
                <li>✅ Spark 3.2以降で<code>pyspark.pandas</code>を使用</li>
                <li>✅ 別途koalasをインストールする必要がない</li>
            </ul>
        `
    }
    ,
    {
        number: 32,
        domain: "Cluster Creation and Management",
        question: "Databricks GPU対応クラスタでGPU加速ライブラリgpu_mlをインストールする推奨方法は？",
        keyPoint: "クラスタのGPUライブラリ依存関係に追加する",
        choices: [
            "Databricks Runtime for GPUを使用するようにクラスタを編集する",
            "クラスタ設定でPYTHON_GPU_LIB変数を設定してgpu_mlを含める",
            "クラスタに接続されたノートブックで一度%pip install gpu_mlを実行する",
            "クラスタのGPUライブラリ依存関係にgpu_mlを追加する",
            "DatabricksクラスタにGPU加速ライブラリをインストールする方法はない"
        ],
        correctIndex: 3,
        explanation: `
            <p>Databricks GPU対応クラスタですべてのノートブックでPythonライブラリ（GPU加速ライブラリを含む）を使用する必要がある場合、推奨される最もシームレスなアプローチは、<strong>クラスタライブラリとしてインストール</strong>することです。</p>

            <h4>なぜこの方法が最適か:</h4>
            <ul>
                <li><strong>クラスタ全体でアクセス可能:</strong>
                    そのクラスタに接続されたすべてのノートブックが自動的にライブラリにアクセスできる
                </li>
                <li><strong>永続性:</strong>
                    クラスタを再起動しても毎回ライブラリを再インストールする必要がない
                </li>
                <li><strong>環境統一:</strong>
                    GPUランタイムが使用しているのと同じ環境にライブラリがインストールされることを保証
                </li>
            </ul>

            <h4>Databricks UIでの手順:</h4>
            <ol>
                <li>Databricks UIでクラスタに移動</li>
                <li>「Libraries」を選択</li>
                <li>「Install New」をクリック → PyPIを選択 → ライブラリ名（gpu_ml）を入力 → 「Install」をクリック</li>
            </ol>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>GPU Runtimeに編集:</strong>
                    GPU加速を活用するにはGPUランタイムが必要だが、ランタイムに切り替えるだけではgpu_mlが自動的にインストールされない。ライブラリ自体をインストールする必要がある
                </li>
                <li><strong>PYTHON_GPU_LIB変数を設定:</strong>
                    Databricksは環境変数に基づいてPythonパッケージをインストールしない。クラスタライブラリまたは%pip installのいずれかを使用
                </li>
                <li><strong>%pip install gpu_ml:</strong>
                    可能だが、そのノートブックセッションの環境にのみライブラリをインストール。すべてのノートブックでパッケージを永続化したい場合（特に再起動後）は推奨されない。クラスタ再起動後に%pip installを再実行する必要がある
                </li>
                <li><strong>インストール方法がない:</strong>
                    明らかに不正確。Databricksは、GPU対応クラスタを選択してライブラリを適切にインストールすれば、GPU加速ライブラリとワークロードを完全にサポート
                </li>
            </ul>
        `
    }
    ,
    {
        number: 33,
        domain: "AutoML",
        question: "AutoMLでカスタム補完メソッドが指定された列がある場合、どうなりますか？",
        keyPoint: "セマンティックタイプ検出が実行されない",
        choices: [
            "訓練中に列を無視する",
            "それらの列にセマンティックタイプ検出を実行する",
            "それらの列に特徴エンジニアリングを実行する",
            "それらの列にセマンティックタイプ検出を実行しない"
        ],
        correctIndex: 3,
        explanation: `
            <p>Databricks AutoMLで、列にカスタム補完メソッドが指定されている場合、AutoMLはその列の<strong>セマンティックタイプ検出をスキップ</strong>します。</p>

            <h4>これが意味すること:</h4>
            <ul>
                <li><strong>列タイプの自動推論なし:</strong>
                    通常、AutoMLはセマンティックタイプ（例: カテゴリカル、数値、日時、テキスト）を検出して、エンコーディングや特徴エンジニアリングなどの前処理ステップを適用する。
                    カスタム補完戦略が定義されている場合、AutoMLはユーザーが欠損値を手動で処理したと想定し、補完メソッドを上書きしない
                </li>
                <li><strong>指定された補完以外の変更なし:</strong>
                    AutoMLはユーザー定義の補完メソッドを尊重し、列をさらに変更しない。
                    補完された値が有効であれば、モデル訓練に列を使用する
                </li>
            </ul>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>「訓練中に列を無視する」:</strong>
                    不正確。補完が適用されている場合、AutoMLは列を無視しない。訓練に含める
                </li>
                <li><strong>「セマンティックタイプ検出を実行する」:</strong>
                    不正確。カスタム補完は、AutoMLの自動タイプ検出を無効にする。
                    タイプはユーザーによって正しく処理されていると想定される
                </li>
                <li><strong>「特徴エンジニアリングを実行する」:</strong>
                    不正確。特徴エンジニアリング（エンコーディングや変換など）は通常セマンティックタイプ検出に依存するが、AutoMLはカスタム補完列に対してはこれをスキップする
                </li>
            </ul>

            <h4>結論:</h4>
            <p>カスタム補完が指定されている場合、AutoMLはそれらの列に対してセマンティックタイプ検出を実行せず、ユーザーが前処理を正しく処理したと想定します。</p>
        `
    }
    ,
    {
        number: 34,
        domain: "Cluster Creation and Management",
        question: "探索的データ分析とモデルプロトタイピングのために最も適切なクラスタタイプは？",
        keyPoint: "Single-nodeクラスタがEDAとプロトタイピングに最適",
        choices: [
            "Multi-node Cluster",
            "Single-node Cluster",
            "Task-specific Cluster",
            "Standard Cluster"
        ],
        correctIndex: 1,
        explanation: `
            <p>探索的データ分析（EDA）とモデルプロトタイピングには、通常<strong>Single-nodeクラスタ</strong>が最も適切な選択です。</p>

            <h4>なぜSingle-Nodeクラスタか？</h4>
            <ul>
                <li><strong>複雑性の低減:</strong>
                    Single-nodeクラスタはドライバとexecutorを同じノードで実行。早期の実験フェーズでリソース割り当てと環境セットアップを簡素化
                </li>
                <li><strong>コスト効率:</strong>
                    データセットが大規模でない場合、Single-nodeクラスタは複数ノードを起動するよりコスト効率が良い
                </li>
                <li><strong>高速起動と反復:</strong>
                    Single-nodeクラスタは一般的に迅速に起動し、データを探索しモデルをプロトタイプする際により速く反復できる
                </li>
            </ul>

            <h4>他の選択肢が不適切な理由:</h4>
            <ul>
                <li><strong>Multi-node Cluster:</strong>
                    通常、並列処理と分散コンピューティングが必要な大規模データ処理または訓練ワークロードに使用。EDAやモデルプロトタイピングが大規模データセットを含まない場合、これは過剰
                </li>
                <li><strong>Task-specific Cluster:</strong>
                    ジョブタスクを実行するために特別に作成されるジョブクラスタを指すことが多い。ジョブ用に起動し、その後シャットダウン。インタラクティブな探索とプロトタイピングで永続的なセッションが必要な場合は柔軟性がない
                </li>
                <li><strong>Standard Cluster:</strong>
                    多くのDatabricksコンテキストで、「standardクラスタ」はマルチノードクラスタである可能性がある。大規模なデータ用に分散コンピューティングを特に必要としない限り、迅速なEDAとプロトタイピングにはSingle-nodeがより直接的
                </li>
            </ul>

            <h4>結論:</h4>
            <p>探索と初期モデル実験には、<strong>Single-nodeクラスタ</strong>が通常最良の選択です。</p>
        `
    }
    ,
    {
        number: 35,
        domain: "Pandas API on Spark",
        question: "pandas-on-Sparkのtransform_batch()のような関数での'batch'接尾辞は何を意味しますか？",
        keyPoint: "pandas-on-Spark DataFrameまたはSeriesのチャンクを指定する",
        choices: [
            "DataFrameの特定の行を示す",
            "pandas-on-Spark DataFrameまたはSeriesのチャンクを指定する",
            "DataFrame全体を指す",
            "列に対する単一の操作を示す"
        ],
        correctIndex: 1,
        explanation: `
            <h4>説明:</h4>

            <p><strong>分散処理:</strong></p>
            <p>Pandas API on SparkはSparkの分散処理機能を活用して、大規模データセットを効率的に処理します。</p>

            <p><strong>チャンクベースの操作:</strong></p>
            <p>これを実現するために、DataFrameまたはSeriesを処理のためのより小さな管理可能なチャンクに分割することが多いです。</p>

            <p><strong>'batch'接尾辞:</strong></p>
            <p><code>transform_batch()</code>のような関数の<code>batch</code>接尾辞は、これらが一度にDataFrameまたはSeries全体ではなく、データのこれらのチャンクで動作することを意味します。</p>

            <h4>重要ポイント:</h4>
            <ul>
                <li><strong>最適化された性能:</strong>
                    チャンクベースの操作は、大規模データセットの性能とメモリ効率を改善することが多い
                </li>
                <li><strong>関数固有の動作:</strong>
                    batch接尾辞を持つ関数の正確な動作は、関数の目的によって異なる:
                    <ul>
                        <li>一部は各チャンクに独立して関数を適用し、結合された結果を返す</li>
                        <li>他は特定の方法でチャンク間で結果を集約または結合する場合がある</li>
                    </ul>
                </li>
            </ul>

            <div class="code-block">import pandas as pd

# サンプルpandas-on-Spark DataFrame
df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})

# チャンクに適用する関数を定義
def square_values(df):
    return df * df

# transform_batchを使用してチャンクに関数を適用
df_squared = df.pandas_on_spark.transform_batch(square_values)
# 出力: 列AとBが二乗されたDataFrame、チャンクで処理</div>

            <h4>チャンクベース操作の利点:</h4>
            <ul>
                <li><strong>スケーラビリティ:</strong> 単一マシンのメモリに収まらない可能性がある大規模データセットの効率的な処理を可能にする</li>
                <li><strong>並列性:</strong> Sparkクラスタ内の複数ノード間でチャンクの並列処理を可能にし、計算を加速する可能性がある</li>
                <li><strong>リソース管理:</strong> より小さな部分でデータを処理することでメモリ使用を最適化</li>
            </ul>
        `
    }
    ,
    {
        number: 36,
        domain: "Scaling ML Models",
        question: "分散コンピューティングシステムにおけるデータローカリティの目的は何ですか？",
        keyPoint: "データ転送オーバーヘッドを削減する",
        choices: [
            "データレプリケーションを最小化する",
            "データ転送オーバーヘッドを削減する",
            "データ一貫性を確保する",
            "データ圧縮を強化する"
        ],
        correctIndex: 1,
        explanation: `
            <p>分散コンピューティング（例: Apache Spark）におけるデータローカリティとは、データが存在する場所で計算を実行することにより、ネットワーク上のデータ移動を最小化する原則を指します。これは以下のために重要です:</p>

            <h4>性能最適化:</h4>
            <ul>
                <li>ノード間で大規模データセットを転送することを回避し、レイテンシとネットワーク輻輳を削減</li>
            </ul>

            <h4>効率性:</h4>
            <ul>
                <li>必要なデータパーティションを既に持っているノードでタスクがスケジュールされる</li>
            </ul>

            <h4>Sparkがデータローカリティを実現する方法:</h4>
            <ul>
                <li><strong>RDD/DataFrameパーティショニング:</strong> データはワーカー間に分散されたパーティションに分割される</li>
                <li><strong>タスクスケジューリング:</strong> Sparkは関連データを保持するノードでタスクをスケジュールすることを優先（「推奨場所」）</li>
            </ul>

            <div class="code-block"># データを読み込む（パーティションがワーカー間に分散）
df = spark.read.parquet("data.parquet")

# フィルタ操作はデータが存在するノードで実行（正しくパーティション化されていればシャッフルなし）
filtered_df = df.filter(df["value"] > 100)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>データレプリケーションを最小化:</strong>
                    データローカリティはレプリケーションを削減しない（レプリケーションはフォールトトレランスのため）
                </li>
                <li><strong>データ一貫性を確保:</strong>
                    一貫性はストレージシステム（例: Delta Lake）によって管理され、ローカリティではない
                </li>
                <li><strong>データ圧縮を強化:</strong>
                    圧縮はローカリティとは独立している
                </li>
            </ul>

            <h4>重要性:</h4>
            <p>データローカリティは以下によってネットワークオーバーヘッドを削減します:</p>
            <ul>
                <li>✅ 計算とデータの共存</li>
                <li>✅ パーティショニングの活用（例: 結合でのキーによる）</li>
            </ul>

            <p><strong>注意:</strong> 特定のワークフローのローカリティを最適化するには、<code>repartition()</code>または<code>partitionBy()</code>を使用</p>
        `
    }
    ,
    {
        number: 37,
        domain: "Spark ML",
        question: "pandas APIとネイティブSpark DataFramesを比較した場合、特に大規模データセットでパフォーマンス速度が低下する理由は？",
        keyPoint: "internalFrameのメタデータ管理オーバーヘッド",
        choices: [
            "メタデータを維持するためのinternalFrameの採用",
            "増加したコード量の要件",
            "CSVファイルへの依存",
            "すべての処理操作の即時評価",
            "データ分散の欠如"
        ],
        correctIndex: 0,
        explanation: `
            <p>Pandas API on Spark（Koalas）は、Pandas風の操作とSparkの分散実行を橋渡しするための<strong>internalFrame</strong>レイヤーを導入しています。これにより使い慣れた構文が可能になりますが、以下によるオーバーヘッドが追加されます:</p>

            <h4>メタデータ管理:</h4>
            <ul>
                <li>internalFrameはPandas風のインデックス、列名、データ型を追跡し、追加の記録管理が必要</li>
            </ul>

            <h4>変換コスト:</h4>
            <ul>
                <li>Pandas操作はこのレイヤーを介してSparkプランに変換され、ネイティブSpark DataFramesと比較して実行が遅くなる可能性がある</li>
            </ul>

            <div class="code-block">import pyspark.pandas as ps

# pandas API on Spark（internalFrameを使用）
kdf = ps.DataFrame(...)
result = kdf.groupby("col1").sum()  # メタデータ処理により遅い

# ネイティブSpark（直接実行）
sdf = spark.createDataFrame(...)
result = sdf.groupBy("col1").sum()  # より速い</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>「増加したコード量」:</strong>
                    無関係。パフォーマンスはコード量ではなく実行に関する
                </li>
                <li><strong>「CSVファイルへの依存」:</strong>
                    無関係。データソース形式はAPIパフォーマンスに影響しない
                </li>
                <li><strong>「即時評価」:</strong>
                    両方のAPIは遅延評価を使用
                </li>
                <li><strong>「データ分散の欠如」:</strong>
                    誤り。pandas API on Sparkはデータを分散する（バニラpandasとは異なる）
                </li>
            </ul>

            <h4>重要性:</h4>
            <p>大規模データセットには、以下の場合にネイティブSpark DataFramesを優先:</p>
            <ul>
                <li>✅ パフォーマンスが重要（internalFrameオーバーヘッドを回避）</li>
                <li>✅ 高度なSpark最適化（例: 述語プッシュダウン）が必要</li>
            </ul>

            <p>pandas API on Sparkの使用:</p>
            <ul>
                <li>✅ 小〜中規模の分散データでのPandas使い慣れた感覚のため</li>
            </ul>
        `
    }
    ,
    {
        number: 38,
        domain: "AutoML",
        question: "AutoMLが時系列で同じタイムスタンプに複数の値がある予測問題をどのように処理しますか？",
        keyPoint: "値を平均化する",
        choices: [
            "最大値を取る",
            "最小値を取る",
            "値を平均化する",
            "追加の値を破棄する"
        ],
        correctIndex: 2,
        explanation: `
            <p>AutoMLが時系列予測問題で同じタイムスタンプに複数の値がある場合、デフォルトで<strong>値を平均化</strong>して集約します。これにより以下が保証されます:</p>

            <h4>一貫性:</h4>
            <ul>
                <li>データのノイズまたは重複を平滑化</li>
            </ul>

            <h4>安定性:</h4>
            <ul>
                <li>極端な値（例: 最大/最小）へのバイアスを防止</li>
            </ul>

            <div class="code-block">from databricks import automl

# AutoMLは重複タイムスタンプを自動的に処理
summary = automl.forecast(
    dataset=df,
    time_col="timestamp",
    target_col="value",
    horizon=7  # 7期間先を予測
)</div>

            <h4>入力データ:</h4>
            <div class="code-block">| timestamp           | value |
|---------------------|-------|
| 2023-01-01 00:00:00 | 10    |
| 2023-01-01 00:00:00 | 20    | → AutoMLは**15**に平均化</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>最大/最小値:</strong>
                    バイアスを導入（トレンドを過大評価/過小評価）
                </li>
                <li><strong>値を破棄:</strong>
                    情報を失い、データセットの品質を低下させる
                </li>
            </ul>

            <h4>重要性:</h4>
            <p>重複タイムスタンプでの予測には:</p>
            <ul>
                <li>✅ AutoMLはデフォルトで値を平均化</li>
                <li>✅ カスタム集約（例: 合計、中央値）が必要な場合は手動で前処理</li>
            </ul>
        `
    }
    ,
    {
        number: 39,
        domain: "Databricks ML",
        question: "MLflowでハイパーパラメータチューニングを行い、異なる実行のパフォーマンスを比較して最適なハイパーパラメータを選択するために使用すべき関数は？",
        keyPoint: "mlflow.search_runsで実行を検索・比較する",
        choices: [
            "mlflow.search_runs",
            "mlflow.log_param",
            "mlflow.start_run",
            "mlflow.search_hyperparams"
        ],
        correctIndex: 0,
        explanation: `
            <p>MLflowでハイパーパラメータチューニングを実行する場合、データサイエンティストは異なる実行を比較し、最適なハイパーパラメータ値を見つける必要があります。<code>mlflow.search_runs()</code>関数を使用すると、メトリクス、パラメータ、タグに基づいて過去の実行をクエリおよびフィルタリングでき、理想的な選択肢となります。</p>

            <h4>mlflow.search_runs()がハイパーパラメータチューニングにどう役立つか:</h4>
            <ul>
                <li><strong>すべての実験実行を取得・比較:</strong>
                    ハイパーパラメータ（log_param）と評価メトリクス（log_metric）を含む、記録されたすべての実行をリスト化
                </li>
                <li><strong>性能に基づいて実行をフィルタリング:</strong>
                    例: 最高精度または最低損失の実行を見つける
                </li>
                <li><strong>実行をソートして最良のモデルを識別:</strong>
                    特定のメトリクスでソートして最良のハイパーパラメータの組み合わせを選択
                </li>
            </ul>

            <div class="code-block">import mlflow
import pandas as pd

# 実験からすべての実行を検索・取得
experiment_id = mlflow.get_experiment_by_name("my_experiment").experiment_id
runs_df = mlflow.search_runs(experiment_ids=[experiment_id])

# 精度に基づいて最良の実行を見つける
best_run = runs_df.loc[runs_df["metrics.accuracy"].idxmax()]
print(best_run[["run_id", "params.learning_rate", "metrics.accuracy"]])</div>

            <p>これにより、すべての実行を取得し、精度でソートして最良のハイパーパラメータを選択します。</p>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>mlflow.log_param:</strong>
                    実行内で単一のハイパーパラメータ値をログするために使用されるが、複数の実行を比較しない
                </li>
                <li><strong>mlflow.start_run:</strong>
                    MLflow実行を開始するが、ハイパーパラメータチューニングのために過去の実行を取得しない
                </li>
                <li><strong>mlflow.search_hyperparams:</strong>
                    この関数はMLflowに存在しない
                </li>
            </ul>

            <h4>結論:</h4>
            <p>ハイパーパラメータチューニングのために過去の実行を取得、比較、フィルタリングするには、<code>mlflow.search_runs()</code>が正しい関数です。</p>
        `
    }
    ,
    {
        number: 40,
        domain: "Spark ML Algorithms",
        question: "非常に不均衡なクラスを扱う機械学習プロジェクトで、クラス不均衡に対処してモデル性能を改善するのに適したSpark MLアルゴリズムは？",
        keyPoint: "SMOTEで少数クラスを合成的にオーバーサンプリングする",
        choices: [
            "Decision Trees",
            "Random Forest",
            "Support Vector Machines",
            "Synthetic Minority Over-sampling Technique (SMOTE)"
        ],
        correctIndex: 3,
        explanation: `
            <p>機械学習プロジェクトで非常に不均衡なクラスを扱う場合、最良のアプローチは、モデルが多数クラスに偏らないように訓練前にデータセットのバランスを取ることです。Spark MLでは、<strong>SMOTE（Synthetic Minority Over-sampling Technique）</strong>が合成サンプルを生成することで少数クラスをオーバーサンプリングする適切な技術です。</p>

            <h4>なぜSMOTE？</h4>
            <ul>
                <li><strong>クラス分布のバランスを取る:</strong>
                    SMOTEは既存のインスタンスを単に複製するのではなく、少数クラスの合成サンプルを作成し、データセットをよりバランスの取れたものにする
                </li>
                <li><strong>モデル汎化を改善:</strong>
                    支配的なクラスに過適合するのではなく、モデルがより良い決定境界を学習するのを助ける
                </li>
                <li><strong>Spark MLで利用可能:</strong>
                    Imbalanced-learn（imbalanced-learn.org）などのライブラリと組み合わせたpyspark.mlを使用するか、Sparkでカスタム実装が可能
                </li>
            </ul>

            <div class="code-block">from imblearn.over_sampling import SMOTE
from pyspark.sql import SparkSession
import pandas as pd

# Sparkセッションを初期化
spark = SparkSession.builder.appName("SMOTE Example").getOrCreate()

# サンプル不均衡データセット（SMOTE用にPandasに変換）
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [5, 4, 3, 6, 7, 2, 8, 9, 1, 0],
    'label': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 不均衡クラス
})

# SMOTEを適用
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(data[['feature1', 'feature2']], data['label'])

# Spark DataFrameに戻す
balanced_df = spark.createDataFrame(
    pd.DataFrame(X_resampled, columns=['feature1', 'feature2']).assign(label=y_resampled)
)
balanced_df.show()</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>Decision Trees:</strong>
                    本質的にクラス不均衡をうまく処理しない。不均衡データで訓練すると多数クラスを優先する傾向がある
                </li>
                <li><strong>Random Forest:</strong>
                    クラス不均衡を直接解決しない。アンサンブル手法は役立つ可能性があるが、Random Forestはより良い性能のためにSMOTEのような再サンプリング技術が依然として必要
                </li>
                <li><strong>Support Vector Machines (SVMs):</strong>
                    重み付きクラスまたは再サンプリングと共に使用しない限り、不均衡データを本質的にうまく処理しない
                </li>
            </ul>

            <h4>結論:</h4>
            <p>非常に不均衡なクラスを扱う場合、Spark MLでの最良のアプローチは<strong>SMOTE（Synthetic Minority Over-sampling Technique）</strong>です。合成サンプルを生成してデータセットのバランスを取り、モデル性能を改善します。</p>
        `
    },
    {
        number: 41,
        domain: "Spark ML",
        question: "データサイエンティストがSparkSQLを使用して機械学習パイプラインにデータをインポートしています。データをインポートした後、すべてのデータをpandas DataFrameに収集し、scikit-learnを使用して機械学習タスクを実行します。このユースケースに最も適したDatabricksクラスタモードはどれですか？",
        keyPoint: "Single-Nodeは単一ノードライブラリに最適",
        choices: [
            "SQL Endpoint",
            "Standard",
            "High Concurrency",
            "Pooled",
            "Single Node"
        ],
        correctIndex: 4,
        explanation: `<strong>正解：Single Node</strong><br><br>このシナリオでは、データをpandas DataFrameに収集し、scikit-learnで処理しています。これらは単一ノードで動作する非分散ライブラリです。<br><br><strong>理由：</strong><br>・SparkSQLはデータインポートのみに使用され、分散ML訓練には使用されていない<br>・Single Nodeクラスタはコスト効率が良く、このワークロードに十分<br>・分散コンピューティングの不要なオーバーヘッドを回避<br><br><strong>ワークフロー例：</strong><br><div class="code-block"># SparkSQLでデータインポート（ドライバーで実行）<br>spark_df = spark.sql("SELECT * FROM table")<br># pandasに変換（ドライバーにデータを収集）<br>pandas_df = spark_df.toPandas()<br># scikit-learnモデル訓練（単一ノード）<br>from sklearn.ensemble import RandomForestClassifier<br>model = RandomForestClassifier().fit(pandas_df[features], pandas_df[label])</div><br><strong>他の選択肢が不正解な理由：</strong><br>・Standard Mode：分散ワークロード（Spark MLlib等）向け。scikit-learnには過剰<br>・High Concurrency：複数ユーザーのSQLクエリ向け。単一ユーザーMLワークロードには不適<br>・SQL Endpoint：サーバーレスSQLサービス。PythonのMLコードには不適<br>・Pooled：ジョブクラスター用。ここでは無関係`
    },
    {
        number: 42,
        domain: "Pandas API on Spark",
        question: "pandas API on Spark関数でスキーマ推論に依存することが潜在的に高コストになる理由は何ですか？",
        keyPoint: "スキーマ推論はSparkジョブを2回実行する可能性",
        choices: [
            "スキーマ推論は常に安価である",
            "パフォーマンスが向上する",
            "Sparkジョブを2回実行する可能性がある",
            "pandas API on Sparkではスキーマ推論がサポートされていない"
        ],
        correctIndex: 2,
        explanation: `<strong>正解：Sparkジョブを2回実行する可能性がある</strong><br><br>pandas API on Spark（pyspark.pandas）でスキーマ推論に依存すると、Sparkジョブが2回実行される可能性があるため、潜在的に高コストになります。<br><br><strong>理由：</strong><br>・Sparkはスキーマを決定するためにデータをスキャンする必要がある<br>・大規模データセットでは、これが追加の計算オーバーヘッドになる<br>・特定の操作では、スキーマ推論で1回、実際の計算で1回実行される<br><br><strong>高コストなスキーマ推論の例：</strong><br><div class="code-block">import pyspark.pandas as ps<br># スキーマ指定なしでCSV読み込み<br>df = ps.read_csv("large_dataset.csv")<br># 変換実行<br>df["new_col"] = df["existing_col"] * 2</div><br>スキーマが明示的に提供されていないため、Sparkは列の型を推論するためにデータをスキャンします。変換で再スキャンが必要な場合、Sparkはジョブを再実行し、パフォーマンスが低下します。<br><br><strong>ベストプラクティス：</strong><br>・read_csv()やread_parquet()でスキーマを明示的に定義<br>・キャッシュされたDataFrameを使用して冗長スキャンを回避<br><div class="code-block">df = ps.read_csv("large_dataset.csv", dtype={"existing_col": "int"})</div>`
    },
    {
        number: 43,
        domain: "Scaling ML Models",
        question: "正確な結果を保証するために分散ノードの同期が必要な機械学習モデルを実装しています。並列処理環境で分散ノードを同期するプロセスは何と呼ばれますか？",
        keyPoint: "同期は分散ノードの計算を整列させる",
        choices: [
            "並列化（Parallelization）",
            "同期（Synchronization）",
            "調整（Coordination）",
            "集約（Aggregation）"
        ],
        correctIndex: 1,
        explanation: `<strong>正解：同期（Synchronization）</strong><br><br>分散ML（Spark MLlib、Horovod、TensorFlow分散訓練等）において、同期はワーカーノードが計算（ディープラーニングの勾配更新等）を整列させ、モデルの一貫性を保つことを保証します。<br><br><strong>同期の例：</strong><br>・AllReduce（Horovod等のフレームワーク）：GPU/CPU間で勾配を同期<br>・Barrier実行（Spark）：ノードが他のノードを待ってから処理を続行<br><br><strong>コード例（HorovodのTensorFlow）：</strong><br><div class="code-block">import horovod.tensorflow as hvd<br>hvd.init()  # 同期コンテキスト初期化<br>optimizer = hvd.DistributedOptimizer(optimizer)  # 勾配同期</div><br><strong>他の選択肢が不正解な理由：</strong><br>・並列化（Parallelization）：ノード間で作業を分割するが、調整を意味しない<br>・調整（Coordination）：同期より広範（タスクスケジューリング等）。計算整列に特化していない<br>・集約（Aggregation）：結果を結合（勾配平均化等）するが、ノード整列を保証しない<br><br><strong>重要ポイント：</strong><br>同期は、ML訓練で分散ノードを整列させる正確な用語であり、競合状態を回避し正確性を保証するために重要です。`
    },
    {
        number: 44,
        domain: "AutoML",
        question: "Databricks AutoMLが分類モデルで使用するアルゴリズムはどれですか？",
        keyPoint: "決定木、ランダムフォレスト、ロジスティック回帰、XGBoost",
        choices: [
            "決定木とランダムフォレスト",
            "決定木、ランダムフォレスト、Auto-ARIMA",
            "決定木、ランダムフォレスト、ロジスティック回帰、XGBoost",
            "決定木、ランダムフォレスト、ロジスティック回帰、XGBoost、Prophet"
        ],
        correctIndex: 2,
        explanation: `<strong>正解：決定木、ランダムフォレスト、ロジスティック回帰、XGBoost</strong><br><br>Databricks AutoMLは分類タスクで複数のアルゴリズムを自動的に訓練・比較します。<br><br><strong>使用されるアルゴリズム：</strong><br>・決定木（Decision Trees）：シンプルで解釈可能なモデル<br>・ランダムフォレスト（Random Forests）：決定木のアンサンブルで精度向上<br>・ロジスティック回帰（Logistic Regression）：確率的分類の線形モデル<br>・XGBoost：高性能の勾配ブースティング木<br><br>これらのアルゴリズムは、解釈可能性から高精度まで幅広いユースケースをカバーし、分散環境で効果的に機能します。<br><br><strong>AutoMLワークフロー例：</strong><br><div class="code-block">from databricks import automl<br>summary = automl.classify(<br>    dataset=df,<br>    target_col="label",<br>    timeout_minutes=30<br>)<br># 出力には4つのアルゴリズムタイプの試行が含まれる</div><br><strong>他の選択肢が不正解な理由：</strong><br>・「決定木とランダムフォレスト」：限定的。ロジスティック回帰とXGBoostが欠けている<br>・「Auto-ARIMA含む」：Auto-ARIMAは時系列予測用。分類には無関係<br>・「Prophet含む」：Prophetも時系列ツール。分類タスクでは使用されない`
    },
    {
        number: 45,
        domain: "Spark ML",
        question: "グループ化されたマップPandas UDFをPySpark DataFrameに適用するにはどうすればよいですか？",
        keyPoint: "groupBy().applyInPandas()を使用",
        choices: [
            "DataFrameの列にapplyメソッドを使用",
            "DataFrameにapplyInPandasメソッドを使用",
            "DataFrameでgroupByメソッドに続いてapplyInPandasメソッドを使用",
            "DataFrameでgroupByメソッドに続いてaggメソッドを使用"
        ],
        correctIndex: 2,
        explanation: `<strong>正解：DataFrameでgroupByメソッドに続いてapplyInPandasメソッドを使用</strong><br><br>グループ化されたマップPandas UDF（User-Defined Function）を使用すると、カスタムPandas関数をグループ化されたPySpark DataFrameの各グループに適用できます。<br><br><strong>正しい使用方法：</strong><br><div class="code-block">grouped_df = df.groupBy("column_name").applyInPandas(pandas_udf_function, schema)</div><br><strong>applyInPandas()を使用する理由：</strong><br>・各グループを独立して処理可能<br>・各グループはpandas DataFrameとして渡され、関数が個別に適用される<br>・構造化されたPySpark DataFrameを返す<br><br><strong>コード例：</strong><br><div class="code-block">from pyspark.sql.types import StructType, StructField, DoubleType<br>import pandas as pd<br><br># 出力スキーマ定義<br>schema = StructType([<br>    StructField("category", StringType()),<br>    StructField("mean_value", DoubleType())<br>])<br><br># グループ化マップPandas UDF関数定義<br>def compute_mean(pdf):<br>    return pd.DataFrame({<br>        "category": [pdf["category"].iloc[0]],<br>        "mean_value": [pdf["value"].mean()]<br>    })<br><br># 適用<br>result_df = df.groupBy("category").applyInPandas(compute_mean, schema)</div><br><strong>他の選択肢が不正解な理由：</strong><br>・「apply()メソッド」：PySparkにはDataFrame用のapply()メソッドが存在しない<br>・「agg()メソッド」：集約関数（sum、count、avg等）用。カスタムPandas UDFには使用不可`
    },
    {
        number: 46,
        domain: "Scaling ML Models",
        question: "機械学習プロジェクトでSparkパフォーマンスを最適化しています。複数の小さなタスクを大きなタスクに結合してスケジューリングオーバーヘッドを削減し、処理効率を向上させる技術は何ですか？",
        keyPoint: "タスク集約はオーバーヘッド削減に有効",
        choices: [
            "タスク集約（Task Aggregation）",
            "タスク分解（Task Decomposition）",
            "タスク並列性（Task Parallelism）",
            "タスク融合（Task Fusion）"
        ],
        correctIndex: 0,
        explanation: `<strong>正解：タスク集約（Task Aggregation）</strong><br><br>タスク集約は、複数の小さなタスクを大きなタスクに結合してスケジューリングオーバーヘッドを削減し、処理効率を向上させる技術です。<br><br><strong>説明：</strong><br>Sparkのような分散環境では、データの各パーティションは一般的に1つのタスクとして処理されます。非常に小さなパーティションが多数ある場合、大量のタスクをスケジューリング・調整するオーバーヘッドが大きくなります。<br><br><strong>タスク集約の利点：</strong><br>・スケジューリングオーバーヘッド全体の削減（管理するタスク数が減少）<br>・最適なサイズに達すると並列性が向上（大きすぎてストラグラーを引き起こさず、小さすぎてオーバーヘッドが支配的にならない）<br>・Adaptive Query Execution（AQE）等の機能でランタイムにパーティションを結合可能<br><br><strong>他の選択肢が不正解な理由：</strong><br>・タスク分解（Task Decomposition）：逆の概念。大きなタスクを小さく分割<br>・タスク並列性（Task Parallelism）：タスクを並列実行。タスクのマージには言及していない<br>・タスク融合（Task Fusion）：標準的なSpark用語ではない。演算子融合（コード生成最適化）を指す可能性があるが、タスク結合とは異なる`
    },
    {
        number: 47,
        domain: "MLflow",
        question: "MLflowモデルレジストリに既に存在するmodel_nameというモデルに対して、mlflow.sklearn.log_model()でregistered_model_name=model_nameを指定すると何が起こりますか？",
        keyPoint: "既存モデルの新バージョンを登録",
        choices: [
            "後続のmlflow.register_model呼び出しでモデル名を指定する必要がなくなる",
            "MLflowモデルレジストリにmodel_nameという新しいモデルを記録する",
            "MLflow Experimentでログされたモデルの名前を表す",
            "MLflowモデルレジストリのmodel_nameモデルの新バージョンを登録する",
            "MLflow ExperimentのRunの名前を示す"
        ],
        correctIndex: 3,
        explanation: `<strong>正解：MLflowモデルレジストリのmodel_nameモデルの新バージョンを登録する</strong><br><br>registered_model_name=model_nameがmlflow.sklearn.log_model()で使用され、モデルが既にMLflowモデルレジストリに存在する場合、既存モデルの新しいバージョンを作成します（例：v1からv2に増加）。<br><br><strong>ワークフロー例：</strong><br><div class="code-block">import mlflow<br><br># 既存モデルの新バージョンをログ・登録<br>mlflow.sklearn.log_model(<br>    sk_model=retrained_model,<br>    artifact_path="model",<br>    registered_model_name="churn_prediction"  # レジストリ内の既存モデル<br>)</div><br>churn_predictionに既にv1、v2が存在する場合、このコードはv3を追加します。<br><br><strong>他の選択肢が不正解な理由：</strong><br>・「mlflow.register_model呼び出しが不要」：部分的に正しい（log_modelがログと登録の両方を実行）が、重要なポイントはバージョニング<br>・「新しいモデルを記録」：model_nameが存在する場合、新規作成ではなくバージョン追加<br>・「Experimentでのモデル名」：Experimentは実行を追跡するが、registered_model_nameはレジストリを参照<br>・「Runの名前」：Run名は別途設定（mlflow.start_run(run_name="...")）<br><br><strong>重要ポイント：</strong><br>registered_model_name=model_nameを使用して、MLflowレジストリ内の既存モデルを自動バージョニング。各実行で新バージョン（v1、v2等）を追加し、トレーサビリティとロールバックを可能にします。`
    },
    {
        number: 48,
        domain: "Distributed Computing Concepts",
        question: "分散コンピューティングの文脈で、機械学習モデルのスケーリングとは何を指しますか？",
        keyPoint: "大規模な機械学習タスクの処理",
        choices: [
            "個々のアルゴリズムのサイズを増やす",
            "データセットのサイズを拡大する",
            "ハードウェアリソースを調整する",
            "大規模な機械学習タスクを処理する"
        ],
        correctIndex: 3,
        explanation: `<strong>正解：大規模な機械学習タスクを処理する</strong><br><br>分散コンピューティングにおけるMLモデルのスケーリングは、複数のノードやクラスター全体で大規模なデータセットと複雑な計算を効率的に処理する能力を指します。<br><br><strong>含まれる内容：</strong><br>・訓練の並列化（Spark MLlib、Horovod、TensorFlow分散戦略等を使用）<br>・推論の分散（RayやKubernetes等でモデルを提供）<br>・リソース管理（Databricksでクラスターの自動スケーリング等）<br><br><strong>焦点：</strong><br>アルゴリズムやデータサイズだけでなく、システムレベルのスケーラビリティに焦点を当てています。<br><br><strong>分散訓練の例（Spark MLlib）：</strong><br><div class="code-block">from pyspark.ml.classification import LogisticRegression<br><br># Sparkは自動的にワーカー間でモデル訓練を分散<br>lr = LogisticRegression(maxIter=10)<br>model = lr.fit(large_spark_df)  # 大規模データを処理</div><br><strong>他の選択肢が不正解な理由：</strong><br>・「個々のアルゴリズムのサイズを増やす」：モデルの複雑さ（深いニューラルネット等）を指し、分散スケーリングではない<br>・「データセットのサイズを拡大」：スケーリングは大規模データを伴うが、用語はシステムがそれを処理する方法を指す<br>・「ハードウェアリソース調整」：スケーリングの一部（GPU追加等）だが、より広い概念にはソフトウェア/アルゴリズム最適化も含まれる<br><br><strong>重要ポイント：</strong><br>分散システムでのMLモデルのスケーリングは、並列性とリソース最適化を活用し、クラスター全体で大規模タスク（訓練/推論）を効率的に管理することを意味します。`
    },
    {
        number: 49,
        domain: "Cluster Creation and Management",
        question: "分散コンピューティング能力を必要とする大規模な機械学習プロジェクトに取り組んでいます。Databricksでこのようなタスク向けに設計されたクラスタータイプはどれですか？",
        keyPoint: "マルチノードクラスタは分散処理向け",
        choices: [
            "標準クラスタ（Standard Cluster）",
            "マルチノードクラスタ（Multi-node Cluster）",
            "シングルノードクラスタ（Single-node Cluster）",
            "タスク固有クラスタ（Task-specific Cluster）"
        ],
        correctIndex: 1,
        explanation: `<strong>正解：マルチノードクラスタ（Multi-node Cluster）</strong><br><br>Databricksのマルチノードクラスタは、分散コンピューティング向けに特別に設計されており、ワークロード（大規模ML訓練等）を複数のワーカーノード間で並列化します。<br><br><strong>特徴：</strong><br>・Apache Sparkを基盤とし、データと計算を分割してビッグデータとMLタスクの水平スケーリングを実現<br>・ワーカーノード：タスクを並列実行（Sparkエグゼキューター）<br>・ドライバーノード：クラスターを調整（Sparkドライバー）<br>・自動スケーリング：ワークロードに基づいてノードを動的調整<br><br><strong>ユースケース例：</strong><br>・テラバイト規模のデータでディープラーニングモデルを訓練<br>・HyperOptで分散ハイパーパラメータチューニングを実行<br><br><strong>分散ML訓練の例：</strong><br><div class="code-block">from pyspark.ml.classification import RandomForestClassifier<br>model = RandomForestClassifier().fit(train_df)  # 複数ノードで実行</div><br><strong>他の選択肢が不正解な理由：</strong><br>・標準クラスタ：汎用クラスタ（マルチノード可能）だが、「マルチノード」ほど分散コンピューティングを明示的に強調していない<br>・シングルノードクラスタ：1つのノードのみ（並列性なし）。小規模タスクやテストには有用だが、大規模MLには不適<br>・タスク固有クラスタ：標準的なDatabricks用語ではない。GPUクラスタ等はマルチノードクラスタのサブセット<br><br><strong>ヒント：</strong><br>Databricks Runtime for ML（TensorFlow/PyTorchプリインストール）を使用してパフォーマンス最適化。`
    },
    {
        number: 50,
        domain: "Hyperopt & Sparktail",
        question: "fmin()関数のearly_stop_fn引数の目的と使用方法は何ですか？",
        keyPoint: "max_evalsに達する前に早期停止",
        choices: [
            "fmin()呼び出しが取ることができる最大秒数を定義する",
            "Hyperoptの適応性レベルを指定する",
            "max_evalsに達する前に停止するオプションの早期停止関数として機能する",
            "同時に評価する最大試行数を制御する"
        ],
        correctIndex: 2,
        explanation: `<strong>正解：max_evalsに達する前に停止するオプションの早期停止関数として機能する</strong><br><br><strong>目的：</strong><br>early_stop_fn引数を使用すると、最大評価数（max_evals）に達していなくても、ハイパーパラメータ最適化プロセスを早期に停止するカスタム関数を定義できます。これにより、進捗が停滞した場合や目標パフォーマンスに達した場合、不要な評価を回避して時間とリソースを節約できます。<br><br><strong>動作方法：</strong><br>1. 現在のTrialsオブジェクトを入力とし、ブール値を返す関数を作成（Trueで停止、Falseで継続）<br>2. fmin()呼び出し時にearly_stop_fn引数としてこの関数を渡す<br>3. 各試行後にHyperoptが早期停止関数を呼び出す<br>4. 関数がTrueを返すと、Hyperoptは最適化プロセスを終了<br><br><strong>一般的なユースケース：</strong><br>・損失の改善なし：一定数の試行で損失が改善しない場合に停止<br>・目標パフォーマンス達成：目標メトリック（精度等）に到達した場合に停止<br>・時間制約：最大時間制限を超えた場合に停止<br><br><strong>例：</strong><br><div class="code-block">from hyperopt import Trials, fmin, tpe<br>from hyperopt.early_stop import no_progress_loss<br><br># 20試行で損失が改善しない場合に停止する早期停止関数を定義<br>early_stop_fn = no_progress_loss(20)<br><br># fmin()でearly_stop_fnを使用<br>best = fmin(fn=objective, space=space, algo=tpe.suggest,<br>            max_evals=500, trials=Trials(),<br>            early_stop_fn=early_stop_fn)</div><br><strong>他の選択肢が不正解な理由：</strong><br>・「最大秒数を定義」：timeout引数が最大時間制限を制御<br>・「適応性レベルを指定」：適応性は検索アルゴリズム自体によって決定され、early_stop_fnではない<br>・「同時試行数を制御」：max_queue_len引数が同時試行を制御`
    },
    {
        number: 51,
        domain: "Hyperopt & Sparktail",
        question: "Hyperoptのような確率的検索アルゴリズムを使用する場合、各実行で損失が単調に減少しない理由は何ですか？",
        keyPoint: "探索のため損失増加を許容し収束を加速",
        choices: [
            "確率的検索アルゴリズムは常に損失を単調に減少させる",
            "Hyperoptのバグにより損失が減少しない",
            "Hyperoptの検索アルゴリズムはより速い収束を目指し、時折の損失増加を許容する",
            "単調な損失減少はHyperoptの要件である"
        ],
        correctIndex: 2,
        explanation: `<strong>正解：Hyperoptの検索アルゴリズムはより速い収束を目指し、時折の損失増加を許容する</strong><br><br>確率的検索アルゴリズム（HyperoptのTPE等）は、ハイパーパラメータ空間を確率的に探索し、貪欲ではありません。<br><br><strong>理由：</strong><br>・一時的により悪い構成を受け入れることで：<br>  - 局所最小値から脱出<br>  - 長期的により良い結果をもたらす可能性のある有望な領域を探索<br>・この非単調な動作は設計によるものであり、より良い大域最適解につながることが多い<br><br><strong>Hyperoptの例：</strong><br><div class="code-block">from hyperopt import fmin, tpe, Trials<br><br>def objective(params):<br>    # ランダム性を含むシミュレーション損失（検証精度等）<br>    loss = (params["x"] - 2) ** 2 + np.random.normal(0, 0.1)<br>    return loss<br><br>trials = Trials()<br>best = fmin(<br>    fn=objective,<br>    space={"x": hp.uniform("x", -10, 10)},<br>    algo=tpe.suggest,<br>    max_evals=50,<br>    trials=trials,<br>)<br># 探索と活用のトレードオフにより損失が変動する可能性がある</div><br><strong>他の選択肢が不正解な理由：</strong><br>・「常に単調減少」：決定論的アルゴリズム（勾配降下法等）のみが単調性を保証<br>・「バグによる非減少」：非単調性は期待される動作であり、バグではない<br>・「単調性が要件」：Hyperoptは明示的にこれを強制せず、探索を優先<br><br><strong>ヒント：</strong><br>Trials.trialsを使用して進捗を追跡し、必要に応じてmax_evals/early_stop_fnを調整。試行数に対する損失を可視化して、探索ノイズと真の収束問題を区別。`
    },
    {
        number: 52,
        domain: "Pandas API on Spark",
        question: "Pandas API on Sparkで特定のオプション値を使用してコードを実行するにはどうすればよいですか？",
        keyPoint: "option_contextコンテキストマネージャ使用",
        choices: [
            "config.pyファイルを使用",
            "Spark構成を直接変更",
            "set_option()関数を使用",
            "option_contextコンテキストマネージャを使用"
        ],
        correctIndex: 3,
        explanation: `<strong>正解：option_contextコンテキストマネージャを使用</strong><br><br>Pandas API on Spark（旧Koalas）のoption_contextコンテキストマネージャを使用すると、特定のコードブロック内でオプションを一時的に設定でき、変更がグローバルに永続化されないことが保証されます。<br><br><strong>理由：</strong><br>・テストや動作調整（表示フォーマット、計算モード等）に理想的<br>・コードの他の部分に影響を与えずに変更可能<br><br><strong>使用例：</strong><br><div class="code-block">import pyspark.pandas as ps<br><br># 一時的に最大表示行数を10に設定<br>with ps.option_context("display.max_rows", 10):<br>    print(ps.DataFrame(range(100)))  # 10行のみ表示</div><br><strong>一般的なオプション：</strong><br>・compute.max_rows：遅延評価のために計算される行数を制限<br>・display.max_columns：列の表示を制御<br><br><strong>他の選択肢が不正解な理由：</strong><br>・config.pyファイル：Pandas API on Sparkは独立したconfigファイルを使用せず、Sparkセッションやランタイムオプションに依存<br>・Spark構成の変更：spark.conf経由の変更（spark.sql.shuffle.partitions等）はSpark SQLに影響し、Pandas API on Sparkオプションには影響しない<br>・set_option()関数：Pandasにはpd.set_option()があるが、Pandas API on Sparkはスコープ付き変更にps.option_contextを使用<br><br><strong>重要ポイント：</strong><br>Pandas API on Sparkでの一時的でローカライズされたオプション変更には、常に以下を使用：<br><div class="code-block">with ps.option_context("option_name", value):<br>    # コードをここに記述</div><br>これにより副作用を回避し、Sparkの遅延評価モデルに整合します。<br><br><strong>補足：</strong><br>グローバル設定にはps.set_option("compute.default_index_type", "distributed")を使用可能ですが、安全性のためoption_contextを推奨。`
    },
    {
        number: 53,
        domain: "Pandas API on Spark",
        question: "pandas-on-Spark DataFrameとpandas DataFrameの主な違いは何ですか？",
        keyPoint: "前者は分散、後者は単一マシン",
        choices: [
            "前者は分散で、後者は単一マシンである",
            "前者は単一マシンで、後者は分散である",
            "分散の点で同一である",
            "前者の方が後者よりも高度な機能を持つ"
        ],
        correctIndex: 0,
        explanation: `<strong>正解：前者は分散で、後者は単一マシンである</strong><br><br><strong>Pandas-on-Spark DataFrame（pyspark.pandas）：</strong><br>・Apache Spark上で実行され、データと計算をクラスター全体に分散<br>・並列処理により大規模データセット（TB以上）にスケール<br>・例：<br><div class="code-block">import pyspark.pandas as ps<br>df = ps.read_csv("s3://large_dataset.csv")  # 分散読み込み</div><br><strong>Pandas DataFrame：</strong><br>・単一マシン上で実行され、RAMとCPUによって制限される<br>・小～中規模データ（GB単位）に最適<br>・例：<br><div class="code-block">import pandas as pd<br>df = pd.read_csv("local_data.csv")  # 単一ノード読み込み</div><br><strong>主要な違い：</strong><br><table><tr><th>機能</th><th>Pandas-on-Spark</th><th>Pandas</th></tr><tr><td>実行</td><td>分散（Spark）</td><td>単一ノード</td></tr><tr><td>スケーラビリティ</td><td>TB以上のデータ処理</td><td>RAMで制限</td></tr><tr><td>API互換性</td><td>Pandas APIを模倣</td><td>ネイティブPandas</td></tr></table><br><strong>他の選択肢が不正解な理由：</strong><br>・「Pandas-on-Sparkが単一マシン、Pandasが分散」：逆。Pandas-on-SparkはSparkの分散エンジンを活用<br>・「分散の点で同一」：Pandasにはネイティブな分散機能がない<br>・「Pandas-on-Sparkの方が機能が多い」：両者は類似APIを共有するが、核心的な違いは分散<br><br><strong>重要ポイント：</strong><br>ビッグデータ（Sparkクラスター）にはpandas-on-Sparkを、小規模データ（単一マシンワークフロー）にはPandasを使用。<br><br><strong>ヒント：</strong><br>両者間の変換：<br><div class="code-block">pandas_df = ps_df.to_pandas()  # 分散 → 単一ノード<br>ps_df = ps.from_pandas(pandas_df)  # 単一ノード → 分散</div>`
    },
    {
        number: 54,
        domain: "Feature Store",
        question: "Unity Catalogのフィーチャーテーブルへのアクセスを制御するにはどうすればよいですか？",
        keyPoint: "Catalog ExplorerのPermissionsボタン使用",
        choices: [
            "fe.control_access関数を使用",
            "Catalog Explorerテーブル詳細ページのPermissionsボタンを使用",
            "fe.manage_permissions関数を使用",
            "fe.update_access関数を使用"
        ],
        correctIndex: 1,
        explanation: `<strong>正解：Catalog Explorerテーブル詳細ページのPermissionsボタンを使用</strong><br><br>Unity Catalogは、フィーチャーテーブルを含むテーブルへのアクセスを管理するための一元化されたガバナンスインターフェースを提供します。<br><br><strong>Permissions機能：</strong><br>Catalog Explorer UI（Databricksワークスペース）のPermissionsボタンを使用すると：<br>・ユーザー/グループへの読み取り/書き込みアクセスを付与/取り消し<br>・行レベル/列レベルのセキュリティを設定（細粒度の権限）<br>・フィーチャーテーブルの所有権を管理<br><br><strong>アクセス構成手順：</strong><br>1. Catalog Explorer → フィーチャーテーブルを選択 → Permissionsをクリック<br>2. ユーザー/グループを追加し、ロール（SELECT、MODIFY、OWN）を割り当て<br><br><strong>他の選択肢が不正解な理由：</strong><br>・fe.control_access / fe.manage_permissions / fe.update_access：<br>  Unity CatalogやFeature Engineering（FE）APIには存在しない<br>  権限はUI/REST API/SDK（databricks-permissions CLI等）経由で管理され、架空の関数ではない<br><br><strong>重要ポイント：</strong><br>Unity Catalogのフィーチャーテーブルには常に以下を使用：<br>・UI：Catalog Explorer → Permissions<br>・API/SDK：Databricks TerraformプロバイダーまたはREST API（/api/2.1/unity-catalog/permissions）<br><br><strong>ヒント：</strong><br>Delta Sharingと組み合わせて、組織間でのセキュアなアクセスを実現。`
    },
    {
        number: 55,
        domain: "Scaling ML Models",
        question: "機械学習モデルのブースティングをどのように特徴付けますか？",
        keyPoint: "各モデルが前のモデルの誤差から学習",
        choices: [
            "ブースティングは、各モデルが前のモデルの誤差から学習しながら、機械学習モデルを順次訓練するアンサンブルプロセスである",
            "ブースティングは、訓練データのブートストラップサンプルのセットで各サンプルに対して機械学習モデルを訓練し、各モデルの予測を組み合わせて最終推定値を得るアンサンブルプロセスである",
            "ブースティングは、各モデルがデータの異なるサブセットで訓練される機械学習モデルを順次訓練するアンサンブルプロセスである",
            "ブースティングは、各モデルが訓練データの徐々に大きなサンプルで訓練される機械学習モデルを順次訓練するアンサンブルプロセスである",
            "ブースティングは、訓練データのブートストラップサンプルのセットで各サンプルに対して機械学習モデルを訓練し、次にモデル推定値を訓練セットのフィーチャー変数として追加し、それを使用して別のモデルを訓練するアンサンブルプロセスである"
        ],
        correctIndex: 0,
        explanation: `<strong>正解：各モデルが前のモデルの誤差から学習しながら機械学習モデルを順次訓練するアンサンブルプロセス</strong><br><br>ブースティングは、各新しいモデル（弱学習器）が前のモデルの誤差を修正することに焦点を当てる逐次アンサンブル手法です。<br><br><strong>主要な特性：</strong><br>・適応学習：モデルは以前の誤りに基づいて調整される<br>・重み付き投票：最終予測はモデルを重みで組み合わせる（精度の高いモデルほど影響力が大きい）<br><br><strong>例（AdaBoost、Gradient Boosting、XGBoost/LightGBM）：</strong><br>・AdaBoost：誤分類されたサンプルを再重み付け<br>・Gradient Boosting：残差を適合<br>・XGBoost/LightGBM：最適化された勾配ブースティング<br><br><strong>コード例（AdaBoost）：</strong><br><div class="code-block">from sklearn.ensemble import AdaBoostClassifier<br>from sklearn.tree import DecisionTreeClassifier<br><br># 各ツリーが前のツリーの誤差を修正<br>model = AdaBoostClassifier(<br>    estimator=DecisionTreeClassifier(max_depth=1),  # 弱学習器（スタンプ）<br>    n_estimators=50,<br>    learning_rate=1.0<br>)<br>model.fit(X_train, y_train)</div><br><strong>他の選択肢が不正解な理由：</strong><br>・「ブートストラップサンプルで訓練」：バギング（ランダムフォレスト等）を説明。ブースティングではない<br>・「異なるサブセットで訓練」：並列訓練（バギング）を示唆。ブースティングは逐次<br>・「徐々に大きなサンプル」：ブースティングは全データセットを使用するが、サンプルを再重み付け（段階的なサイズ変更なし）<br>・「推定値をフィーチャーとして追加」：スタッキングに似ており、ブースティングではない<br><br><strong>重要ポイント：</strong><br>ブースティングの核心は逐次誤差修正です。高バイアス（アンダーフィッティング）問題に強力ですが、過学習を避けるために慎重なチューニングが必要です。<br><br><strong>補足：</strong><br>XGBoost/LightGBMでearly_stopping_roundsを使用して、検証パフォーマンスが停滞した場合に訓練を停止。`
    },
    {
        number: 56,
        domain: "ML Workflows",
        question: "どのようなシナリオでStringIndexerを使用すべきですか？",
        keyPoint: "カテゴリカル変数として識別させる",
        choices: [
            "機械学習アルゴリズムに列をカテゴリカル変数として識別させたい場合",
            "データ型を知らずにカテゴリカルと非カテゴリカルデータを区別したい場合",
            "最終出力列をテキスト表現に戻したい場合",
            "入力データで次元削減を実行したい場合"
        ],
        correctIndex: 0,
        explanation: `<strong>正解：機械学習アルゴリズムに列をカテゴリカル変数として識別させたい場合</strong><br><br>PySpark MLのStringIndexerは、文字列/カテゴリカル列を数値インデックスに変換するために使用されます。これはほとんどのMLアルゴリズムに必要です（決定木、ロジスティック回帰等）。<br><br><strong>使用方法：</strong><br><div class="code-block">from pyspark.ml.feature import StringIndexer<br><br>indexer = StringIndexer(inputCol="category", outputCol="category_index")<br>indexed_df = indexer.fit(df).transform(df)</div><br>変換：<br><table><tr><th>category</th><th>category_index</th></tr><tr><td>"dog"</td><td>0.0</td></tr><tr><td>"cat"</td><td>1.0</td></tr></table><br><strong>主なユースケース：</strong><br>MLパイプライン用のカテゴリカルテキストデータ（"red"、"blue"等）の準備<br><br><strong>他の選択肢が不正解な理由：</strong><br>・「カテゴリカルと非カテゴリカルを区別」：StringIndexerはデータ型を検出せず、カテゴリカル列を指定する必要がある<br>・「出力をテキストに戻す」：IndexToString（StringIndexerの逆）が行う<br>・「次元削減」：StringIndexerは次元を削減しない。PCAやFeatureHasherを使用<br><br><strong>重要ポイント：</strong><br>訓練前にカテゴリカル文字列を数値インデックスとしてエンコードするためにStringIndexerを使用。アルゴリズムがワンホットベクトルを期待する場合（線形モデル等）は、常にOneHotEncoderと組み合わせる。<br><br><strong>ヒント：</strong><br>順序カテゴリ（"low"、"medium"、"high"等）にはOrdinalEncoderを使用。`
    },
    {
        number: 57,
        domain: "Scaling ML Models",
        question: "分散決定木の文脈で、ランダムフォレストのようなアンサンブル手法を単一の決定木よりも使用する主な利点は何ですか？",
        keyPoint: "過学習しにくい",
        choices: [
            "アンサンブル手法は単一の決定木よりも解釈可能である",
            "アンサンブル手法は単一の決定木よりも過学習しにくい",
            "アンサンブル手法は単一の決定木よりも速く訓練できる",
            "アンサンブル手法は単一の決定木よりもメモリを必要としない",
            "上記のいずれでもない"
        ],
        correctIndex: 1,
        explanation: `<strong>正解：アンサンブル手法は単一の決定木よりも過学習しにくい</strong><br><br>ランダムフォレスト（アンサンブル手法）は、バギング（ブートストラップ集約）とフィーチャーランダム性を通じて複数の決定木を結合します。<br><br><strong>利点：</strong><br>・多様な木からの予測を平均化することで分散を削減<br>・過学習を軽減（訓練データのノイズを記憶する単一木の一般的な問題）<br>・例：単一木は外れ値に過学習する可能性があるが、フォレストの多数決はより堅牢<br><br><strong>分散設定での主な利点：</strong><br>Spark MLlibでは、ランダムフォレストの分散訓練がワーカー間で木の構築を並列化し、アンサンブルのスケーラビリティと汎化の利点を維持します。<br><br><strong>他の選択肢が不正解な理由：</strong><br>・「より解釈可能」：アンサンブルは解釈しにくい（単一木より決定を追跡するのが困難）<br>・「より速く訓練」：アンサンブルは遅い（複数の木を訓練）が、より正確<br>・「メモリ削減」：アンサンブルはより多くのメモリを使用（複数の木を保存）<br><br><strong>重要ポイント：</strong><br>分散MLでは、ランダムフォレストがバイアスと分散のバランスを取り、特にノイズの多いデータや高次元データで単一木を上回ります。スケーラブルな訓練にはPySpark MLlibのRandomForestClassifierを使用。<br><br><strong>補足：</strong><br>numTreesとmaxDepthを調整してバイアス-分散トレードオフを最適化。`
    },
    {
        number: 58,
        domain: "ML Workflows",
        question: "データサイエンティストが2つの回帰モデルを作成しました。最初のモデルはpriceをラベル変数として使用し、2番目のモデルはlog(price)をラベル変数として使用します。ラベル予測を実際のprice値と比較して各モデルのRMSEを評価すると、2番目のモデルのRMSEが最初のモデルよりもはるかに大きいことに気付きました。この違いに対するどの説明が有効ですか？",
        keyPoint: "log予測を指数化してから評価する必要",
        choices: [
            "2番目のモデルは最初のモデルよりもはるかに正確である",
            "データサイエンティストはRMSE計算前に最初のモデルの予測の対数を取らなかった",
            "データサイエンティストはRMSE計算前に2番目のモデルの予測を指数化しなかった",
            "RMSEは回帰問題の無効な評価メトリックである",
            "最初のモデルは2番目のモデルよりもはるかに正確である"
        ],
        correctIndex: 2,
        explanation: `<strong>正解：データサイエンティストはRMSE計算前に2番目のモデルの予測を指数化しなかった</strong><br><br>対数変換されたラベル（log(price)等）でモデルを訓練する場合、モデルの予測も対数空間になります。元のスケール（実際のprice等）と比較するには、変換を逆にする（予測を指数化する）必要があります。<br><br><strong>理由：</strong><br>・モデル1：priceを直接予測 → RMSEは同じスケールで計算される<br>・モデル2：log(price)を予測 → 予測をpriceスケールに一致させるために指数化（exp(pred)）する必要がある<br>・指数化しない場合：RMSEはlog(pred)とpriceを比較するため無効（リンゴとオレンジ）<br><br><strong>例：</strong><br><div class="code-block">import numpy as np<br>from sklearn.metrics import mean_squared_error<br><br># 実際のprice<br>actual_prices = np.array([100, 200, 300])<br><br># モデル1：priceを直接予測<br>pred_prices = np.array([110, 190, 310])<br>rmse1 = np.sqrt(mean_squared_error(actual_prices, pred_prices))  # 有効<br><br># モデル2：log(price)を予測<br>pred_log_prices = np.array([4.7, 5.2, 5.7])<br><br># 不正確なRMSE（log(pred) vs. actual price）<br>rmse_incorrect = np.sqrt(mean_squared_error(actual_prices, pred_log_prices))  # 間違い！<br><br># 正確なRMSE（exp(pred) vs. actual price）<br>rmse_correct = np.sqrt(mean_squared_error(actual_prices, np.exp(pred_log_prices)))  # 有効</div><br><strong>他の選択肢が不正解な理由：</strong><br>・「2番目のモデルがはるかに正確」：RMSE比較が欠陥があるため無効（対数 vs. 線形スケール）<br>・「最初のモデルの予測の対数を取らなかった」：最初のモデルはpriceを直接予測。対数変換は不要<br>・「RMSEが無効」：RMSEは回帰の標準メトリックだが、スケールの一貫性が重要<br>・「最初のモデルがはるかに正確」：比較が無効なため、予測が同じスケールでない限り判断不可<br><br><strong>重要ポイント：</strong><br>対数変換されたラベルで訓練されたモデルの場合：<br>・RMSE計算前に予測を指数化（exp(pred)）<br>・元のスケール（price等）で比較<br><br><strong>指数化をスキップした場合の影響：</strong><br>RMSEはlog(pred)値がpriceよりはるかに小さいため、人為的に高く表示されます。`
    },
    {
        number: 59,
        domain: "AutoML",
        question: "Databricks AutoMLを使用して100万レコードと50フィーチャーのデータセットで分類モデルを訓練しています。手動作業を最小限に抑えながら最高のモデルパフォーマンスを確保したい場合、どの構成を適用すべきですか？",
        keyPoint: "primary_metric=f1、timeout_minutes=60",
        choices: [
            "timeout_minutes=10でAutoMLを実行し、最適なアルゴリズムを選択させる",
            "データを手動で前処理してから、exclude_frameworks=[\"XGBoost\"]でAutoMLを実行して訓練を高速化する",
            "primary_metric=\"f1\"、timeout_minutes=60でAutoMLを使用して徹底的な探索を可能にする",
            "AutoMLでフィーチャーエンジニアリングを無効にして予期しない変換を回避する"
        ],
        correctIndex: 2,
        explanation: `<strong>正解：primary_metric="f1"、timeout_minutes=60でAutoMLを使用して徹底的な探索を可能にする</strong><br><br>分類用のDatabricks AutoMLを使用する場合、目標は手動作業を最小限に抑えながら最高のパフォーマンスのモデルを見つけることです。データセットに100万レコードと50フィーチャーがある場合、AutoMLが複数のモデルを効果的に訓練、評価、最適化できるように十分な探索時間が必要です。<br><br><strong>primary_metric="f1"を使用する理由：</strong><br>・F1スコアは精度と再現率のバランスを取るため、不均衡データセットに理想的<br>・モデルが再現率を犠牲にして多数クラスを優遇しないことを保証<br>・データセットにクラス不均衡がある場合、f1は精度よりも優れたメトリック<br><br><strong>timeout_minutes=60を使用する理由：</strong><br>・10分では大規模データセットでの徹底的なモデル探索には短すぎる<br>・60分はAutoMLがさまざまなアルゴリズムを評価し、ハイパーパラメータチューニングとフィーチャー選択を最適化するのに十分な時間を提供<br><br><strong>Databricksでの構成方法：</strong><br><div class="code-block">from databricks.automl import classification<br><br>classification.classify(<br>    input_df=df,<br>    target_col="label",<br>    primary_metric="f1",  # バランスの取れたパフォーマンスを最適化<br>    timeout_minutes=60  # 徹底的なモデル選択のための十分な時間<br>)</div><br><strong>他の選択肢が不正解な理由：</strong><br>・「timeout_minutes=10で実行」：<br>  ❌ 100万レコードと50フィーチャーでさまざまなモデルを適切に評価するには10分では不十分<br>  リスク：早期に停止し、最適なモデルを見つけられない可能性<br><br>・「手動前処理してexclude_frameworks=["XGBoost"]」：<br>  ❌ XGBoostは構造化データに最適なモデルの1つ<br>  XGBoostを除外すると高品質モデルを排除し、精度を低下させる可能性<br>  AutoMLは既に自動フィーチャーエンジニアリングを実行するため、手動前処理の必要性を削減<br><br>・「フィーチャーエンジニアリングを無効化」：<br>  ❌ フィーチャーエンジニアリングはモデルパフォーマンス向上に重要なステップ<br>  AutoMLはスケーリング、エンコーディング、欠損値補完等の技術を使用し、パフォーマンスを向上させる<br>  手動で無効化するとモデルが最適化されない可能性<br><br><strong>最終結論：</strong><br>大規模データセットの場合、Databricks AutoMLを以下で構成：<br>primary_metric="f1"、timeout_minutes=60<br>これにより、モデル探索とハイパーパラメータチューニングのための十分な時間が確保され、より良いパフォーマンスにつながります。`
    },
    {
        number: 60,
        domain: "Model Interpretability",
        question: "ブラックボックスGBMモデルを訓練した後、ステークホルダーに予測を説明する必要があります。SHAP値を生成しますが、100万行での計算が遅すぎます。これをどのように最適化しますか？",
        keyPoint: "1K行にダウンサンプルしてSHAP計算",
        choices: [
            "より速い近似のためにKernel SHAPの代わりにTree SHAPを使用",
            "1K行にダウンサンプルし、このサブセットのみでSHAPを計算",
            "nvidia-smi経由でGPU加速を有効化",
            "SHAP計算前にモデルをメモリにキャッシュ"
        ],
        correctIndex: 1,
        explanation: `<strong>正解：1K行にダウンサンプルし、このサブセットのみでSHAPを計算</strong><br><br>SHAP（SHapley Additive exPlanations）値は、Gradient Boosting Machines（GBM）のようなブラックボックスモデルを解釈するための強力な技術です。しかし、Shapley値の組み合わせ的性質により、SHAP計算は非常に遅くなる可能性があり、特に100万行のような大規模データセットでは時間がかかります。<br><br><strong>ダウンサンプリングが最適なアプローチである理由：</strong><br>・SHAP値は類似インスタンス間で高度に相関している<br>  代表的なサブセット（1K行等）でSHAPを計算することで、完全な計算コストなしに洞察が得られる<br>・計算時間を大幅に削減<br>  SHAPはデータサイズに対して指数的にスケール<br>  100万行のSHAP計算には数時間かかる可能性があるが、1K行では数分<br>・手動介入なしで解釈可能性を保証<br>  ステークホルダーはすべての行のSHAPを必要とせず、一般的な洞察のみ必要<br>  適切に選択されたサンプル（層別化またはランダム）は解釈可能性を保持<br><br><strong>ダウンサンプリングを使用した最適化されたSHAP計算：</strong><br><div class="code-block">import shap<br>import pandas as pd<br>import numpy as np<br>from sklearn.model_selection import train_test_split<br>from lightgbm import LGBMClassifier<br><br># サンプルGBMモデルを訓練<br>X, y = np.random.rand(1000000, 20), np.random.randint(2, size=1000000)<br>X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)<br>model = LGBMClassifier()<br>model.fit(X_train, y_train)<br><br># 1K行にダウンサンプル<br>X_sample = X_test[:1000]<br><br># TreeExplainerを使用（GBM用に最適化）<br>explainer = shap.TreeExplainer(model)<br>shap_values = explainer.shap_values(X_sample)<br><br># SHAPサマリーを可視化<br>shap.summary_plot(shap_values, X_sample)</div><br>TreeExplainerは効率的だが、大規模データセットでは依然として遅い。1K行のみを使用することで、解釈可能性を維持しながらSHAPを実現可能にする。<br><br><strong>他の選択肢が不正解な理由：</strong><br>・「Tree SHAPの代わりにKernel SHAPを使用」：<br>  ❌ Kernel SHAPはモンテカルロ近似に依存するため、さらに遅い<br>  Tree SHAPはGBM用に最適化されており、優先すべき<br><br>・「nvidia-smi経由でGPU加速を有効化」：<br>  ❌ nvidia-smiはGPU使用状況のみを監視。SHAP計算を加速しない<br>  SHAPはGPUサポートが限定的で、GPU加速は主にディープラーニングモデルに利益をもたらし、GBMではない<br><br>・「SHAP計算前にモデルをキャッシュ」：<br>  ❌ SHAP計算はフィーチャーの相互作用に依存し、キャッシュには依存しない<br>  キャッシュはSHAP計算時間を大幅に削減しない<br><br><strong>最終結論：</strong><br>大規模データセット（100万行）の場合、1K行にダウンサンプルし、このサブセットのみでSHAPを計算することで、パフォーマンスを最適化しながら解釈可能性を維持します。`
    },
    {
        number: 61,
        domain: "Scaling ML Models",
        question: "分散決定木の文脈で、ランダムフォレストのようなアンサンブル手法を単一の決定木よりも使用する主な利点は何ですか？",
        keyPoint: "過学習しにくい",
        choices: [
            "アンサンブル手法は単一の決定木よりも解釈可能である",
            "アンサンブル手法は単一の決定木よりも過学習しにくい",
            "アンサンブル手法は単一の決定木よりも速く訓練できる",
            "アンサンブル手法は単一の決定木よりもメモリを必要としない",
            "上記のいずれでもない"
        ],
        correctIndex: 1,
        explanation: `<strong>正解：アンサンブル手法は単一の決定木よりも過学習しにくい</strong><br><br>ランダムフォレスト（アンサンブル手法）は、バギング（ブートストラップ集約）とフィーチャーランダム性を通じて複数の決定木を結合します。<br><br><strong>利点：</strong><br>・多様な木からの予測を平均化することで分散を削減<br>・過学習を軽減（訓練データのノイズを記憶する単一木の一般的な問題）<br>・例：単一木は外れ値に過学習する可能性があるが、フォレストの多数決はより堅牢<br><br><strong>分散設定での主な利点：</strong><br>Spark MLlibでは、ランダムフォレストの分散訓練がワーカー間で木の構築を並列化し、アンサンブルのスケーラビリティと汎化の利点を維持します。<br><br><strong>他の選択肢が不正解な理由：</strong><br>・「より解釈可能」：アンサンブルは解釈しにくい（単一木より決定を追跡するのが困難）<br>・「より速く訓練」：アンサンブルは遅い（複数の木を訓練）が、より正確<br>・「メモリ削減」：アンサンブルはより多くのメモリを使用（複数の木を保存）<br><br><strong>重要ポイント：</strong><br>分散MLでは、ランダムフォレストがバイアスと分散のバランスを取り、特にノイズの多いデータや高次元データで単一木を上回ります。スケーラブルな訓練にはPySpark MLlibのRandomForestClassifierを使用。<br><br><strong>補足：</strong><br>numTreesとmaxDepthを調整してバイアス-分散トレードオフを最適化。`
    },
    {
        number: 62,
        domain: "Feature Store",
        question: "リアルタイム推薦システムを構築しています。user_engagement_30m等のフィーチャーは、ストリーミングDeltaテーブルから計算されます。低レイテンシのモデル推論のために、これらのフィーチャーがFeature Storeで継続的に更新されることを保証する必要があります。最も効率的なアプローチは何ですか？",
        keyPoint: "writeStreamで増分更新",
        choices: [
            "30分ごとにフィーチャーバッチを再計算し、Feature Storeテーブルを上書きするスケジュールジョブを使用",
            "writeStreamを使用してFeature Storeを増分更新するStructured Streamingジョブを設定",
            "新しいデータが到着するたびにfs.write_table()を使用してフィーチャーを再計算するノートブックを手動でトリガー",
            "より高速な更新のためにFeature Storeの代わりにRedisキャッシュにフィーチャーを保存"
        ],
        correctIndex: 1,
        explanation: `<strong>正解：writeStreamを使用してFeature Storeを増分更新するStructured Streamingジョブを設定</strong><br><br>リアルタイム推薦システムでは、user_engagement_30m等のフィーチャーを低レイテンシ推論のために継続的に更新する必要があります。最も効率的なアプローチは、writeStreamでStructured Streamingを使用して、ストリーミングDeltaテーブルに新しいデータが到着したときにFeature Storeを増分更新することです。<br><br><strong>writeStreamでStructured Streamingを使用する理由：</strong><br>・リアルタイムフィーチャー更新<br>  Feature Storeで最新のフィーチャー値が常に利用可能であることを保証<br>・低レイテンシ更新<br>  writeStreamは新しいデータを増分的に取り込み、バッチ再計算と比較してオーバーヘッドを削減<br>・ML推論用に最適化<br>  Feature Storeは増分フィーチャー更新をサポートし、リアルタイムMLアプリケーションに理想的<br><br><strong>例：DatabricksへのストリーミングフィーチャーFeature Store更新：</strong><br><div class="code-block">from databricks.feature_store import feature_table<br>from pyspark.sql.functions import col<br>from pyspark.sql.streaming import DataStreamWriter<br><br># Deltaテーブルからストリーミングフィーチャーを読み込み<br>streaming_df = spark.readStream.format("delta").table("user_engagement")<br><br># Feature Store用にフィーチャーを処理・準備<br>feature_df = streaming_df.select(<br>    col("user_id"),<br>    col("engagement_score"),<br>    col("timestamp")<br>)<br><br># Feature Storeテーブル名を定義<br>feature_table_name = "user_features"<br><br># Structured Streamingを使用してFeature Storeにフィーチャーを書き込み<br>feature_df.writeStream \<br>    .format("delta") \<br>    .outputMode("append") \<br>    .option("checkpointLocation", "/mnt/checkpoints/user_features") \<br>    .table(feature_table_name)</div><br>・ストリーミングDeltaテーブル（user_engagement）から読み込み<br>・処理されたフィーチャーをStructured Streamingを使用してFeature Storeテーブルに増分的に書き込み<br>・フォールトトレランスのためにチェックポイントを維持<br><br><strong>他の選択肢が不正解な理由：</strong><br>・「30分ごとに再計算してスケジュールジョブ」：<br>  ❌ バッチ処理は遅延を導入（30分のラグ）<br>  テーブルの上書きはリアルタイムMLユースケースには非効率的<br>  Structured Streamingがリアルタイム更新のためのより良いアプローチ<br><br>・「新データ到着時に手動でノートブックトリガー」：<br>  ❌ 手動実行はスケーラブルでもリアルタイムでもない<br>  リアルタイム推薦には自動化されたストリーミング更新が必要<br><br>・「RedisキャッシュにFeature Store代替保存」：<br>  ❌ Redisは取得は高速だが、フィーチャーのバージョニング、系統、ML統合をDatabricks Feature Storeのようにサポートしない<br>  Feature StoreはMLワークフロー用に最適化され、モデルサービングと統合<br><br><strong>最終結論：</strong><br>推薦システムでのリアルタイムフィーチャー更新には、Structured Streaming（writeStream）を使用してDatabricks Feature Storeを増分更新し、ML推論のための低レイテンシで最新のフィーチャーを保証します。`
    },
    {
        number: 63,
        domain: "AutoML",
        question: "Databricks AutoMLを使用して100万レコードと50フィーチャーのデータセットで分類モデルを訓練しています。手動作業を最小限に抑えながら最高のモデルパフォーマンスを確保したい場合、どの構成を適用すべきですか？",
        keyPoint: "primary_metric=f1、timeout_minutes=60",
        choices: [
            "timeout_minutes=10でAutoMLを実行し、最適なアルゴリズムを選択させる",
            "データを手動で前処理してから、exclude_frameworks=[\"XGBoost\"]でAutoMLを実行して訓練を高速化する",
            "primary_metric=\"f1\"、timeout_minutes=60でAutoMLを使用して徹底的な探索を可能にする",
            "AutoMLでフィーチャーエンジニアリングを無効にして予期しない変換を回避する"
        ],
        correctIndex: 2,
        explanation: `<strong>正解：primary_metric="f1"、timeout_minutes=60でAutoMLを使用して徹底的な探索を可能にする</strong><br><br>分類用のDatabricks AutoMLを使用する場合、目標は手動作業を最小限に抑えながら最高のパフォーマンスのモデルを見つけることです。データセットに100万レコードと50フィーチャーがある場合、AutoMLが複数のモデルを効果的に訓練、評価、最適化できるように十分な探索時間が必要です。<br><br><strong>primary_metric="f1"を使用する理由：</strong><br>・F1スコアは精度と再現率のバランスを取るため、不均衡データセットに理想的<br>・モデルが再現率を犠牲にして多数クラスを優遇しないことを保証<br>・データセットにクラス不均衡がある場合、f1は精度よりも優れたメトリック<br><br><strong>timeout_minutes=60を使用する理由：</strong><br>・10分では大規模データセットでの徹底的なモデル探索には短すぎる<br>・60分はAutoMLがさまざまなアルゴリズムを評価し、ハイパーパラメータチューニングとフィーチャー選択を最適化するのに十分な時間を提供<br><br><strong>Databricksでの構成方法：</strong><br><div class="code-block">from databricks.automl import classification<br><br>classification.classify(<br>    input_df=df,<br>    target_col="label",<br>    primary_metric="f1",  # バランスの取れたパフォーマンスを最適化<br>    timeout_minutes=60  # 徹底的なモデル選択のための十分な時間<br>)</div><br><strong>他の選択肢が不正解な理由：</strong><br>・「timeout_minutes=10で実行」：<br>  ❌ 100万レコードと50フィーチャーでさまざまなモデルを適切に評価するには10分では不十分<br>  リスク：早期に停止し、最適なモデルを見つけられない可能性<br><br>・「手動前処理してexclude_frameworks=["XGBoost"]」：<br>  ❌ XGBoostは構造化データに最適なモデルの1つ<br>  XGBoostを除外すると高品質モデルを排除し、精度を低下させる可能性<br>  AutoMLは既に自動フィーチャーエンジニアリングを実行するため、手動前処理の必要性を削減<br><br>・「フィーチャーエンジニアリングを無効化」：<br>  ❌ フィーチャーエンジニアリングはモデルパフォーマンス向上に重要なステップ<br>  AutoMLはスケーリング、エンコーディング、欠損値補完等の技術を使用し、パフォーマンスを向上させる<br>  手動で無効化するとモデルが最適化されない可能性<br><br><strong>最終結論：</strong><br>大規模データセットの場合、Databricks AutoMLを以下で構成：<br>primary_metric="f1"、timeout_minutes=60<br>これにより、モデル探索とハイパーパラメータチューニングのための十分な時間が確保され、より良いパフォーマンスにつながります。`
    },
    {
        number: 64,
        domain: "Model Interpretability",
        question: "ブラックボックスGBMモデルを訓練した後、ステークホルダーに予測を説明する必要があります。SHAP値を生成しますが、100万行での計算が遅すぎます。これをどのように最適化しますか？",
        keyPoint: "1K行にダウンサンプルしてSHAP計算",
        choices: [
            "より速い近似のためにKernel SHAPの代わりにTree SHAPを使用",
            "1K行にダウンサンプルし、このサブセットのみでSHAPを計算",
            "nvidia-smi経由でGPU加速を有効化",
            "SHAP計算前にモデルをメモリにキャッシュ"
        ],
        correctIndex: 1,
        explanation: `<strong>正解：1K行にダウンサンプルし、このサブセットのみでSHAPを計算</strong><br><br>SHAP（SHapley Additive exPlanations）値は、Gradient Boosting Machines（GBM）のようなブラックボックスモデルを解釈するための強力な技術です。しかし、Shapley値の組み合わせ的性質により、SHAP計算は非常に遅くなる可能性があり、特に100万行のような大規模データセットでは時間がかかります。<br><br><strong>ダウンサンプリングが最適なアプローチである理由：</strong><br>・SHAP値は類似インスタンス間で高度に相関している<br>  代表的なサブセット（1K行等）でSHAPを計算することで、完全な計算コストなしに洞察が得られる<br>・計算時間を大幅に削減<br>  SHAPはデータサイズに対して指数的にスケール<br>  100万行のSHAP計算には数時間かかる可能性があるが、1K行では数分<br>・手動介入なしで解釈可能性を保証<br>  ステークホルダーはすべての行のSHAPを必要とせず、一般的な洞察のみ必要<br>  適切に選択されたサンプル（層別化またはランダム）は解釈可能性を保持<br><br><strong>ダウンサンプリングを使用した最適化されたSHAP計算：</strong><br><div class="code-block">import shap<br>import pandas as pd<br>import numpy as np<br>from sklearn.model_selection import train_test_split<br>from lightgbm import LGBMClassifier<br><br># サンプルGBMモデルを訓練<br>X, y = np.random.rand(1000000, 20), np.random.randint(2, size=1000000)<br>X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)<br>model = LGBMClassifier()<br>model.fit(X_train, y_train)<br><br># 1K行にダウンサンプル<br>X_sample = X_test[:1000]<br><br># TreeExplainerを使用（GBM用に最適化）<br>explainer = shap.TreeExplainer(model)<br>shap_values = explainer.shap_values(X_sample)<br><br># SHAPサマリーを可視化<br>shap.summary_plot(shap_values, X_sample)</div><br>TreeExplainerは効率的だが、大規模データセットでは依然として遅い。1K行のみを使用することで、解釈可能性を維持しながらSHAPを実現可能にする。<br><br><strong>他の選択肢が不正解な理由：</strong><br>・「Tree SHAPの代わりにKernel SHAPを使用」：<br>  ❌ Kernel SHAPはモンテカルロ近似に依存するため、さらに遅い<br>  Tree SHAPはGBM用に最適化されており、優先すべき<br><br>・「nvidia-smi経由でGPU加速を有効化」：<br>  ❌ nvidia-smiはGPU使用状況のみを監視。SHAP計算を加速しない<br>  SHAPはGPUサポートが限定的で、GPU加速は主にディープラーニングモデルに利益をもたらし、GBMではない<br><br>・「SHAP計算前にモデルをキャッシュ」：<br>  ❌ SHAP計算はフィーチャーの相互作用に依存し、キャッシュには依存しない<br>  キャッシュはSHAP計算時間を大幅に削減しない<br><br><strong>最終結論：</strong><br>大規模データセット（100万行）の場合、1K行にダウンサンプルし、このサブセットのみでSHAPを計算することで、パフォーマンスを最適化しながら解釈可能性を維持します。`
    },
    {
        number: 65,
        domain: "Feature Store",
        question: "リアルタイム推薦システムを構築しています。user_engagement_30m等のフィーチャーは、ストリーミングDeltaテーブルから計算されます。低レイテンシのモデル推論のために、これらのフィーチャーがFeature Storeで継続的に更新されることを保証する必要があります。最も効率的なアプローチは何ですか？",
        keyPoint: "writeStreamで増分更新",
        choices: [
            "30分ごとにフィーチャーバッチを再計算し、Feature Storeテーブルを上書きするスケジュールジョブを使用",
            "writeStreamを使用してFeature Storeを増分更新するStructured Streamingジョブを設定",
            "新しいデータが到着するたびにfs.write_table()を使用してフィーチャーを再計算するノートブックを手動でトリガー",
            "より高速な更新のためにFeature Storeの代わりにRedisキャッシュにフィーチャーを保存"
        ],
        correctIndex: 1,
        explanation: `<strong>正解：writeStreamを使用してFeature Storeを増分更新するStructured Streamingジョブを設定</strong><br><br>リアルタイム推薦システムでは、user_engagement_30m等のフィーチャーを低レイテンシ推論のために継続的に更新する必要があります。最も効率的なアプローチは、writeStreamでStructured Streamingを使用して、ストリーミングDeltaテーブルに新しいデータが到着したときにFeature Storeを増分更新することです。<br><br><strong>writeStreamでStructured Streamingを使用する理由：</strong><br>・リアルタイムフィーチャー更新<br>  Feature Storeで最新のフィーチャー値が常に利用可能であることを保証<br>・低レイテンシ更新<br>  writeStreamは新しいデータを増分的に取り込み、バッチ再計算と比較してオーバーヘッドを削減<br>・ML推論用に最適化<br>  Feature Storeは増分フィーチャー更新をサポートし、リアルタイムMLアプリケーションに理想的<br><br><strong>例：DatabricksへのストリーミングフィーチャーFeature Store更新：</strong><br><div class="code-block">from databricks.feature_store import feature_table<br>from pyspark.sql.functions import col<br>from pyspark.sql.streaming import DataStreamWriter<br><br># Deltaテーブルからストリーミングフィーチャーを読み込み<br>streaming_df = spark.readStream.format("delta").table("user_engagement")<br><br># Feature Store用にフィーチャーを処理・準備<br>feature_df = streaming_df.select(<br>    col("user_id"),<br>    col("engagement_score"),<br>    col("timestamp")<br>)<br><br># Feature Storeテーブル名を定義<br>feature_table_name = "user_features"<br><br># Structured Streamingを使用してFeature Storeにフィーチャーを書き込み<br>feature_df.writeStream \<br>    .format("delta") \<br>    .outputMode("append") \<br>    .option("checkpointLocation", "/mnt/checkpoints/user_features") \<br>    .table(feature_table_name)</div><br>・ストリーミングDeltaテーブル（user_engagement）から読み込み<br>・処理されたフィーチャーをStructured Streamingを使用してFeature Storeテーブルに増分的に書き込み<br>・フォールトトレランスのためにチェックポイントを維持<br><br><strong>他の選択肢が不正解な理由：</strong><br>・「30分ごとに再計算してスケジュールジョブ」：<br>  ❌ バッチ処理は遅延を導入（30分のラグ）<br>  テーブルの上書きはリアルタイムMLユースケースには非効率的<br>  Structured Streamingがリアルタイム更新のためのより良いアプローチ<br><br>・「新データ到着時に手動でノートブックトリガー」：<br>  ❌ 手動実行はスケーラブルでもリアルタイムでもない<br>  リアルタイム推薦には自動化されたストリーミング更新が必要<br><br>・「RedisキャッシュにFeature Store代替保存」：<br>  ❌ Redisは取得は高速だが、フィーチャーのバージョニング、系統、ML統合をDatabricks Feature Storeのようにサポートしない<br>  Feature StoreはMLワークフロー用に最適化され、モデルサービングと統合<br><br><strong>最終結論：</strong><br>推薦システムでのリアルタイムフィーチャー更新には、Structured Streaming（writeStream）を使用してDatabricks Feature Storeを増分更新し、ML推論のための低レイテンシで最新のフィーチャーを保証します。`
    }
];

// Make questions2 available globally
if (typeof window !== 'undefined') {
    window.questions2 = questions2;
}
