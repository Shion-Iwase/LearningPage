// DBMLA 試験問題データ3

const questions3 = [
    {
        number: 1,
        domain: "AutoML",
        question: "Scenario: 回帰AutoMLタスクで、生成されたノートブックと実験を特定のディレクトリに保存したい場合、どのパラメータを使用すべきですか？",
        keyPoint: "experiment_dirパラメータでAutoMLの出力先ディレクトリを指定",
        choices: [
            "experiment_dir",
            "feature_store_lookups",
            "exclude_cols",
            "primary_metric"
        ],
        correctIndex: 0,
        explanation: `
            <p>Databricks AutoMLでは、<strong>experiment_dir</strong>パラメータを使用して、AutoMLで生成されるすべてのアセット（ノートブック、実験、アーティファクト）を保存するディレクトリパスを明示的に指定します。</p>

            <p>これにより、実行の組織的で一元的な追跡と再現性が確保されます。</p>

            <div class="code-block">from databricks import automl

automl.regress(
    dataset=df,
    target_col="price",
    experiment_dir="/Shared/automl/price_prediction",  # カスタムディレクトリ
    primary_metric="rmse"
)</div>

            <p>出力先:</p>
            <div class="code-block">/Shared/automl/price_prediction/
├── notebooks/          # 生成されたトレーニングノートブック
├── experiment/        # MLflow実験データ
└── artifacts/         # モデルとメトリクス</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>feature_store_lookups:</strong> Feature Storeテーブルを参照するために使用され、出力場所の制御には使用されません。</li>
                <li><strong>exclude_cols:</strong> トレーニング中に無視する列をリストアップするもので、保存パスとは無関係です。</li>
                <li><strong>primary_metric:</strong> 最適化メトリック（例: "rmse"、"r2"）を定義するもので、ファイルパスではありません。</li>
            </ul>

            <p><strong>重要ポイント:</strong> experiment_dirを使用してAutoML出力の構造化された保存場所を設定します。デフォルトパス（例: /Users/email@company/automl）は散らかりにつながる可能性があります。</p>

            <p><strong>プロのヒント:</strong> MLflowトラッキングと組み合わせて、統一された名前空間の下で実験をログに記録します。</p>
        `
    },
    {
        number: 2,
        domain: "Feature Store",
        question: "Databricksで、Unity CatalogとFeature Storeのクライアントを使用してサポートされているシナリオは何ですか？",
        keyPoint: "Feature Storeは特徴テーブルの作成・読み書き、モデル訓練・スコアリング、オンラインストアへの公開をサポート",
        choices: [
            "特徴テーブルの読み取りのみがサポートされている",
            "特徴テーブルの作成、読み取り、書き込み、特徴データでのモデルのトレーニングとスコアリング、リアルタイム配信のためのオンラインストアへの特徴テーブルの公開がサポートされている",
            "モックフレームワークを使用した単体テストのみがサポートされている",
            "Databricksで実行される統合テストの作成はサポートされていない"
        ],
        correctIndex: 1,
        explanation: `
            <p>Databricks Feature EngineeringのUnity CatalogとFeature Storeは、エンドツーエンドのMLワークフローをサポートします:</p>

            <ul>
                <li><strong>特徴テーブル管理:</strong> Unity Catalogで特徴テーブルを作成、読み取り、更新します。</li>
                <li><strong>モデルトレーニング/スコアリング:</strong> バッチまたはリアルタイム推論のために特徴を使用します（例: FeatureStoreClientのFeatureLookup）。</li>
                <li><strong>オンライン配信:</strong> publish_tableを使用して、低レイテンシストア（例: DynamoDB、Redis）に特徴を公開します。</li>
            </ul>

            <div class="code-block">from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

# 特徴の作成/書き込み
fs.create_table(name="features.price_prediction", df=feature_df)

# 特徴ルックアップを使用したモデルトレーニング
model = train_with_features(fs, training_df)

# オンラインストアへの公開
fs.publish_table("features.price_prediction", online_store="redis")</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>「読み取りのみ」:</strong> Feature StoreはフルCRUD操作をサポートしており、読み取りだけではありません。</li>
                <li><strong>「単体テストのみ」:</strong> テストは可能ですが、Feature Storeの主な範囲ではありません。</li>
                <li><strong>「統合テスト未サポート」:</strong> DatabricksはMLflowとのCI/CDパイプラインなどの統合テストをサポートしています。</li>
            </ul>

            <p><strong>重要ポイント:</strong> Unity CatalogのFeature Engineeringは、特徴作成からオンライン配信まで、DatabricksのMLエコシステムと統合されたフルライフサイクル管理を可能にします。</p>

            <p><strong>プロのヒント:</strong> Feature Monitoring（2024年GA）を使用して、公開されたテーブルのデータドリフトを追跡します。</p>
        `
    },
    {
        number: 3,
        domain: "Databricks ML",
        question: "機械学習モデルがMLflowを使用してDatabricks環境に正常にデプロイされています。モデルは現在、多数のリクエストを受信しており、データサイエンティストは高可用性を確保する必要があります。データサイエンティストはどのような戦略を採用すべきですか？",
        keyPoint: "高可用性のためにREST APIのロードバランシングを実装",
        choices: [
            "Databricksクラスタのサイズを増やす",
            "デプロイされたモデルのREST APIにロードバランシングを実装する",
            "Databricksクラスタの定期的な再起動をスケジュールする",
            "本番デプロイメント用に別のDatabricksワークスペースを使用する"
        ],
        correctIndex: 1,
        explanation: `
            <p>ロードバランシングは、モデルエンドポイントの複数のレプリカに受信推論リクエストを分散し、以下を確保します:</p>

            <ul>
                <li><strong>高可用性:</strong> 単一障害点がありません。</li>
                <li><strong>スケーラビリティ:</strong> レプリカを動的に追加することで、トラフィックの増加に対応します。</li>
                <li><strong>低レイテンシ:</strong> リクエストは最も負荷の少ないインスタンスにルーティングされます。</li>
            </ul>

            <p>DatabricksはMLflow Model Servingと統合されており、水平スケーリング（例: Kubernetesベースのロードバランシング）をサポートしています。</p>

            <div class="code-block"># 複数のモデルサーバー（レプリカ）を起動
mlflow models serve -m models:/prod_model/1 -p 5001 &
mlflow models serve -m models:/prod_model/1 -p 5002 &

# NGINXを設定してロードバランシング
upstream model_servers {
    server localhost:5001;
    server localhost:5002;
}</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>クラスタサイズの増加:</strong> バッチ処理にのみ役立ちます（リアルタイムAPIには適用されません）。Model Servingは別のコンピュートを使用します。</li>
                <li><strong>クラスタの再起動スケジュール:</strong> サービスを中断します。可用性は冗長性によって達成され、再起動では達成されません。</li>
                <li><strong>別のワークスペース:</strong> 環境を分離しますが、リクエスト負荷には対処しません。</li>
            </ul>

            <p><strong>重要ポイント:</strong> 高可用性ML APIには、ロードバランシングが必須です。以下を使用してください:</p>
            <ul>
                <li>Databricksネイティブ配信（レプリカを自動スケール）</li>
                <li>クラウドロードバランサー（例: AWS ELB、Azure Traffic Manager）をハイブリッドセットアップで使用</li>
            </ul>

            <p><strong>プロのヒント:</strong> Databricks Lakeview Dashboardsでレイテンシ/エラーを監視します。</p>
        `
    },
    {
        number: 4,
        domain: "ML Workflows",
        question: "パラメータAの2つの値、パラメータBの5つの値、パラメータCの10つの値からなるハイパーパラメータ空間でグリッドサーチを使用した3分割交差検証を実行する場合、合計で何回のモデル実行が実行されますか？",
        keyPoint: "グリッドサーチ×交差検証の総実行回数 = パラメータ組み合わせ数 × フォールド数",
        choices: [
            "18",
            "300",
            "50",
            "100",
            "上記のいずれでもない"
        ],
        correctIndex: 1,
        explanation: `
            <p>グリッドサーチはハイパーパラメータのすべての組み合わせを評価します。3分割交差検証では、各組み合わせに対してモデルを3回（フォールドごとに1回）トレーニングします。</p>

            <h4>計算:</h4>
            <ul>
                <li><strong>パラメータの組み合わせ:</strong> 2 (A) × 5 (B) × 10 (C) = 100通りのユニークな組み合わせ</li>
                <li><strong>総実行回数:</strong> 100組み合わせ × 3フォールド = <strong>300</strong>モデル実行</li>
            </ul>

            <div class="code-block">from sklearn.model_selection import GridSearchCV

param_grid = {
    'param_A': [val1, val2],          # 2値
    'param_B': [val1, val2, val3, val4, val5],  # 5値
    'param_C': [val1, ..., val10]     # 10値
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_search.fit(X, y)  # 300回実行 (2×5×10 × 3)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>18:</strong> フォールド数とパラメータ数の合計を誤って乗算（3 × (2+5+10)）</li>
                <li><strong>50:</strong> フォールドを無視（2×5×10 = 100だが3分割CVを考慮していない）</li>
                <li><strong>100:</strong> ユニークなパラメータセットのみをカウントし、フォールドを除外</li>
            </ul>

            <p><strong>重要ポイント:</strong> グリッドサーチ + 交差検証の場合、総実行回数 = (Aの値数 × Bの値数 × ...) × フォールド数</p>

            <p><strong>プロのヒント:</strong> 空間が大きい場合は、RandomizedSearchCVを使用して組み合わせをサンプリングします。</p>
        `
    },
    {
        number: 5,
        domain: "Hyperopt and SparkTrials",
        question: "ハイパーパラメータチューニングを開始する際、トレーニング時間が長いモデルに対する推奨事項は何ですか？",
        keyPoint: "小さなデータセットで多くのハイパーパラメータを実験",
        choices: [
            "大きなデータセットと少数のハイパーパラメータで実験する",
            "小さなデータセットと多くのハイパーパラメータで実験する",
            "実験前にすべてのハイパーパラメータを固定する",
            "最高性能モデルの特定にMLflowの使用を避ける"
        ],
        correctIndex: 1,
        explanation: `
            <p>トレーニング時間が長いモデルでは、完全なデータセットで多くのハイパーパラメータの組み合わせをテストすることは非実用的です。</p>

            <h4>推奨アプローチ:</h4>
            <ul>
                <li><strong>小さく始める:</strong> データのサブセットを使用して、ハイパーパラメータのパフォーマンスを迅速に評価します。</li>
                <li><strong>広範囲の検索:</strong> スケールアップする前に有望な範囲を特定するために多くのハイパーパラメータをテストします。</li>
            </ul>

            <h4>ワークフロー例:</h4>
            <ol>
                <li>データの10%を使用して100のハイパーパラメータ組み合わせをテスト</li>
                <li>上位5つの構成に絞り込む</li>
                <li>完全なデータで上位候補を再トレーニング</li>
            </ol>

            <p><strong>実装ツール:</strong></p>
            <ul>
                <li>Hyperopt（並列チューニング用のSparkTrials使用）</li>
                <li>Databricks AutoML（このプロセスを自動化）</li>
            </ul>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>「大規模データセットと少数のハイパーパラメータ」:</strong> 限られた探索で高いコンピュートコスト</li>
                <li><strong>「すべてのハイパーパラメータを固定」:</strong> チューニングの目的を無効にします</li>
                <li><strong>「MLflowを避ける」:</strong> MLflowは実行全体のパフォーマンス追跡に不可欠です</li>
            </ul>

            <p><strong>重要ポイント:</strong> 長時間トレーニングモデルの場合:</p>
            <ul>
                <li>初期ハイパーパラメータスクリーニングのためにデータサイズを削減</li>
                <li>多様な構成をカバーするために検索空間を拡大</li>
                <li>上位候補についてのみスケールアップ</li>
            </ul>

            <p><strong>プロのヒント:</strong> Hyperoptでmax_evalsを使用して試行回数を制御します。</p>
        `
    },
    {
        number: 6,
        domain: "AutoML",
        question: "Databricks AutoMLを使用できる問題のタイプは何ですか？",
        keyPoint: "AutoMLは回帰、分類、予測問題をサポート",
        choices: [
            "回帰問題のみ",
            "分類問題のみ",
            "予測問題のみ",
            "回帰、分類、予測問題"
        ],
        correctIndex: 3,
        explanation: `
            <p>Databricks AutoMLは3つの主要な問題タイプをサポートします:</p>

            <ul>
                <li><strong>回帰:</strong> 連続値の予測（例: 住宅価格）</li>
                <li><strong>分類:</strong> 離散ラベルの予測（例: スパムか否か）</li>
                <li><strong>予測:</strong> 時系列予測（例: 売上予測）</li>
            </ul>

            <p>AutoMLは以下を自動化します:</p>
            <ul>
                <li>特徴エンジニアリング</li>
                <li>モデル選択（例: XGBoost、Random Forest）</li>
                <li>ハイパーパラメータチューニング</li>
            </ul>

            <h4>ワークフロー例:</h4>

            <p><strong>分類:</strong></p>
            <div class="code-block">from databricks import automl

automl.classify(df, target_col="label")</div>

            <p><strong>時系列予測:</strong></p>
            <div class="code-block">automl.forecast(df, time_col="date", target_col="sales")</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>「回帰/分類/予測のみ」:</strong> AutoMLは3つすべてを処理します、1つだけではありません</li>
            </ul>

            <p><strong>重要ポイント:</strong> Databricks AutoMLを以下に使用:</p>
            <ul>
                <li>構造化データ（回帰/分類）</li>
                <li>時系列データ（予測）</li>
                <li>非構造化データ（Delta Lake統合経由）</li>
            </ul>

            <p><strong>プロのヒント:</strong> ディープラーニング（例: 画像/テキスト）の場合、MLflowとHugging Faceを組み合わせます。</p>
        `
    },
    {
        number: 7,
        domain: "AutoML",
        question: "分類問題を扱う際、AutoML実験の各実行で自動的に計算される標準評価メトリックは何ですか？",
        keyPoint: "AutoMLは分類でAccuracy、AUC-ROC、Recall、F1スコアなどすべてを計算",
        choices: [
            "すべて",
            "Accuracy",
            "AUC-ROC",
            "Recall",
            "F1スコア"
        ],
        correctIndex: 0,
        explanation: `
            <p>Databricks AutoMLの分類では、以下を含む複数の標準メトリックを自動的に計算します:</p>

            <ul>
                <li><strong>Accuracy:</strong> 全体的な正確性 ((TP+TN)/Total)</li>
                <li><strong>AUC-ROC:</strong> クラスを区別するモデルの能力（高いほど良い）</li>
                <li><strong>Recall:</strong> 真陽性率 (TP/(TP+FN))</li>
                <li><strong>F1スコア:</strong> 精度と再現率の調和平均</li>
            </ul>

            <p>これらのメトリックは、異なる閾値とクラスの不均衡にわたるモデルパフォーマンスの全体的な視点を提供します。</p>

            <div class="code-block">from databricks import automl

summary = automl.classify(df, target_col="label")
display(summary.trials)  # 試行ごとのすべてのメトリックを表示</div>

            <table border="1">
                <tr>
                    <th>Trial</th>
                    <th>Accuracy</th>
                    <th>AUC-ROC</th>
                    <th>Recall</th>
                    <th>F1</th>
                </tr>
                <tr>
                    <td>1</td>
                    <td>0.92</td>
                    <td>0.98</td>
                    <td>0.91</td>
                    <td>0.93</td>
                </tr>
            </table>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li>個々のメトリック（Accuracy、AUC-ROC、Recall、F1）は完全なセットの一部であり、単独ではありません</li>
            </ul>

            <p><strong>重要ポイント:</strong> AutoMLはデフォルトですべての主要メトリックを使用して分類モデルを評価します。リーダーボードを使用して試行を比較し、優先順位に基づいて最適なモデルを選択します（例: 不均衡データにはAUC）。</p>

            <p><strong>プロのヒント:</strong> 1つのメトリックが最も重要な場合は、primary_metric（例: "f1"）をカスタマイズします。</p>
        `
    },
    {
        number: 8,
        domain: "Hyperopt and SparkTrials",
        question: "HyperoptでNaNの損失が報告された場合、通常何を示し、どのように対処できますか？",
        keyPoint: "NaN損失は目的関数のエラーを意味し、ハイパーパラメータ空間の調整または目的関数の修正で対処",
        choices: [
            "NaN損失はHyperoptのバグであり、開発者に報告すべき",
            "NaN損失は成功した実行を示し、安全に無視できる",
            "NaN損失は目的関数のエラーを意味し、ハイパーパラメータ空間の調整または目的関数の修正で対処できる",
            "NaN損失はSparkTrialsの問題であり、並列性の設定で解決できる"
        ],
        correctIndex: 2,
        explanation: `
            <p>Hyperoptでは、NaN損失は通常、目的関数の問題により発生します。これは、一部の反復で関数がNaN（Not a Number）を返し、Hyperoptが損失を適切に評価できないことを意味します。</p>

            <h4>HyperoptでNaN損失の一般的な原因:</h4>
            <ul>
                <li><strong>無効なハイパーパラメータ値:</strong> 一部のハイパーパラメータの組み合わせが数学的エラーを引き起こす可能性があります（例: log(0)、ゼロ除算、数値オーバーフロー）</li>
                <li><strong>目的関数のエラー:</strong> 関数が例外に遭遇した場合、有効な損失の代わりにNaNを返す可能性があります</li>
                <li><strong>数値的不安定性:</strong> 一部のMLモデル（例: ディープラーニング）がオーバーフローまたはアンダーフローの問題を引き起こす可能性があります</li>
            </ul>

            <h4>HyperoptでNaN損失を修正する方法:</h4>

            <p><strong>ステップ1: 原因を特定するためのデバッグを追加</strong></p>
            <div class="code-block">import numpy as np
import hyperopt

def objective(params):
    try:
        # モデルをトレーニングして損失を返す
        loss = train_model(params)
        if np.isnan(loss):
            return float('inf')  # NaNの代わりに無限大を返す
        return loss
    except Exception as e:
        print(f"エラー: {e}")
        return float('inf')  # Hyperoptが検索を続行できるようにする</div>

            <p><strong>ステップ2: ハイパーパラメータ空間を調整</strong></p>
            <p>パラメータが有効な範囲を持つことを確認します。</p>

            <div class="code-block"># 修正前（不正解）:
space = {
    'learning_rate': hp.uniform('lr', 0, 1),  # 0を含む可能性がある
    'max_depth': hp.uniform('depth', 1, 50)
}

# 修正後（正解）:
space = {
    'learning_rate': hp.loguniform('lr', np.log(0.001), np.log(1)),  # 0を回避
    'max_depth': hp.quniform('depth', 1, 50, 1)
}</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>「HyperoptのバグでDEVに報告」:</strong> NaN損失は通常、ユーザー側の問題であり、Hyperoptのバグではありません</li>
                <li><strong>「成功した実行で無視可能」:</strong> NaNは適切な最適化を妨げるため、決して無視すべきではありません</li>
                <li><strong>「SparkTrialsの問題」:</strong> NaN損失はSparkTrialsの並列性とは無関係です</li>
            </ul>

            <p><strong>重要ポイント:</strong> HyperoptのNaN損失は、通常、無効なハイパーパラメータ値、数学的エラー、または数値的不安定性による目的関数のエラーを意味します。</p>

            <p>修正方法:</p>
            <ul>
                <li>ハイパーパラメータ検索空間を検証</li>
                <li>NaNを適切に処理するように目的関数を修正</li>
                <li>エラーハンドリングを使用して最適化中の失敗を防ぐ</li>
            </ul>
        `
    },
    {
        number: 9,
        domain: "Databricks ML",
        question: "PySparkでは、PythonとApache Sparkの統合を容易にする________ライブラリが提供されています。",
        keyPoint: "Py4jがPythonとJVM（Spark）の橋渡しをする",
        choices: [
            "Py3j",
            "Py5j",
            "Py2j",
            "Py4j"
        ],
        correctIndex: 3,
        explanation: `
            <p><strong>Py4j</strong>は、PythonがJava仮想マシン（JVM）と対話できるようにする公式ライブラリであり、PySparkの機能にとって重要です。</p>

            <p>これはPythonとSparkのJava/Scalaベースのコア間の橋として機能し、以下を可能にします:</p>
            <ul>
                <li>Spark操作の実行（例: RDD/DataFrame変換）</li>
                <li>JVMオブジェクトへのアクセス（例: SparkContext、SparkSession）</li>
            </ul>

            <h4>動作方法:</h4>
            <p>PySparkコードを実行すると、Py4jはPython呼び出しをJVM呼び出しに（およびその逆に）変換します。</p>

            <div class="code-block">from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()  # Py4jがJVM通信を処理</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>Py3j/Py5j/Py2j:</strong> これらは実在のライブラリではありません。Py4jはSparkでのPython-JVM統合のための唯一の有効なツールです</li>
            </ul>

            <p><strong>重要ポイント:</strong> Py4jはPySparkの縁の下の力持ちであり、シームレスなPython-to-JVMの相互運用性を可能にします。これがなければ、PySparkは存在しません！</p>

            <p><strong>面白い事実:</strong> "Py4j"という名前は「Python for Java」の略です。</p>
        `
    },
    {
        number: 10,
        domain: "Hyperopt and SparkTrials",
        question: "なぜオートスケーリングクラスタでSparkTrialsを使用すべきではないのですか？どのような問題が発生する可能性がありますか？",
        keyPoint: "Hyperoptは実行開始時に並列性を選択し、オートスケーリングがこの設定に影響を与える可能性がある",
        choices: [
            "SparkTrialsはオートスケーリングクラスタと互換性がない",
            "オートスケーリングクラスタはSparkTrialsの構成をサポートしていない",
            "Hyperoptはオートスケーリングクラスタで並列性の値を選択できない",
            "Hyperoptは実行開始時に並列性を選択し、オートスケーリングがこの構成に影響を与える可能性がある"
        ],
        correctIndex: 3,
        explanation: `
            <p>SparkTrialsは、初期クラスタサイズに基づいて起動時に並列性（同時試行数）を設定します。</p>

            <p>実行中にクラスタがオートスケールアップ/ダウンする場合:</p>
            <ul>
                <li><strong>リソース競合:</strong> 新しいワーカーが既存の試行に利用されない可能性があります</li>
                <li><strong>未活用:</strong> ワーカーが減少すると、試行がキューに入る可能性があります</li>
            </ul>

            <div class="code-block">from hyperopt import SparkTrials

spark_trials = SparkTrials(parallelism=4)  # 起動時に固定</div>

            <p>クラスタが4 → 8ノードにスケールした場合、並列性は4のままです。</p>

            <h4>結果として生じる問題:</h4>
            <ul>
                <li><strong>非効率的なスケーリング:</strong> 過剰なノードがアイドル状態になります</li>
                <li><strong>停滞した試行:</strong> ノードがスケールダウンすると、試行がハングする可能性があります</li>
            </ul>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>「SparkTrialsが互換性がない」:</strong> 機能しますが、オートスケーリングが効果を減少させます</li>
                <li><strong>「オートスケーリングがSparkTrialsをサポートしない」:</strong> オートスケーリングは機能しますが、並列性は動的ではありません</li>
                <li><strong>「Hyperoptが並列性を選択できない」:</strong> 選択しますが、起動時に1回だけです</li>
            </ul>

            <p><strong>重要ポイント:</strong> Hyperopt + SparkTrialsの場合、並列性に合わせて固定サイズのクラスタを使用します。オートスケーリングは試行の分散を妨げます。</p>

            <p><strong>プロのヒント:</strong> オートスケーリングクラスタの場合、ベースラインリソースに合わせてSparkTrials(parallelism=min_workers)を使用します。</p>
        `
    },
    {
        number: 11,
        domain: "Databricks ML",
        question: "マルチタスクMLワークフローを管理するために現代のデータサイエンスで価値があるスキルは何ですか？",
        keyPoint: "ワークフローオーケストレーションがマルチタスクMLパイプライン管理に不可欠",
        choices: [
            "効率的なクラスタ作成",
            "ワークフローオーケストレーション",
            "パフォーマンス最適化",
            "特徴エンジニアリング"
        ],
        correctIndex: 1,
        explanation: `
            <p>ワークフローオーケストレーション（例: Airflow、Databricks Workflows、Kubeflow）は、以下を含むマルチタスクMLパイプラインの管理に不可欠です:</p>

            <ul>
                <li><strong>依存関係:</strong> タスク（データ準備 → トレーニング → デプロイ）が順番に実行されることを保証</li>
                <li><strong>スケジューリング:</strong> 定期実行の自動化（例: 夜間のモデル再トレーニング）</li>
                <li><strong>エラーハンドリング:</strong> 再試行、アラート、堅牢性のためのログ記録</li>
            </ul>

            <div class="code-block"># Databricks Workflowsの例
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
w.jobs.submit(run_name="retrain_model", tasks=[data_task, train_task, deploy_task])</div>

            <h4>マルチタスクMLへの影響:</h4>
            <ul>
                <li><strong>再現性:</strong> タスク間の入出力を追跡</li>
                <li><strong>スケーラビリティ:</strong> クラスタ全体でタスクを並列化</li>
            </ul>

            <h4>他の選択肢がそれほど重要ではない理由:</h4>
            <ul>
                <li><strong>効率的なクラスタ作成:</strong> 重要ですが、インフラチーム/オーケストレータによって処理されます</li>
                <li><strong>パフォーマンス最適化:</strong> タスク固有（例: モデルチューニング）であり、パイプライン全体ではありません</li>
                <li><strong>特徴エンジニアリング:</strong> ワークフロー内の単一タスクです</li>
            </ul>

            <p><strong>重要ポイント:</strong> オーケストレーションツール（例: Airflow、MLflow Pipelines、Prefect）をマスターして、回復力があり自動化されたMLワークフローを設計します。</p>

            <p><strong>プロのヒント:</strong> 実験追跡とのシームレスなオーケストレーションのために、MLflow Projects + Databricks Jobsを使用します。</p>
        `
    },
    {
        number: 12,
        domain: "ML Workflows",
        question: "True or False? ビニング（Binning）は数値データをカテゴリカルデータに変換するプロセスです。",
        keyPoint: "ビニングは連続値をカテゴリカルなbin/範囲に変換",
        choices: [
            "True",
            "False"
        ],
        correctIndex: 0,
        explanation: `
            <p>ビニング（または離散化）は、連続的な数値をカテゴリカルなbin/範囲に変換します。</p>

            <p><strong>例:</strong> 年齢を[0-18, 19-35, 36-60, 61+]のようなグループに変換。</p>

            <h4>一般的な方法:</h4>
            <ul>
                <li><strong>等幅ビニング:</strong> 固定範囲（例: 0-10、10-20）</li>
                <li><strong>等頻度ビニング:</strong> 各binにほぼ同数のサンプル</li>
                <li><strong>分位数ビニング:</strong> パーセンタイルカットを使用</li>
            </ul>

            <div class="code-block">import pandas as pd

df["age_bin"] = pd.cut(df["age"], bins=[0, 18, 35, 60, 100],
                        labels=["0-18", "19-35", "36-60", "61+"])</div>

            <h4>なぜ"False"が不正解か:</h4>
            <p>ビニングは明示的に数値 → カテゴリカルに変換します（例: 決定木用やノイズ削減のため）。</p>

            <p><strong>重要ポイント:</strong> ビニングを使用して以下を実現:</p>
            <ul>
                <li>モデルの簡素化</li>
                <li>外れ値の処理</li>
                <li>非線形関係のキャプチャ</li>
            </ul>

            <p><strong>プロのヒント:</strong> MLの場合、ビニングの順序性が意味を持たない場合はワンホットエンコーディングと組み合わせます。</p>
        `
    },
    {
        number: 13,
        domain: "ML Workflows",
        question: "データサイエンティストは、log(price)を目的変数として利用する線形回帰モデルを開発しました。このモデルを使用して予測を実行し、結果と実際のラベル値はpreds_dfという名前のSpark DataFrameに保持されています。彼らは次のコードブロックを適用してモデルを評価します:\nregression_evaluator.setMetricName(\"rmse\").evaluate(preds_df)\n元のprice尺度と比較可能にするために、RMSE評価アプローチにどのような調整を行うべきですか？",
        keyPoint: "log変換された目的変数の場合、RMSE計算前に予測値を指数化",
        choices: [
            "RMSEを計算する前に予測値に対数を適用する",
            "log変換された予測値のMSEを計算してRMSEを取得する",
            "RMSEを計算する前に予測値に指数関数を適用する",
            "計算されたRMSE値の指数を取る",
            "導出されたRMSE値の対数を計算する"
        ],
        correctIndex: 2,
        explanation: `
            <p>モデルはlog(price)を予測するため、予測値(preds_df)は対数スケールです。</p>

            <p>元のprice尺度でRMSEを評価するには、以下を行う必要があります:</p>
            <ul>
                <li><strong>予測値を指数化:</strong> exp()を使用してlog(price)をpriceに戻す</li>
                <li><strong>RMSE計算:</strong> 指数化された予測値と実際のpriceの間でRMSEを計算</li>
            </ul>

            <div class="code-block">from pyspark.sql.functions import exp
from pyspark.ml.evaluation import RegressionEvaluator

# 予測値を元のスケールに変換
preds_df = preds_df.withColumn("price_pred", exp("prediction"))

# 元のスケールでRMSEを計算
evaluator = RegressionEvaluator(
    predictionCol="price_pred",
    labelCol="price",
    metricName="rmse"
)
rmse = evaluator.evaluate(preds_df)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>「予測値に対数を適用」:</strong> 不正解—これはRMSEを対数スケールに保ちます</li>
                <li><strong>「log変換された予測値のMSEを計算」:</strong> 依然として対数スケールのエラーを評価し、元のpriceではありません</li>
                <li><strong>「計算されたRMSEの指数」:</strong> 間違い—RMSEではなく、最初に予測値を指数化します</li>
                <li><strong>「RMSEの対数」:</strong> 無意味—log(RMSE)はメトリックを歪めます</li>
            </ul>

            <p><strong>重要ポイント:</strong> log変換された目的変数を使用する場合、元の単位でエラーを比較するために、常にRMSEを計算する前に予測値を指数化します。</p>

            <p><strong>プロのヒント:</strong> 非対称なエラーペナルティの場合、代わりにMean Absolute Percentage Error (MAPE)を検討してください。</p>
        `
    },
    {
        number: 14,
        domain: "Spark ML Algorithms",
        question: "Spark MLアルゴリズムでL1およびL2正則化のような正則化技術をいつ適用すべきですか？",
        keyPoint: "モデルが過学習している場合に正則化を適用",
        choices: [
            "カテゴリカルデータを扱う場合",
            "モデルが過学習している場合",
            "データセットサイズが小さい場合",
            "データセットサイズが大きい場合"
        ],
        correctIndex: 1,
        explanation: `
            <p>正則化（L1/L2）は主に以下によって過学習を防ぐために使用されます:</p>

            <ul>
                <li><strong>L1 (Lasso):</strong> 絶対係数に比例したペナルティを追加（特徴選択のために一部をゼロに縮小可能）</li>
                <li><strong>L2 (Ridge):</strong> 二乗係数に比例したペナルティを追加（重みを滑らかにする）</li>
            </ul>

            <p>Spark MLは以下のようなアルゴリズムに正則化を統合しています:</p>
            <ul>
                <li>LinearRegression（elasticNetParamがL1/L2をブレンド）</li>
                <li>LogisticRegression（regParamがペナルティ強度を制御）</li>
            </ul>

            <div class="code-block">from pyspark.ml.regression import LinearRegression

# Ridge回帰 (L2)
lr = LinearRegression(regParam=0.1, elasticNetParam=0)  # 純粋なL2
model = lr.fit(train_df)</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>「カテゴリカルデータ」:</strong> 正則化は数値係数用であり、カテゴリカルエンコーディングではありません</li>
                <li><strong>「小さいデータセット」:</strong> 過学習リスクは存在しますが、正則化は小さいデータに限定されません</li>
                <li><strong>「大きいデータセット」:</strong> 過学習しにくいですが、正則化はノイズの多い特徴に対して依然として役立ちます</li>
            </ul>

            <p><strong>重要ポイント:</strong> Spark MLでL1/L2を使用する場合:</p>
            <ul>
                <li>高分散（過学習）が検出された場合（例: 優れたtrain精度だが不十分なtest精度）</li>
                <li>特徴選択が必要な場合（L1）</li>
            </ul>

            <p><strong>プロのヒント:</strong> 最適な正則化強度のためにSpark MLのCrossValidatorを介してregParamを調整します。</p>
        `
    },
    {
        number: 15,
        domain: "Pandas API on Spark",
        question: "Spark DataFrameからpandas-on-Spark DataFrameを作成する際、デフォルトインデックスに関してどのような注意を考慮すべきですか？",
        keyPoint: "pandas-on-Spark DataFrame作成時に新しいデフォルトインデックスが作成される",
        choices: [
            "デフォルトインデックスは変更されません",
            "新しいデフォルトインデックスが作成されます",
            "データセットのサイズによって異なります",
            "デフォルトインデックスは'index_col'に設定されます"
        ],
        correctIndex: 1,
        explanation: `
            <p>Spark DataFrameをpandas-on-Spark DataFrameに変換する際、明示的に指定しない限り、ライブラリは自動的に新しいデフォルトインデックス（連番の整数）を生成します。</p>

            <h4>理由:</h4>
            <ul>
                <li>Spark DataFrameは分散されており、本質的に行インデックスを持ちません</li>
                <li>pandas-on-Sparkはpandasの動作を模倣し、インデックスが基本です</li>
            </ul>

            <div class="code-block">import pyspark.pandas as ps

spark_df = spark.createDataFrame([(1, "A"), (2, "B")], ["id", "value"])
ps_df = ps.DataFrame(spark_df)  # 新しいデフォルトインデックス (0, 1, ...) が作成される</div>

            <h4>主な影響:</h4>
            <ul>
                <li><strong>パフォーマンスオーバーヘッド:</strong> インデックス作成には一意性を確保するためにデータのシャッフルが必要です</li>
                <li><strong>データ整合性:</strong> 新しいインデックスは元のSpark行の順序を保持しません</li>
            </ul>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>「変更されません」:</strong> Spark DataFrameにはデフォルトインデックスがないため、保持するものはありません</li>
                <li><strong>「データセットサイズによって異なります」:</strong> インデックス作成は一貫しています（常に発生します）</li>
                <li><strong>「'index_col'に設定」:</strong> 変換中に明示的にindex_colを設定した場合のみ発生します</li>
            </ul>

            <p><strong>重要ポイント:</strong> 驚きを避けるために:</p>
            <ul>
                <li>必要に応じてインデックスを明示的に設定（例: ps.DataFrame(spark_df, index="id")）</li>
                <li>明確にするためにspark_df.to_pandas_on_spark()を使用</li>
            </ul>

            <p><strong>プロのヒント:</strong> 大きなDataFrameの場合、デフォルトインデックスを避け、シャッフルを最小限に抑えるために既存の列をインデックスとして使用します。</p>
        `
    },
    {
        number: 16,
        domain: "Feature Store",
        question: "提供された例で言及されているように、統合テストにクライアントを使用する目的は何ですか？",
        keyPoint: "統合テストはFeatureEngineeringClientまたはFeatureStoreClientのwrite_table等の関数呼び出しを検証",
        choices: [
            "Databricksで単体テストを実行する",
            "FeatureEngineeringClientまたはFeatureStoreClientのwrite_tableなどの関数をメソッドが正しく呼び出すことを検証する",
            "開発用にFeature Engineering in Unity CatalogクライアントまたはFeature Storeクライアントをローカルにインストールする",
            "クライアントのCI/CDを有効にする"
        ],
        correctIndex: 1,
        explanation: `
            <p>統合テストは、コードと外部クライアント（例: FeatureEngineeringClient）間の相互作用が期待通りに機能することを保証します。</p>

            <h4>検証内容:</h4>
            <ul>
                <li><strong>正しいAPI呼び出し:</strong> 例: write_tableは正しいパラメータでクライアントのメソッドを実際に呼び出すか？</li>
                <li><strong>エンドツーエンドワークフロー:</strong> コードと統合されたときにクライアントは期待通りに応答するか？</li>
            </ul>

            <div class="code-block">def test_write_table_integration():
    mock_client = Mock(FeatureStoreClient)
    your_method(mock_client)  # mock_client.write_table(...)を呼び出す
    mock_client.write_table.assert_called_once()  # 統合を検証</div>

            <h4>主な利点:</h4>
            <p>本番環境に入る前に、誤設定されたパラメータやクライアントの誤用のような問題を検出します。</p>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>「Databricksで単体テストを実行」:</strong> 単体テストはコンポーネントを分離します。統合テストは相互作用を検証します</li>
                <li><strong>「クライアントをローカルにインストール」:</strong> 開発セットアップであり、テストではありません</li>
                <li><strong>「CI/CDを有効化」:</strong> CI/CDパイプラインはテストを使用しますが、統合テストの目的ではありません</li>
            </ul>

            <p><strong>重要ポイント:</strong> 統合テストを使用してクライアントの相互作用（例: Feature Store書き込み）を検証し、実行時の失敗を防ぎます。</p>

            <p><strong>プロのヒント:</strong> 再現可能なテストのためにpytestフィクスチャと組み合わせてクライアントをモックします。</p>
        `
    },
    {
        number: 17,
        domain: "ML Workflows",
        question: "データサイエンティストは、Spark MLを利用して3クラス決定木分類器を設計し、次のスキーマを持つpreds_dtという名前のSpark DataFrameで予測を計算しました:\nprediction DOUBLE, actual DOUBLE.\npreds_dtのデータに基づいてモデルの精度を計算し、結果をaccuracy変数に割り当てるために使用できるコードセグメントはどれですか？",
        keyPoint: "Spark MLの多クラス分類精度計算にはMulticlassClassificationEvaluatorを使用",
        choices: [
            "None",
            "accuracy = MulticlassClassificationEvaluator(predictionCol=\"prediction\", labelCol=\"actual\", metricName=\"accuracy\")",
            "accuracy = RegressionEvaluator(predictionCol=\"prediction\", labelCol=\"actual\", metricName=\"accuracy\")",
            "classification_evaluator = MulticlassClassificationEvaluator(predictionCol=\"prediction\", labelCol=\"actual\", metricName=\"accuracy\")\naccuracy = classification_evaluator.evaluate(preds_df)",
            "accuracy = Summarizer(predictionCol=\"prediction\", labelCol=\"actual\", metricName=\"accuracy\")"
        ],
        correctIndex: 3,
        explanation: `
            <p>Spark MLの多クラス分類器の場合、MulticlassClassificationEvaluatorは精度のようなメトリックを計算する正しいツールです。</p>

            <h4>主要なステップ:</h4>
            <ol>
                <li>評価器を以下で初期化:
                    <ul>
                        <li>predictionCol="prediction"（モデル出力）</li>
                        <li>labelCol="actual"（真のラベル）</li>
                        <li>metricName="accuracy"</li>
                    </ul>
                </li>
                <li>DataFrame(preds_df)でevaluate()を呼び出す</li>
            </ol>

            <div class="code-block">from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(
    predictionCol="prediction",
    labelCol="actual",
    metricName="accuracy"
)
accuracy = evaluator.evaluate(preds_df)  # 浮動小数点数として精度を返す</div>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>RegressionEvaluator:</strong> 回帰タスク用であり、分類用ではありません</li>
                <li><strong>Summarizer:</strong> Spark MLには存在しません</li>
                <li><strong>evaluate()呼び出しなし:</strong> 評価器を初期化するだけではメトリックを計算しません</li>
            </ul>

            <p><strong>重要ポイント:</strong> Spark MLの多クラス精度の場合:</p>
            <ul>
                <li>MulticlassClassificationEvaluatorを使用</li>
                <li>列名がDataFrameスキーマと一致することを確認</li>
            </ul>

            <p><strong>プロのヒント:</strong> 他のサポートされているメトリックには、f1、weightedPrecision、weightedRecallが含まれます。</p>
        `
    },
    {
        number: 18,
        domain: "Feature Store",
        question: "Unity Catalogの既存の特徴テーブルを新しいデータで更新するにはどうすればよいですか？",
        keyPoint: "fe.write_table関数をmode=\"merge\"で使用して特徴テーブルを更新",
        choices: [
            "新しいdataframeでfe.create_table関数を使用する",
            "mode=\"merge\"と新しいdataframeを提供してfe.write_table関数を使用する",
            "fe.read_table関数を使用してdataframeを更新する",
            "新しいdataframeでfe.update_table関数を使用する"
        ],
        correctIndex: 1,
        explanation: `
            <p>Databricks Feature Engineering (Unity Catalog)のfe.write_table関数はmode="merge"をサポートしており、これは:</p>

            <ul>
                <li>既存のレコードを更新（プライマリキーで一致）</li>
                <li>新しいレコードを挿入（存在しない場合）</li>
            </ul>

            <p>これは特徴テーブルの増分更新の標準的な方法です。</p>

            <div class="code-block">from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# 新しいデータを既存のテーブルにマージ
fe.write_table(
    name="catalog.schema.feature_table",
    df=new_data_df,
    mode="merge"  # キーパラメータ
)</div>

            <p>プライマリキー（テーブル作成時に定義）は、更新/挿入する行を決定します。</p>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>fe.create_table:</strong> 新しいテーブルを作成します（テーブルが既に存在する場合は失敗します）</li>
                <li><strong>fe.read_table + 手動更新:</strong> 元のテーブルへの変更を永続化しません</li>
                <li><strong>fe.update_table:</strong> APIにそのような関数は存在しません</li>
            </ul>

            <p><strong>重要ポイント:</strong> Unity Catalog特徴テーブルの増分更新の場合:</p>
            <ul>
                <li>mode="merge"でfe.write_tableを使用</li>
                <li>新しいDataFrameに一致のためのプライマリキー列が含まれていることを確認</li>
            </ul>

            <p><strong>プロのヒント:</strong> 初期セットアップにはfe.create_tableと組み合わせ、バージョン管理にはタイムトラベルを使用します。</p>
        `
    },
    {
        number: 19,
        domain: "Scaling ML Models",
        question: "AutoML実験で、回帰問題を扱う際に各実行で自動的に計算される評価メトリックは何ですか？",
        keyPoint: "AutoMLは回帰でMAE、R二乗、RMSE、MSEをすべて自動計算",
        choices: [
            "Mean Absolute Error (MAE)",
            "Coefficient of Determination (R-squared)",
            "Root Mean Square Error (RMSE)",
            "Mean Square Error (MSE)",
            "上記のすべて"
        ],
        correctIndex: 4,
        explanation: `
            <p>Databricks AutoMLは、回帰問題に対して以下を含む複数の評価メトリックを自動的に計算します:</p>

            <ul>
                <li><strong>Mean Absolute Error (MAE)</strong></li>
                <li><strong>Coefficient of Determination (R-squared, R²)</strong></li>
                <li><strong>Root Mean Square Error (RMSE)</strong></li>
                <li><strong>Mean Square Error (MSE)</strong></li>
            </ul>

            <h4>各回帰メトリックの内訳:</h4>
            <table border="1">
                <tr>
                    <th>メトリック</th>
                    <th>説明</th>
                    <th>最適な使用ケース</th>
                </tr>
                <tr>
                    <td>MAE</td>
                    <td>実際の値と予測値の平均絶対差を測定</td>
                    <td>すべての予測エラーを等しく重み付けする場合に有用</td>
                </tr>
                <tr>
                    <td>R²</td>
                    <td>モデルによって説明される分散の割合を表す</td>
                    <td>適合度を測定—高い値はより良い適合を意味する</td>
                </tr>
                <tr>
                    <td>RMSE</td>
                    <td>平均二乗誤差の平方根を測定</td>
                    <td>MAEよりも大きなエラーにペナルティを与える</td>
                </tr>
                <tr>
                    <td>MSE</td>
                    <td>実際の値と予測値の平均二乗誤差を測定</td>
                    <td>大きなエラーをより大幅にペナルティする場合に使用</td>
                </tr>
            </table>

            <div class="code-block">from pyspark.ml.evaluation import RegressionEvaluator

# 評価器を初期化
evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="actual")

# 異なるメトリックを計算
mae = evaluator.setMetricName("mae").evaluate(preds_df)
r2 = evaluator.setMetricName("r2").evaluate(preds_df)
rmse = evaluator.setMetricName("rmse").evaluate(preds_df)
mse = evaluator.setMetricName("mse").evaluate(preds_df)

print(f"MAE: {mae}, R-squared: {r2}, RMSE: {rmse}, MSE: {mse}")</div>

            <p>Databricks AutoMLはすべてのモデル実行に対してこれらのメトリックをログに記録し、簡単な比較を可能にします。</p>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>「Mean Absolute Error (MAE)」:</strong> MAEは回帰で使用されるいくつかの評価メトリックの1つにすぎません</li>
                <li><strong>「Coefficient of Determination (R-squared)」:</strong> AutoMLはMSE、RMSE、MAEも計算します</li>
                <li><strong>「Root Mean Square Error (RMSE)」:</strong> RMSEと共に他の回帰メトリックも計算されます</li>
                <li><strong>「Mean Square Error (MSE)」:</strong> AutoMLはMAEやR²などの追加メトリックを追跡します</li>
            </ul>

            <p><strong>最終結論:</strong> AutoML回帰実験では、自動的に計算される評価メトリックにMAE、R²、RMSE、MSEが含まれるため、正解は「上記のすべて」です。</p>
        `
    },
    {
        number: 20,
        domain: "ML Workflows",
        question: "データサイエンティストは、機密データを含むDatabricksの機械学習プロジェクトに取り組んでいます。プロジェクトには、データへのアクセスが制限されるべき外部の関係者との協力が必要です。データサイエンティストは、外部の協力者とDatabricksノートブックを安全に共有するにはどうすればよいですか？",
        keyPoint: "Workspace ACLsを使用して特定のユーザーへのアクセスを制限",
        choices: [
            "Databricks Workspace ACLs（アクセス制御リスト）を使用して特定のユーザーへのアクセスを制限する",
            "ノートブックをエクスポートして外部の協力者に電子メールで共有する",
            "外部の協力者がノートブックにアクセスするための共有秘密鍵を作成する",
            "Databricks Managed Identityを利用して外部の協力者に一時的なアクセスを許可する"
        ],
        correctIndex: 0,
        explanation: `
            <p>DatabricksのWorkspace ACLsは、ノートブックアクセスに対するきめ細かい制御を可能にします:</p>

            <ul>
                <li>特定の外部ユーザー（電子メールまたはグループ経由）に読み取り専用または編集権限を付与</li>
                <li>ノートブック、フォルダ、またはワークスペースレベルで権限を制限することで機密データへのアクセスを制限</li>
            </ul>

            <h4>手順例:</h4>
            <ol>
                <li>ワークスペース → ノートブックを右クリック → 権限に移動</li>
                <li>外部の協力者の電子メールを追加し、適切なアクセスレベルを設定（例: "表示可能"）</li>
            </ol>

            <h4>セキュリティ上の利点:</h4>
            <ul>
                <li>データ/ノートブックをエクスポートする必要がありません（制御されていないコピーを回避）</li>
                <li>外部ユーザーはDatabricks UI経由で安全にノートブックにアクセス（ローカルストレージなし）</li>
            </ul>

            <h4>他の選択肢が不正解な理由:</h4>
            <ul>
                <li><strong>「エクスポートして電子メールで共有」:</strong> 安全でない：エクスポート後、データに対する制御を失います</li>
                <li><strong>「共有秘密鍵」:</strong> ネイティブのDatabricks機能ではありません。鍵を安全に管理するのは困難です</li>
                <li><strong>「Managed Identity」:</strong> Azure AD認証に使用され、外部ノートブック共有には使用されません</li>
            </ul>

            <p><strong>重要ポイント:</strong> 安全なコラボレーションのために:</p>
            <ul>
                <li>Workspace ACLsを使用して、最小権限アクセスで外部ユーザーを招待します</li>
                <li>非常に機密性の高いデータの場合、Delta Lake ACLsまたはデータマスキングと組み合わせます</li>
            </ul>

            <p><strong>プロのヒント:</strong> 一時的なアクセスの場合、権限に有効期限を設定します（Enterprise限定機能）。</p>
        `
    },
    {
        number: 21,
        domain: "Databricks ML",
        question: "機械学習プロジェクトでDatabricksクラスターを設定する際、高度な設定を含むより多くの制御が必要です。どのDatabricksクラスター設定モードが推奨されますか？",
        keyPoint: "高度な設定が必要な場合のクラスター設定モード",
        choices: [
            "シンプルモード",
            "カスタムモード",
            "アドバンストモード",
            "ベーシックモード"
        ],
        correctIndex: 1,
        explanation: `
            <h3>正解: カスタムモード</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>Databricksクラスターの<strong>カスタムモード</strong>は、以下を可能にします：</p>
            <ul>
                <li><strong>高度なSpark設定:</strong> メモリ、シャッフル、並列処理のチューニング</li>
                <li><strong>Initスクリプト:</strong> クラスター起動時にカスタムスクリプトを実行（依存関係のインストールなど）</li>
                <li><strong>環境変数:</strong> ジョブやノートブック全体で使用するためのカスタム変数を設定</li>
                <li><strong>タグとアクセス制御:</strong> 詳細なリソース管理</li>
            </ul>

            <h5>例：カスタムモードでのクラスター設定</h5>
            <pre><code class="language-json">
{
  "spark_conf": {
    "spark.executor.memory": "16g",
    "spark.sql.shuffle.partitions": "200"
  },
  "init_scripts": [{
    "dbfs": {
      "destination": "dbfs:/databricks/init/install_libs.sh"
    }
  }],
  "spark_env_vars": {
    "MLFLOW_TRACKING_URI": "databricks"
  }
}
            </code></pre>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>シンプルモード:</strong> 基本的なクラスター作成のみ（高度な設定なし）</li>
                <li><strong>アドバンストモード:</strong> Databricksに存在しない用語</li>
                <li><strong>ベーシックモード:</strong> 同様に、実際のDatabricks設定オプションではない</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>本番環境のMLワークロードの場合：</p>
            <ul>
                <li><strong>カスタムモード</strong>を使用してパフォーマンスをチューニング</li>
                <li>Initスクリプトでライブラリの一貫性を確保</li>
                <li>リソース追跡のためにタグを使用</li>
            </ul>

            <p><strong>プロのヒント:</strong> Databricks CLIまたはREST APIを使用して、クラスター設定をバージョン管理します。</p>
        `
    },
    {
        number: 22,
        domain: "Databricks ML",
        question: "データサイエンティストが、大規模なデータセットを含む機械学習プロジェクトでDatabricksを使用しています。データの読み込みと処理のパフォーマンスを最適化したいと考えています。どのDelta Lakeの機能がこの目的に役立ちますか？",
        keyPoint: "Delta Lakeでのデータ読み込みと処理のパフォーマンス最適化",
        choices: [
            "Deltaキャッシング",
            "Delta統計",
            "Z-オーダリング（マルチディメンション・クラスタリング）",
            "以上のすべて"
        ],
        correctIndex: 3,
        explanation: `
            <h3>正解: 以上のすべて</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>Delta Lakeの3つの機能すべてがパフォーマンスを向上させます：</p>

            <h5>1. Deltaキャッシング</h5>
            <p>頻繁にアクセスするデータをSSDにキャッシュして、クエリ速度を向上させます。</p>
            <pre><code class="language-python">
# 自動的に有効（手動設定不要）
df = spark.read.format("delta").load("/path/to/delta")
df.cache()  # オプション：メモリに明示的にキャッシュ
            </code></pre>

            <h5>2. Delta統計</h5>
            <p>ファイルレベルのmin/max統計により、データスキッピングが可能になります（不要なファイルを読み飛ばす）。</p>
            <pre><code class="language-python">
# 自動的に収集される統計情報
df.filter("date >= '2023-01-01'")  # 統計情報により古いファイルをスキップ
            </code></pre>

            <h5>3. Z-オーダリング（マルチディメンション・クラスタリング）</h5>
            <p>頻繁にフィルタリングされるカラムに基づいてデータをクラスター化します。</p>
            <pre><code class="language-python">
# Z-オーダリングを実行
spark.sql("OPTIMIZE delta.\`/path/to/delta\` ZORDER BY (user_id, date)")
            </code></pre>

            <h4>組み合わせたインパクト</h4>
            <table border="1" cellpadding="5">
                <tr>
                    <th>機能</th>
                    <th>メリット</th>
                </tr>
                <tr>
                    <td>Deltaキャッシング</td>
                    <td>高速なリピートクエリ</td>
                </tr>
                <tr>
                    <td>Delta統計</td>
                    <td>スキャンするデータを削減</td>
                </tr>
                <tr>
                    <td>Z-オーダリング</td>
                    <td>マルチカラムフィルターを最適化</td>
                </tr>
            </table>

            <h4>重要ポイント</h4>
            <p>大規模なMLデータセットの場合：</p>
            <ul>
                <li>Z-オーダリングをフィルターされる頻繁なカラムで使用</li>
                <li>定期的に<code>OPTIMIZE</code>コマンドを実行して小さなファイルをコンパクト化</li>
                <li>Deltaキャッシングがデフォルトで有効であることを信頼</li>
            </ul>

            <p><strong>プロのヒント:</strong> <code>DESCRIBE DETAIL</code>を使用して、ファイル数とサイズを確認します。多くの小さなファイル = OPTIMIZEが必要です。</p>
        `
    },
    {
        number: 23,
        domain: "Databricks ML",
        question: "データサイエンティストが、大規模なデータセットを効率的に処理するために、Databricksで分散機械学習モデルのトレーニングに取り組んでいます。Databricks MLlibのどの機能が、複数のノードでのモデルのトレーニングをサポートしていますか？",
        keyPoint: "MLlibでの分散モデルトレーニング",
        choices: [
            "pandas UDF",
            "Horovod",
            "MLlibのPipeline API",
            "Spark DataFrames"
        ],
        correctIndex: 1,
        explanation: `
            <h3>正解: Horovod</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p><strong>Horovod</strong>は、Databricksでディープラーニングモデル（TensorFlow、PyTorch、Keras）の分散トレーニングのために構築されたフレームワークです。</p>

            <h5>主な機能</h5>
            <ul>
                <li><strong>データ並列化:</strong> トレーニングデータを複数のワーカーに分割</li>
                <li><strong>勾配の集約:</strong> Ring-AllReduceを使用して効率的にパラメータを同期</li>
                <li><strong>Sparkとの統合:</strong> <code>HorovodRunner</code>を介してDatabricksクラスターでシームレスに動作</li>
            </ul>

            <h5>例：HorovodによるPyTorchの分散トレーニング</h5>
            <pre><code class="language-python">
from sparkdl import HorovodRunner
import torch

def train_model(learning_rate):
    # PyTorchのトレーニングコード
    model = torch.nn.Linear(10, 1)
    # ... トレーニングループ ...
    return model

# 4ノードでトレーニング実行
hr = HorovodRunner(np=4)  # np = プロセス数（ワーカー数）
hr.run(train_model, learning_rate=0.01)
            </code></pre>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>pandas UDF:</strong> カスタムの分散関数用ですが、モデルトレーニング専用ではありません</li>
                <li><strong>MLlib Pipeline API:</strong> ワークフローオーケストレーション用で、分散DL用ではありません</li>
                <li><strong>Spark DataFrames:</strong> データ処理用で、モデルトレーニング用ではありません</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>大規模なDLモデルの場合：</p>
            <ul>
                <li>Horovodを使用してGPUクラスターでトレーニングを分散</li>
                <li>MLlibアルゴリズム（RandomForest、GBT）は既に分散化されている（Horovod不要）</li>
                <li>小規模なモデルの場合、Single-Node MLを検討</li>
            </ul>

            <p><strong>プロのヒント:</strong> HorovodはPetastormと組み合わせて、Deltaテーブルから直接効率的にデータを読み込みます。</p>
        `
    },
    {
        number: 24,
        domain: "Databricks ML",
        question: "データサイエンティストがDatabricksノートブックで作業しており、機械学習モデルの実験と結果を追跡したいと考えています。どのDatabricksのツールまたは機能が実験追跡に最も適していますか？",
        keyPoint: "Databricksでの実験追跡",
        choices: [
            "Databricks Delta",
            "MLflow Tracking",
            "Databricks Repos",
            "Databricks Workflows"
        ],
        correctIndex: 1,
        explanation: `
            <h3>正解: MLflow Tracking</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p><strong>MLflow Tracking</strong>はDatabricksに統合されており、実験のログ記録と比較のために設計されています。</p>

            <h5>主な機能</h5>
            <ul>
                <li><strong>自動ロギング:</strong> パラメータ、メトリクス、モデル、アーティファクトを追跡</li>
                <li><strong>実験UI:</strong> 複数の実行を視覚的に比較</li>
                <li><strong>バージョニング:</strong> 各トレーニング実行を一意のRun IDで保存</li>
            </ul>

            <h5>例：MLflowでの実験追跡</h5>
            <pre><code class="language-python">
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

mlflow.autolog()  # 自動パラメータ/メトリクスのロギング

with mlflow.start_run(run_name="RF_Experiment"):
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)

    # カスタムメトリクスをログ
    mlflow.log_metric("accuracy", model.score(X_test, y_test))
    mlflow.sklearn.log_model(model, "model")
            </code></pre>

            <h4>Databricks UIでの表示</h4>
            <p>ノートブック → 「実験」タブ → すべての実行のパラメータ/メトリクスを表示</p>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>Databricks Delta:</strong> データストレージ用で、実験追跡用ではない</li>
                <li><strong>Databricks Repos:</strong> Gitバージョン管理用で、MLメトリクス追跡用ではない</li>
                <li><strong>Databricks Workflows:</strong> ジョブオーケストレーション用で、実験ログ記録用ではない</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>ML実験の場合：</p>
            <ul>
                <li>常に<code>mlflow.autolog()</code>を使用してボイラープレートを削減</li>
                <li>チームコラボレーションのためにMLflow Experimentsを共有</li>
                <li>ノートブックウィジェットと組み合わせてパラメータ化された実行</li>
            </ul>

            <p><strong>プロのヒント:</strong> <code>mlflow.search_runs()</code>を使用してプログラムで最良の実行を検索します：</p>
            <pre><code class="language-python">
best_run = mlflow.search_runs(order_by=["metrics.accuracy DESC"]).iloc[0]
            </code></pre>
        `
    },
    {
        number: 25,
        domain: "Hyperopt and SparkTrials",
        question: "データサイエンティストがHyperoptを使用して、機械学習モデルのハイパーパラメータを最適化しています。Hyperoptでハイパーパラメータの探索空間を定義するにはどうすればよいですか？",
        keyPoint: "Hyperoptでのハイパーパラメータ探索空間の定義",
        choices: [
            "パラメータのリストを作成する",
            "hp.choice()やhp.uniform()などの関数を使用する",
            "手動でパラメータを調整する",
            "Hyperoptは探索空間を自動的に決定する"
        ],
        correctIndex: 1,
        explanation: `
            <h3>正解: hp.choice()やhp.uniform()などの関数を使用する</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>Hyperoptは、探索空間を定義するための特定の関数を提供します：</p>

            <h5>主な探索空間関数</h5>
            <table border="1" cellpadding="5">
                <tr>
                    <th>関数</th>
                    <th>用途</th>
                    <th>例</th>
                </tr>
                <tr>
                    <td><code>hp.choice()</code></td>
                    <td>カテゴリカル値</td>
                    <td><code>hp.choice('kernel', ['rbf', 'linear'])</code></td>
                </tr>
                <tr>
                    <td><code>hp.uniform()</code></td>
                    <td>一様分布（連続値）</td>
                    <td><code>hp.uniform('lr', 0.001, 0.1)</code></td>
                </tr>
                <tr>
                    <td><code>hp.loguniform()</code></td>
                    <td>対数スケール（学習率など）</td>
                    <td><code>hp.loguniform('alpha', -5, 0)</code></td>
                </tr>
                <tr>
                    <td><code>hp.quniform()</code></td>
                    <td>量子化された値（整数など）</td>
                    <td><code>hp.quniform('n_estimators', 50, 200, 10)</code></td>
                </tr>
            </table>

            <h5>例：探索空間の定義</h5>
            <pre><code class="language-python">
from hyperopt import hp, fmin, tpe, Trials

search_space = {
    'n_estimators': hp.quniform('n_estimators', 50, 200, 10),
    'max_depth': hp.choice('max_depth', [5, 10, 15, 20]),
    'learning_rate': hp.loguniform('learning_rate', -5, 0)
}

def objective(params):
    model = XGBClassifier(**params)
    score = cross_val_score(model, X, y, cv=3).mean()
    return -score  # Hyperoptは最小化

best = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=50)
            </code></pre>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>パラメータのリスト:</strong> グリッドサーチの方法で、Hyperoptの構文ではない</li>
                <li><strong>手動調整:</strong> Hyperoptの目的を無効にする（自動化が目的）</li>
                <li><strong>自動決定:</strong> ユーザーが明示的に空間を定義する必要がある</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>Hyperoptの最適化の場合：</p>
            <ul>
                <li>学習率には<code>hp.loguniform()</code>を使用（対数スケールの方が効率的）</li>
                <li>カテゴリカルには<code>hp.choice()</code>を使用</li>
                <li>整数には<code>hp.quniform()</code>を使用（ステップサイズを指定）</li>
            </ul>

            <p><strong>プロのヒント:</strong> SparkTrialsと組み合わせて並列評価を行います：</p>
            <pre><code class="language-python">
from hyperopt import SparkTrials
spark_trials = SparkTrials(parallelism=4)
best = fmin(fn=objective, space=search_space, trials=spark_trials, max_evals=50)
            </code></pre>
        `
    },
    {
        number: 26,
        domain: "Hyperopt and SparkTrials",
        question: "大規模なデータセットに対してHyperoptのハイパーパラメータチューニングを実行する場合、プロセスを高速化するために分散コンピューティングを活用したいと考えています。どのHyperoptの機能がこれに役立ちますか？",
        keyPoint: "Hyperoptでの分散コンピューティングの活用",
        choices: [
            "HyperoptのGridSearchCV",
            "HyperoptのRandomizedSearchCV",
            "SparkTrials",
            "HyperoptのTPE（Tree-structured Parzen Estimator）アルゴリズム"
        ],
        correctIndex: 2,
        explanation: `
            <h3>正解: SparkTrials</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p><strong>SparkTrials</strong>は、Hyperoptの評価をSparkクラスター全体に分散させます。</p>

            <h5>主な機能</h5>
            <ul>
                <li><strong>並列評価:</strong> 複数のパラメータセットを同時にテスト</li>
                <li><strong>スケーラビリティ:</strong> クラスターサイズに基づいて自動スケーリング</li>
                <li><strong>効率性:</strong> 大規模な探索空間でウォールクロック時間を削減</li>
            </ul>

            <h5>例：SparkTrialsの使用</h5>
            <pre><code class="language-python">
from hyperopt import fmin, tpe, SparkTrials

def objective(params):
    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X, y, cv=3).mean()
    return -score

search_space = {
    'n_estimators': hp.quniform('n_estimators', 50, 200, 10),
    'max_depth': hp.choice('max_depth', [5, 10, 15])
}

# SparkTrialsを使用して4つの並列ワーカーで実行
spark_trials = SparkTrials(parallelism=4)
best = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=100,
    trials=spark_trials
)
            </code></pre>

            <h4>パフォーマンスの比較</h4>
            <table border="1" cellpadding="5">
                <tr>
                    <th>方法</th>
                    <th>評価</th>
                    <th>並列性</th>
                    <th>時間（100評価）</th>
                </tr>
                <tr>
                    <td>デフォルトTrials</td>
                    <td>シーケンシャル</td>
                    <td>なし</td>
                    <td>100分</td>
                </tr>
                <tr>
                    <td>SparkTrials（4ワーカー）</td>
                    <td>並列</td>
                    <td>4x</td>
                    <td>~25分</td>
                </tr>
            </table>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>GridSearchCV/RandomizedSearchCV:</strong> scikit-learnのツールで、Hyperoptの機能ではない</li>
                <li><strong>TPEアルゴリズム:</strong> 最適化戦略だが、分散化はしない（SparkTrialsと組み合わせて使用）</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>大規模なHyperoptの最適化の場合：</p>
            <ul>
                <li>大きな探索空間にはSparkTrialsを使用</li>
                <li>parallelismをクラスターワーカー数に設定</li>
                <li>評価時間が長い場合、<code>timeout</code>パラメータを設定</li>
            </ul>

            <p><strong>プロのヒント:</strong> SparkTrialsはTPEアルゴリズムと組み合わせて、スマートな探索と並列化の両方を実現します：</p>
            <pre><code class="language-python">
best = fmin(fn=objective, space=search_space, algo=tpe.suggest, trials=spark_trials)
            </code></pre>
        `
    },
    {
        number: 27,
        domain: "Databricks ML",
        question: "データサイエンティストが、MLflowを使用してモデルとそのメタデータを追跡しています。モデルの異なるバージョンと本番環境へのデプロイメントを管理したいと考えています。どのMLflowの機能がこの目的に役立ちますか？",
        keyPoint: "MLflowでのモデルバージョン管理とデプロイメント管理",
        choices: [
            "MLflow Tracking",
            "MLflow Projects",
            "MLflow Models",
            "MLflow Model Registry"
        ],
        correctIndex: 3,
        explanation: `
            <h3>正解: MLflow Model Registry</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p><strong>MLflow Model Registry</strong>は、モデルのライフサイクル管理のために設計されています：</p>

            <h5>主な機能</h5>
            <ul>
                <li><strong>バージョン管理:</strong> 各モデルの複数バージョンを保存</li>
                <li><strong>ステージング:</strong> モデルを「Staging」「Production」「Archived」に昇格</li>
                <li><strong>アクセス制御:</strong> 本番モデルへのデプロイを制限</li>
                <li><strong>監査ログ:</strong> すべての変更を追跡</li>
            </ul>

            <h5>例：Model Registryの使用</h5>
            <pre><code class="language-python">
import mlflow
import mlflow.sklearn

# ステップ1: モデルをトレーニングしてログ
with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "model")

# ステップ2: モデルをレジストリに登録
model_uri = "runs:/&lt;run_id&gt;/model"
mlflow.register_model(model_uri, "MyModel")

# ステップ3: モデルをProductionに昇格
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="MyModel",
    version=1,
    stage="Production"
)

# ステップ4: 本番モデルをロード
model = mlflow.pyfunc.load_model("models:/MyModel/Production")
predictions = model.predict(X_test)
            </code></pre>

            <h4>モデルのライフサイクル</h4>
            <p>None（新規）→ Staging（テスト）→ Production（デプロイ済み）→ Archived（非推奨）</p>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>MLflow Tracking:</strong> 実験をログするが、バージョン管理/デプロイはしない</li>
                <li><strong>MLflow Projects:</strong> 再現可能な実行のパッケージング用</li>
                <li><strong>MLflow Models:</strong> モデルの保存形式だが、レジストリ機能なし</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>本番環境のMLの場合：</p>
            <ul>
                <li>常にModel Registryを使用してバージョン管理</li>
                <li>Stagingで新しいバージョンをテスト後、Productionに昇格</li>
                <li>Webhooksを設定してステージ移行時に通知</li>
            </ul>

            <p><strong>プロのヒント:</strong> CI/CDパイプラインと統合してモデルデプロイメントを自動化します：</p>
            <pre><code class="language-python">
# GitHub Actionsの例
mlflow models serve -m "models:/MyModel/Production" --port 5000
            </code></pre>
        `
    },
    {
        number: 28,
        domain: "Databricks ML",
        question: "データサイエンティストが、Databricks上で大規模なデータセットでモデルの推論（予測）を実行したいと考えています。どのMLflowの機能が、Databricksでのモデルデプロイメントと推論をサポートしていますか？",
        keyPoint: "Databricksでのモデルデプロイメントと推論",
        choices: [
            "MLflow Tracking",
            "MLflow Projects",
            "MLflow Models",
            "MLflow Model Serving"
        ],
        correctIndex: 3,
        explanation: `
            <h3>正解: MLflow Model Serving</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p><strong>MLflow Model Serving</strong>は、登録されたモデルをREST APIエンドポイントとしてデプロイします。</p>

            <h5>主な機能</h5>
            <ul>
                <li><strong>リアルタイム推論:</strong> 低レイテンシーのAPIコール</li>
                <li><strong>自動スケーリング:</strong> トラフィックに基づいてスケールアップ/ダウン</li>
                <li><strong>バージョン管理:</strong> Model Registryのステージ（Staging/Production）から直接サービング</li>
                <li><strong>監視:</strong> リクエストレート、レイテンシー、エラーを追跡</li>
            </ul>

            <h5>例：Databricksでモデルをサービング</h5>
            <pre><code class="language-python">
# ステップ1: モデルをModel Registryに登録
mlflow.register_model("runs:/&lt;run_id&gt;/model", "MyModel")

# ステップ2: Databricks UIでModel Servingエンドポイントを有効化
# ナビゲート: Machine Learning → Models → MyModel → "Enable Serving"

# ステップ3: APIを介して予測をリクエスト
import requests
import json

url = "https://&lt;databricks-instance&gt;/model/MyModel/Production/invocations"
headers = {"Authorization": f"Bearer {token}"}
data = {"dataframe_records": [{"feature1": 1.0, "feature2": 2.0}]}

response = requests.post(url, headers=headers, json=data)
print(response.json())  # {"predictions": [0.85]}
            </code></pre>

            <h4>バッチ推論 vs リアルタイム推論</h4>
            <table border="1" cellpadding="5">
                <tr>
                    <th>方法</th>
                    <th>用途</th>
                    <th>ツール</th>
                </tr>
                <tr>
                    <td>バッチ推論</td>
                    <td>大規模なオフライン予測</td>
                    <td>Spark UDF（<code>mlflow.pyfunc.spark_udf</code>）</td>
                </tr>
                <tr>
                    <td>リアルタイム推論</td>
                    <td>低レイテンシーのオンライン予測</td>
                    <td>MLflow Model Serving</td>
                </tr>
            </table>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>MLflow Tracking:</strong> 実験をログするが、推論エンドポイントを作成しない</li>
                <li><strong>MLflow Projects:</strong> 再現可能な実行のパッケージング用</li>
                <li><strong>MLflow Models:</strong> モデル保存形式だが、サービング機能なし（Servingがこれを拡張）</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>本番環境の推論の場合：</p>
            <ul>
                <li>リアルタイム予測には<strong>Model Serving</strong>を使用</li>
                <li>大規模なバッチ処理には<strong>Spark UDF</strong>を使用</li>
                <li>カナリアデプロイメントのために複数バージョンをテスト</li>
            </ul>

            <p><strong>プロのヒント:</strong> バッチ推論の例：</p>
            <pre><code class="language-python">
# Spark DataFrameでの大規模な予測
model_udf = mlflow.pyfunc.spark_udf(spark, "models:/MyModel/Production")
df_with_predictions = df.withColumn("prediction", model_udf(*df.columns))
            </code></pre>
        `
    },
    {
        number: 29,
        domain: "Databricks ML",
        question: "データサイエンティストが、探索的データ分析（EDA）のためにDatabricksを使用しており、異なるカテゴリ間での数値特徴の分布を視覚化したいと考えています。このような視覚化を作成するために最も適したDatabricksライブラリまたはツールは何ですか？",
        keyPoint: "Databricksでの探索的データ分析と視覚化",
        choices: [
            "Matplotlib",
            "Databricks Display関数",
            "MLlib CrossValidator",
            "Databricks Delta"
        ],
        correctIndex: 1,
        explanation: `
            <h3>正解: Databricks Display関数</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>Databricksノートブックの<code>display()</code>関数は、EDA中の迅速でインタラクティブな視覚化のために最適化されています。</p>

            <h5>主な機能</h5>
            <ul>
                <li><strong>ワンクリックチャート:</strong> ヒストグラム、ボックスプロット、棒グラフを組み込みサポート</li>
                <li><strong>大規模データの効率的処理:</strong> サンプリングまたは集約によりデータを処理</li>
                <li><strong>カテゴリ別のグループ化:</strong> カテゴリごとに数値特徴の分布をプロット</li>
            </ul>

            <h5>例：display()の使用</h5>
            <pre><code class="language-python">
df = spark.sql("SELECT category, numerical_feature FROM data")
display(df)  # チャートアイコンを使用してカテゴリ別の分布をプロット
            </code></pre>

            <h4>他の選択肢に対する主な利点</h4>
            <ul>
                <li><strong>Matplotlibと比較:</strong> 基本的なプロットにコードが不要（vs Matplotlibのコーディングオーバーヘッド）</li>
                <li><strong>統合:</strong> Databricksとネイティブ統合（vs 外部ライブラリ）</li>
                <li><strong>スケーラビリティ:</strong> ビッグデータ対応（vs Matplotlibは大きなDataFrameで苦労）</li>
            </ul>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>Matplotlib:</strong> 手動コーディングが必要で、ビッグデータでは効率が悪い</li>
                <li><strong>MLlib CrossValidator:</strong> ハイパーパラメータチューニング用で、視覚化用ではない</li>
                <li><strong>Databricks Delta:</strong> ストレージ形式で、視覚化ツールではない</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>Databricksでの迅速でスケーラブルなEDA視覚化の場合：</p>
            <ul>
                <li>インタラクティブチャートには<code>display(df)</code>を使用</li>
                <li>カスタムプロットにはSeaborn/Plotlyと組み合わせ（ただし、ビッグデータでは遅くなる可能性あり）</li>
            </ul>

            <p><strong>プロのヒント:</strong> display()の<code>groupBy</code>オプションを使用して、カテゴリ別の分布を分割します：</p>
            <pre><code class="language-python">
display(df.groupBy("category"))  # カテゴリ別のプロットを自動生成
            </code></pre>
        `
    },
    {
        number: 30,
        domain: "Pandas API on Spark",
        question: "DataFrame.pandas_on_spark.transform_batch()とDataFrame.pandas_on_spark.apply_batch()の主な違いは何ですか？",
        keyPoint: "pandas-on-Sparkでのtransform_batchとapply_batchの違い",
        choices: [
            "transform_batchとapply_batchは互換性がある",
            "transform_batchは入力と出力の長さが同じである必要があるが、apply_batchにはこの制限がない",
            "apply_batchは入力と出力の長さが同じである必要があるが、transform_batchにはこの制限がない",
            "transform_batchとapply_batchの両方とも同じ長さの制限がある"
        ],
        correctIndex: 1,
        explanation: `
            <h3>正解: transform_batchは入力と出力の長さが同じである必要があるが、apply_batchにはこの制限がない</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>

            <h5>transform_batch()</h5>
            <p><strong>入出力の長さが一致する必要あり:</strong> 関数は入力と同じ行数のDataFrame/Seriesを返す必要があります。行ごとの変換に最適（例：スケーリング、フィルタリング）。</p>
            <pre><code class="language-python">
def scale(df):
    return df * 2  # 出力長 = 入力長
df.pandas_on_spark.transform_batch(scale)
            </code></pre>

            <h5>apply_batch()</h5>
            <p><strong>柔軟な出力長:</strong> 任意のサイズのDataFrame/Seriesを返すことができます（例：集約、グループごとの統計）。</p>
            <pre><code class="language-python">
def summary(df):
    return df.mean()  # 出力長 ≠ 入力長（集約）
df.pandas_on_spark.apply_batch(summary)
            </code></pre>

            <h4>主な違い</h4>
            <table border="1" cellpadding="5">
                <tr>
                    <th>メソッド</th>
                    <th>入出力の長さ</th>
                    <th>用途</th>
                </tr>
                <tr>
                    <td><code>transform_batch</code></td>
                    <td>一致する必要あり</td>
                    <td>行ごとの操作</td>
                </tr>
                <tr>
                    <td><code>apply_batch</code></td>
                    <td>異なってもよい</td>
                    <td>集約、サマリー</td>
                </tr>
            </table>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>「互換性がある」:</strong> 異なる目的を持つ（固定 vs 柔軟な出力サイズ）</li>
                <li><strong>「apply_batchに長さ制限あり」:</strong> 誤り—apply_batchは可変出力を許可</li>
                <li><strong>「同じ制限」:</strong> 誤り—transform_batchのみが長さの一致を強制</li>
            </ul>

            <h4>重要ポイント</h4>
            <ul>
                <li>1:1の行変換には<code>transform_batch</code>を使用</li>
                <li>集約または可変長出力には<code>apply_batch</code>を使用</li>
            </ul>

            <p><strong>プロのヒント:</strong> グループ化された操作の場合、<code>groupby.apply_batch()</code>と組み合わせます。</p>
        `
    },
    {
        number: 31,
        domain: "Scaling ML Models",
        question: "チームが、分散コンピューティング環境で大量のデータを処理する機械学習プロジェクトに取り組んでいます。このシナリオでデータ処理効率を最適化するための主要な考慮事項は何ですか？",
        keyPoint: "分散コンピューティングでのデータ処理効率の最適化",
        choices: [
            "データストレージの最小化",
            "データレプリケーションの最大化",
            "ノード間のデータ転送の削減",
            "データの複雑性の増加"
        ],
        correctIndex: 2,
        explanation: `
            <h3>正解: ノード間のデータ転送の削減</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>分散コンピューティング（例：Spark）では、データ転送（シャッフリング）がノード間で発生することが主要なボトルネックです。最適化には以下が含まれます：</p>
            <ul>
                <li><strong>パーティショニング:</strong> <code>repartition()</code>または<code>coalesce()</code>を使用してデータをワーカーノードに整列</li>
                <li><strong>ブロードキャスト変数:</strong> 小さなデータセット用（<code>spark.sparkContext.broadcast()</code>）</li>
                <li><strong>スキューの回避:</strong> データ分散をバランスさせて、過負荷ノードを防ぐ</li>
            </ul>

            <h5>例：シャッフルの最小化</h5>
            <pre><code class="language-python">
# キーでの事前パーティショニングによりシャッフルを最小化
df = df.repartition(100, "user_id")  # キーで均等に分散
            </code></pre>

            <h4>パフォーマンスへの影響</h4>
            <p>シャッフルが少ない → より高速な実行と低いネットワークオーバーヘッド</p>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>ストレージの最小化:</strong> 転送よりも重要度が低い（ストレージは安価、ネットワークI/Oは高コスト）</li>
                <li><strong>レプリケーションの最大化:</strong> ストレージと転送コストを増加（アンチパターン）</li>
                <li><strong>複雑性の増加:</strong> 最適化を困難にする（シンプルさが効率を高める）</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>分散MLの効率性のために：</p>
            <ul>
                <li>シャッフルを最小化（結合戦略を賢く使用）</li>
                <li>再利用されるDataFrameにはキャッシング（<code>df.cache()</code>）を活用</li>
                <li>Spark UIでシャッフル/転送メトリクスを監視</li>
            </ul>

            <p><strong>プロのヒント:</strong> Delta Lakeを使用して、最適化されたファイル構成（Z-ordering、compaction）を実現します。</p>
        `
    },
    {
        number: 32,
        domain: "Scaling ML Models",
        question: "分散コンピューティングシステムにおいて、タスクフュージョンは何を含みますか？",
        keyPoint: "分散システムでのタスクフュージョンの概念",
        choices: [
            "小さなタスクをより大きなタスクに結合する",
            "大きなタスクを小さなサブタスクに分解する",
            "ノード間でタスクを分散する",
            "タスク実行を同期する"
        ],
        correctIndex: 0,
        explanation: `
            <h3>正解: 小さなタスクをより大きなタスクに結合する</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>分散システム（例：Apache Spark）でのタスクフュージョンは、依存する小さなタスクをより大きなタスクにマージすることを指します。これにより：</p>
            <ul>
                <li><strong>スケジューリングオーバーヘッドの削減:</strong> 管理するタスク数が減少</li>
                <li><strong>データシャッフルの最小化:</strong> 中間結果がローカルに留まる</li>
                <li><strong>CPU/メモリローカリティの向上:</strong> 連続した操作が一緒に実行される</li>
            </ul>

            <h5>例</h5>
            <p>Sparkは、隣接するナロー変換（例：<code>map</code> → <code>filter</code>）を単一のステージに融合します。</p>

            <h4>主なメリット</h4>
            <p>冗長なタスク起動とデータ転送を回避することで、より高速な実行を実現します。</p>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>タスクの分解:</strong> これはタスク分解（フュージョンの逆）</li>
                <li><strong>タスクの分散:</strong> タスク並列性を指し、フュージョンではない</li>
                <li><strong>タスクの同期:</strong> 一貫性を確保するが、タスクをマージしない</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>タスクフュージョンはSparkのコア最適化です。以下を介して有効化します：</p>
            <pre><code class="language-python">
spark.conf.set("spark.default.parallelism", "auto")  # フュージョンを最適化
            </code></pre>

            <p><strong>プロのヒント:</strong> Spark UIを使用して、DAG視覚化で融合されたステージを検査します。</p>
        `
    },
    {
        number: 33,
        domain: "Scaling ML Models",
        question: "チームが、分散コンピューティング環境で非構造化テキストデータを処理する必要がある機械学習モデルを設計しています。分析のためのテキストデータの効率的なインデックス作成と検索を可能にする技術はどれですか？",
        keyPoint: "分散環境でのテキストデータのインデックス作成と検索",
        choices: [
            "テキストクラスタリング",
            "テキストインデックス作成",
            "テキストパーティショニング",
            "テキスト圧縮"
        ],
        correctIndex: 1,
        explanation: `
            <h3>正解: テキストインデックス作成</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>テキストインデックス作成（例：逆インデックスまたはTF-IDFベクトルの使用）は、以下のための最も効率的な方法です：</p>
            <ul>
                <li><strong>高速検索:</strong> 用語の迅速なルックアップを可能にする（本の索引のように）</li>
                <li><strong>分散処理:</strong> Apache LuceneやSpark MLlibの<code>CountVectorizer</code>などのフレームワークを介してスケール</li>
            </ul>

            <h5>Sparkでの例</h5>
            <pre><code class="language-python">
from pyspark.ml.feature import CountVectorizer

# 用語頻度のインデックスを作成
vectorizer = CountVectorizer(inputCol="text", outputCol="features")
model = vectorizer.fit(text_df)  # インデックスを構築
indexed_df = model.transform(text_df)
            </code></pre>

            <h4>主な利点</h4>
            <ul>
                <li><strong>クエリのサポート:</strong> 特定の用語やフレーズを検索</li>
                <li><strong>NLPパイプラインとの連携:</strong> LDAやword2vecなどのモデルに供給</li>
            </ul>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>テキストクラスタリング:</strong> 類似したテキストをグループ化する（例：トピックモデリング）が、検索を最適化しない</li>
                <li><strong>テキストパーティショニング:</strong> データを物理的に分割する（例：ドキュメントIDで）が、用語レベルの検索は不可</li>
                <li><strong>テキスト圧縮:</strong> ストレージを削減するが、直接分析を妨げる（解凍が必要）</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>スケーラブルなテキスト処理のために：</p>
            <ul>
                <li>テキストをインデックス化（例：TF-IDF、BM25）</li>
                <li>本番環境では分散検索を使用（Elasticsearch/Solr）</li>
                <li>高度なパイプラインのためにSpark NLPと統合</li>
            </ul>

            <p><strong>プロのヒント:</strong> 低レイテンシーの検索のために、Delta Lake + 主要な用語でのZ-orderingで事前インデックス化します。</p>
        `
    },
    {
        number: 34,
        domain: "Model Training and Tuning",
        question: "チームが大規模データセットを含む機械学習プロジェクトに取り組んでいます。Databricks MLlibがサポートする、モデルトレーニングのためのデータの効率的なサンプリングと処理を支援する技術はどれですか？",
        keyPoint: "MLlibでの効率的なデータサンプリング",
        choices: [
            "データ補完",
            "層別サンプリング",
            "外れ値検出",
            "特徴量スケーリング"
        ],
        correctIndex: 1,
        explanation: `
            <h3>正解: 層別サンプリング</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>Spark MLlibの層別サンプリング（<code>DataFrame.stat.sampleBy()</code>）は、以下により代表的なサブセットを保証します：</p>
            <ul>
                <li><strong>クラス比率の保持:</strong> 不均衡なデータセットに重要</li>
                <li><strong>分散効率:</strong> Spark経由で大規模データセットにスケール</li>
            </ul>

            <h5>例</h5>
            <pre><code class="language-python">
from pyspark.sql.functions import col

# 各クラスの10%をサンプリング
fractions = {"class_A": 0.1, "class_B": 0.1}
sampled_df = df.stat.sampleBy("label", fractions, seed=42)
            </code></pre>

            <h4>主なメリット</h4>
            <ul>
                <li><strong>バイアスの回避:</strong> 元の分布を維持</li>
                <li><strong>計算コストの削減:</strong> シグナルを失うことなく、より小さなトレーニングセット</li>
            </ul>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>データ補完:</strong> 欠損値を処理するが、データサイズを削減しない</li>
                <li><strong>外れ値検出:</strong> 異常を特定するが、サンプリングしない</li>
                <li><strong>特徴量スケーリング:</strong> 値を正規化（例：MinMaxScaler）するが、データをサブセット化しない</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>大規模MLのために：</p>
            <ul>
                <li>層別サンプリングを使用して、管理可能でバランスの取れたデータセットを作成</li>
                <li>MLlibの<code>CrossValidator</code>と組み合わせて堅牢な評価を実現</li>
            </ul>

            <p><strong>プロのヒント:</strong> 時系列データの場合、時間ベースのサンプリングを使用（例：<code>df.filter("date <= '2023-01-01'")</code>）。</p>
        `
    },
    {
        number: 35,
        domain: "Databricks ML",
        question: "データサイエンティストが、新しい列の作成や変換を含む広範な特徴量エンジニアリングを伴うDatabricksノートブックで作業しています。この特徴量エンジニアリングロジックを再利用可能なコンポーネントにカプセル化したいと考えています。これを達成するための推奨アプローチは何ですか？",
        keyPoint: "Databricksでの特徴量エンジニアリングの再利用可能なカプセル化",
        choices: [
            "カスタムPySpark UDF（ユーザー定義関数）を定義し、DataFrameに適用する",
            "特徴量エンジニアリングロジックのためにSpark MLlib Transformerクラスを作成する",
            "map関数を使用して特徴量エンジニアリング変換を適用する",
            "中間DataFrameを保存し、必要に応じて他のノートブックでロードする"
        ],
        correctIndex: 1,
        explanation: `
            <h3>正解: 特徴量エンジニアリングロジックのためにSpark MLlib Transformerクラスを作成する</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>Spark MLlib Transformerは、特徴量エンジニアリングロジックをカプセル化する標準的で再利用可能な方法です。これらは：</p>
            <ul>
                <li><strong>Spark ML Pipelineとシームレスに統合</strong></li>
                <li><strong>保存/ロード可能</strong>（例：<code>save()</code>/<code>load()</code>経由）</li>
                <li><strong>分散実行をサポート</strong></li>
            </ul>

            <h5>例</h5>
            <pre><code class="language-python">
from pyspark.ml import Transformer
from pyspark.sql.functions import log

class LogTransformer(Transformer):
    def _transform(self, df):
        return df.withColumn("log_feature", log(df["feature"]))

# 使用法
log_transformer = LogTransformer()
transformed_df = log_transformer.transform(input_df)
            </code></pre>

            <h4>主なメリット</h4>
            <ul>
                <li><strong>再利用性:</strong> 同じロジックをノートブック/ジョブ間で適用</li>
                <li><strong>テスト可能性:</strong> Transformerを独立して単体テスト</li>
                <li><strong>Pipelineの互換性:</strong> VectorAssembler、StandardScalerなどと組み合わせ可能</li>
            </ul>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>カスタムUDF:</strong> 効率が低い（Python-シリアライゼーションオーバーヘッド）で再利用が困難</li>
                <li><strong>map関数:</strong> データをPythonに強制（遅い）し、DataFrameの最適化を壊す</li>
                <li><strong>DataFrame保存/ロード:</strong> ロジックをカプセル化せず、データをキャッシュするだけ</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>本番グレードの特徴量エンジニアリングのために：</p>
            <ul>
                <li>複雑なロジックにはカスタムTransformerを記述</li>
                <li>Pipelinesでチェーン化（<code>Pipeline.fit()</code>/<code>transform()</code>）</li>
                <li>デプロイメントのためにPipelinesを保存</li>
            </ul>

            <p><strong>プロのヒント:</strong> Transformersで<code>@keyword_only</code>を使用して、MLflow互換のパラメータロギングを実現します。</p>
        `
    },
    {
        number: 36,
        domain: "Pandas API on Spark",
        question: "pandas API on Sparkの新しいタイプヒントスタイルでは、戻り値の型でSeriesの名前をどのように指定しますか？",
        keyPoint: "pandas-on-Sparkでのタイプヒントの構文",
        choices: [
            "文字列のリストとして",
            "辞書として",
            "文字列として、その後に型を続ける",
            "Seriesの名前は戻り値の型では指定されない"
        ],
        correctIndex: 2,
        explanation: `
            <h3>正解: 文字列として、その後に型を続ける</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>pandas API on Sparkでは、新しいタイプヒントスタイル（PEP 484）は、次の形式を使用してSeriesの名前と型を指定します：</p>
            <pre><code class="language-python">
def function() -> ps.Series["name", dtype]:
            </code></pre>
            <ul>
                <li><code>"name"</code>: 文字列リテラルとしてのSeries名</li>
                <li><code>dtype</code>: データ型（例：<code>int</code>、<code>float</code>）</li>
            </ul>

            <h5>例</h5>
            <pre><code class="language-python">
import pyspark.pandas as ps

def square(x: ps.Series["value", float]) -> ps.Series["squared_value", float]:
    return x ** 2
            </code></pre>

            <h4>主な機能</h4>
            <ul>
                <li><strong>明示的な命名:</strong> 期待される列名と実際の列名間の一貫性を保証</li>
                <li><strong>型安全性:</strong> 静的チェッカー（例：mypy）が型を検証</li>
            </ul>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>文字列のリスト:</strong> DataFrameの列名に使用され、Seriesには使用されない</li>
                <li><strong>辞書:</strong> pandas-on-Sparkのタイプヒントにそのような構文は存在しない</li>
                <li><strong>指定されない:</strong> 可能だが、タイプヒントの目的を無効にする</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>pandas-on-Sparkでの型ヒント付きSeriesのために：</p>
            <ul>
                <li><code>Series["name", dtype]</code>を使用</li>
                <li>完全な型安全性のために<code>DataFrame[{"col": dtype}]</code>と組み合わせる</li>
            </ul>

            <p><strong>プロのヒント:</strong> mypyを有効にして、型の不一致を早期に検出します。</p>
        `
    },
    {
        number: 37,
        domain: "AutoML",
        question: "Databricks AutoML APIの<code>import_notebook</code>関数の目的は何ですか？",
        keyPoint: "AutoMLのimport_notebook関数の役割",
        choices: [
            "カスタムノートブックでAutoML実行を開始する",
            "ワークスペースにMLflowアーティファクトとして保存されたトライアルノートブックをインポートする",
            "AutoML実行中に生成されたノートブックをエクスポートする",
            "MLflowモデルレジストリにモデルを登録してデプロイする"
        ],
        correctIndex: 1,
        explanation: `
            <h3>正解: ワークスペースにMLflowアーティファクトとして保存されたトライアルノートブックをインポートする</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>Databricks AutoMLの<code>import_notebook</code>は、AutoML実験中に生成されたトライアルノートブックを取得してロードするために使用されます。これらのノートブックは：</p>
            <ul>
                <li>実験実行の下でMLflowアーティファクトとして保存される</li>
                <li>特定のモデルトライアルの完全なコードを含む（ハイパーパラメータ、前処理ステップ）</li>
            </ul>

            <h5>例</h5>
            <pre><code class="language-python">
from databricks import automl

automl.import_notebook(
    run_id="123",  # トライアルのMLflow実行ID
    target_path="/Shared/automl_trial"  # ノートブックを保存するワークスペースパス
)
            </code></pre>

            <h4>主な用途</h4>
            <ul>
                <li><strong>デバッグ:</strong> 特定のトライアルがなぜ良い/悪いパフォーマンスだったかを検査</li>
                <li><strong>カスタマイズ:</strong> AutoML生成コードを変更して再利用</li>
            </ul>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>「カスタムノートブックでAutoMLを開始」:</strong> AutoML実行は<code>automl.classify()</code>/<code>regress()</code>経由で開始され、<code>import_notebook</code>では開始されない</li>
                <li><strong>「ノートブックをエクスポート」:</strong> ノートブックはアーティファクトとして自動エクスポートされる。<code>import_notebook</code>はそれらを取得する</li>
                <li><strong>「モデルを登録/デプロイ」:</strong> <code>mlflow.register_model()</code>経由で実行され、ノートブックとは無関係</li>
            </ul>

            <h4>重要ポイント</h4>
            <p><code>import_notebook</code>を使用してAutoMLトライアルを探索または拡張します。MLflow UIと組み合わせて、高パフォーマンスの実行を特定します。</p>

            <p><strong>プロのヒント:</strong> 本番環境では、カスタムノートブックをDatabricksのRepo Filesとしてログし、バージョン管理を行います。</p>
        `
    },
    {
        number: 38,
        domain: "Scaling ML Models",
        question: "チームが機械学習プロジェクトのためにSparkのパフォーマンスを最適化しています。後続のタスク中に計算オーバーヘッドを削減するために中間結果を事前計算して保存する技術は何ですか？",
        keyPoint: "Sparkでの中間結果のキャッシング",
        choices: [
            "遅延評価",
            "結果キャッシング",
            "タスクフュージョン",
            "データ圧縮"
        ],
        correctIndex: 1,
        explanation: `
            <h3>正解: 結果キャッシング</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>結果キャッシング（<code>df.cache()</code>または<code>df.persist()</code>経由）は、中間DataFrames/RDDをメモリまたはディスクに保存して、後続のアクションでの再計算を回避します。</p>

            <h4>主なメリット</h4>
            <ul>
                <li><strong>CPUオーバーヘッドの削減:</strong> 事前計算された結果を再利用</li>
                <li><strong>反復ワークフローの高速化:</strong> MLで一般的（例：特徴量エンジニアリングループ）</li>
            </ul>

            <h5>例</h5>
            <pre><code class="language-python">
df = spark.read.parquet("data.parquet").cache()  # ロード後にキャッシュ
df.count()  # キャッシュを実体化
            </code></pre>

            <h4>使用すべき場合</h4>
            <ul>
                <li><strong>再利用されるDataFrames:</strong> DataFrameが複数回アクセスされる場合</li>
                <li><strong>頻繁にクエリされる小さなテーブル:</strong> スタースキーマのディメンションテーブル</li>
            </ul>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>遅延評価:</strong> アクション（例：<code>count()</code>）が呼ばれるまで計算を延期するが、結果を保存しない</li>
                <li><strong>タスクフュージョン:</strong> スケジューリングオーバーヘッドを削減するために小さなタスクを結合（ストレージなし）</li>
                <li><strong>データ圧縮:</strong> ストレージサイズを削減するが、再計算を回避しない</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>MLパフォーマンスのために：</p>
            <ul>
                <li>特徴量エンジニアリングされたDataFramesをキャッシュ</li>
                <li>ストレージレベルを監視（大きなDataFramesには<code>StorageLevel.MEMORY_AND_DISK</code>）</li>
                <li>完了したらアンパーシスト（<code>df.unpersist()</code>）してリソースを解放</li>
            </ul>

            <p><strong>プロのヒント:</strong> <code>explain()</code>を使用して、Sparkの実行計画でキャッシュヒットを確認します。</p>
        `
    },
    {
        number: 39,
        domain: "Databricks ML",
        question: "シニアデータサイエンティストが、MLflowを使用して機械学習プロジェクトに取り組んでいます。モデルの説明と解釈可能性を可能にする機能を実装したいと考えています。モデル解釈のためにどのMLflowコンポーネントまたはライブラリを使用すべきですか？",
        keyPoint: "MLflowでのモデル解釈可能性",
        choices: [
            "mlflow.sklearn",
            "mlflow.shap",
            "mlflow.pytorch",
            "mlflow.tensorflow"
        ],
        correctIndex: 1,
        explanation: `
            <h3>正解: mlflow.shap</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p><code>mlflow.shap</code>は、モデル解釈可能性の主要なライブラリであるSHAP（SHapley Additive exPlanations）とのMLflowの組み込み統合です。これにより：</p>
            <ul>
                <li><strong>特徴量重要度スコアの生成</strong>（グローバルな解釈可能性）</li>
                <li><strong>ローカルな説明の生成</strong>（予測ごとの理由付け）</li>
                <li><strong>視覚化のログ</strong>（例：force plots、summary plots）をMLflowアーティファクトとして</li>
            </ul>

            <h5>例</h5>
            <pre><code class="language-python">
import mlflow.shap
import shap

# モデルをトレーニング
model = train_model(X_train, y_train)

# SHAPサマリーをログ
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
mlflow.shap.log_explanation(explainer, X_test)
            </code></pre>

            <h4>主な機能</h4>
            <ul>
                <li><strong>統一された追跡:</strong> SHAP出力はMLflow UIの実行のアーティファクト下に表示される</li>
                <li><strong>モデル非依存:</strong> sklearn、PyTorch、TensorFlowなどで動作</li>
            </ul>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>mlflow.sklearn/mlflow.pytorch/mlflow.tensorflow:</strong> これらはモデルとメトリクスをログするが、説明はログしない</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>MLflowでのモデル解釈可能性のために：</p>
            <ul>
                <li>SHAPベースの説明には<code>mlflow.shap</code>を使用</li>
                <li>非SHAPメソッド（例：LIME）の場合、<code>mlflow.log_artifact()</code>経由でカスタムプロットをログ</li>
            </ul>

            <p><strong>プロのヒント:</strong> MLflowのアーティファクト差分ビューを使用して、実行間で説明を比較します。</p>
        `
    },
    {
        number: 40,
        domain: "Feature Store",
        question: "マルチシリーズ予測において、<code>identity_col</code>パラメータはどのような役割を果たしますか？",
        keyPoint: "マルチシリーズ予測でのidentity_colの役割",
        choices: [
            "時系列の頻度を設定する",
            "予測のための時間列を識別する",
            "特徴量ルックアップのための主キー列を指定する",
            "マルチシリーズ予測のために時系列を識別する"
        ],
        correctIndex: 3,
        explanation: `
            <h3>正解: マルチシリーズ予測のために時系列を識別する</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>マルチシリーズ予測（例：複数の製品/店舗の売上予測）では、<code>identity_col</code>パラメータは：</p>
            <ul>
                <li>各時系列を一意に識別する（例：<code>product_id</code>、<code>store_id</code>）</li>
                <li>モデルが複数のシリーズを同時にトレーニングしながら、それぞれの異なるパターンを保持することを可能にする</li>
            </ul>

            <h5>例（Databricks AutoML）</h5>
            <pre><code class="language-python">
from databricks import automl

automl.forecast(
    df=df,
    time_col="date",
    target_col="sales",
    identity_col="product_id"  # 製品でデータをグループ化
)
            </code></pre>

            <h4>主な影響</h4>
            <p><code>identity_col</code>がないと、モデルはすべてのデータを単一のシリーズとして扱い、グループ固有のトレンドを失います。</p>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>「頻度を設定」:</strong> <code>frequency</code>によって制御される（例：<code>freq="D"</code>で日次）</li>
                <li><strong>「時間列を識別」:</strong> これは<code>time_col</code>の役割</li>
                <li><strong>「特徴量ルックアップの主キー」:</strong> 特徴量ルックアップは<code>feature_store_key</code>などのキーを使用し、<code>identity_col</code>ではない</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>マルチシリーズ予測のために：</p>
            <ul>
                <li><code>identity_col</code>を使用してシリーズをグループ化（例：製品、地域ごと）</li>
                <li><code>time_col</code>と<code>target_col</code>と組み合わせて完全な構成を実現</li>
            </ul>

            <p><strong>プロのヒント:</strong> <code>identity_col</code>が低いカーディナリティ（例：10〜100のグループ）を持つことを確保して、最適なパフォーマンスを実現します。</p>
        `
    },
    {
        number: 41,
        domain: "AutoML",
        question: "Databricks AutoMLを使用して分類モデルをトレーニングするタスクが与えられています。作業しているデータセットにはいくつかの列がありますが、そのうちのいくつかは分類タスクに無関係です。AutoMLの計算中に特定の列を除外するためにどのパラメータを使用しますか？",
        keyPoint: "AutoMLでの無関係な列の除外",
        choices: [
            "target_col",
            "exclude_cols",
            "max_trials",
            "pos_label"
        ],
        correctIndex: 1,
        explanation: `
            <h3>正解: exclude_cols</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>Databricks AutoMLの<code>exclude_cols</code>パラメータは、モデルトレーニング中に無視する列を明示的にリストします。これにより：</p>
            <ul>
                <li><strong>ノイズの除去:</strong> 無関係な特徴量（例：ID、タイムスタンプ）を除外</li>
                <li><strong>効率の向上:</strong> 計算時間とメモリ使用量を削減</li>
            </ul>

            <h5>例</h5>
            <pre><code class="language-python">
from databricks import automl

automl.classify(
    dataset=df,
    target_col="label",
    exclude_cols=["id", "timestamp"]  # 無視する列
)
            </code></pre>

            <h4>主なメリット</h4>
            <p>AutoMLが予測的な特徴量のみに焦点を当て、無関係なデータからのバイアスを回避します。</p>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>target_col:</strong> ラベル列を指定するが、除外する列ではない</li>
                <li><strong>max_trials:</strong> ハイパーパラメータトライアルの数を制御するが、特徴量選択ではない</li>
                <li><strong>pos_label:</strong> 二値分類での正のクラスを定義（列除外とは無関係）</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>クリーンなAutoML実行のために：</p>
            <ul>
                <li>非予測列を削除するために<code>exclude_cols</code>を使用</li>
                <li>外部特徴量を含めるために<code>feature_store_lookups</code>と組み合わせる</li>
            </ul>

            <p><strong>プロのヒント:</strong> <code>df.display()</code>でデータを事前検査して、無関係な列を特定します。</p>
        `
    },
    {
        number: 42,
        domain: "Databricks ML",
        question: "データサイエンティストが、DatabricksノートブックでSpark MLlibを使用して機械学習モデルをトレーニングしています。将来の参照のために、トレーニング済みモデルをDatabricks MLflow追跡サーバーに保存したいと考えています。どのコードスニペットを使用すべきですか？",
        keyPoint: "Spark MLlibモデルのMLflowへのログ記録",
        choices: [
            "mlflow.log_model(trained_model, \"model\")",
            "mlflow.start_run()\\nmlflow.spark.log_model(trained_model, \"model\")\\nmlflow.end_run()",
            "mlflow.spark.save_model(trained_model, \"model\")",
            "mlflow.spark.logModel(trained_model, \"model\")"
        ],
        correctIndex: 1,
        explanation: `
            <h3>正解: mlflow.start_run()\\nmlflow.spark.log_model(trained_model, "model")\\nmlflow.end_run()</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p><code>mlflow.spark.log_model()</code>は、Spark MLlibモデルをMLflow追跡サーバーにログする正しいメソッドです。これにより：</p>
            <ul>
                <li>モデルをMLflowアーティファクトとして保存</li>
                <li>メタデータを記録（例：Sparkバージョン、モデルタイプ）</li>
                <li>アクティブなMLflow実行が必要（したがって<code>start_run()</code>/<code>end_run()</code>）</li>
            </ul>

            <h5>例</h5>
            <pre><code class="language-python">
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
import mlflow

# Spark MLモデルをトレーニング
lr = LogisticRegression()
pipeline = Pipeline(stages=[lr])
model = pipeline.fit(train_df)

# MLflowにログ
with mlflow.start_run():
    mlflow.spark.log_model(model, "spark-model")
            </code></pre>

            <h4>主な機能</h4>
            <ul>
                <li><strong>再現性:</strong> すべての依存関係をログ（Python/Sparkバージョン）</li>
                <li><strong>デプロイメント準備完了:</strong> 後で<code>mlflow.spark.load_model()</code>経由でロード可能</li>
            </ul>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>mlflow.log_model():</strong> 汎用的だが、Spark MLlib用に最適化されていない（Spark固有のメタデータがない）</li>
                <li><strong>mlflow.spark.save_model():</strong> ディスクに保存するが、追跡サーバーにログしない</li>
                <li><strong>mlflow.spark.logModel():</strong> 誤った構文（大文字小文字を区別、正しくは<code>log_model</code>）</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>Spark MLlib + MLflowの場合：</p>
            <ul>
                <li>実行コンテキスト内で<code>mlflow.spark.log_model()</code>を使用</li>
                <li>非Sparkモデルの場合、<code>mlflow.sklearn.log_model()</code>または同等のものを使用</li>
            </ul>

            <p><strong>プロのヒント:</strong> <code>registered_model_name</code>を追加して、MLflow Model Registryにモデルを自動登録します。</p>
        `
    },
    {
        number: 43,
        domain: "Spark ML Algorithms",
        question: "Spark MLで使用されるニューラルネットワークにおいて、活性化関数の主要な役割は何ですか？",
        keyPoint: "ニューラルネットワークでの活性化関数の役割",
        choices: [
            "重みを初期化する",
            "学習率を決定する",
            "モデルに複雑性を追加する",
            "モデルに非線形性を導入する"
        ],
        correctIndex: 3,
        explanation: `
            <h3>正解: モデルに非線形性を導入する</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>活性化関数（例：ReLU、Sigmoid、Tanh）はニューラルネットワークで不可欠です。なぜなら：</p>
            <ul>
                <li><strong>非線形性を導入</strong>し、ネットワークが複雑なパターンを学習できるようにします（活性化関数がなければ、ネットワークは単なる線形回帰モデルになります）</li>
                <li><strong>ニューロン出力を制御:</strong> ニューロンが「発火」すべきか（次の層に情報を渡すか）を決定</li>
            </ul>

            <h5>Spark MLでの例（MultilayerPerceptron）</h5>
            <pre><code class="language-python">
from pyspark.ml.classification import MultilayerPerceptronClassifier

layers = [4, 5, 3]  # 入力、隠れ、出力層
mlp = MultilayerPerceptronClassifier(layers=layers, activation="relu")
            </code></pre>
            <p>ここで、<code>activation="relu"</code>が層間に非線形性を追加します。</p>

            <h4>主な影響</h4>
            <p>非線形性により、ネットワークが任意の関数を近似できます（Universal Approximation Theorem）。</p>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>「重みを初期化」:</strong> 重み初期化スキーム（例：Xavier、He）によって行われ、活性化関数ではない</li>
                <li><strong>「学習率を決定」:</strong> オプティマイザ（例：Adam、SGD）によって制御される</li>
                <li><strong>「複雑性を追加」:</strong> 部分的に真実だが、主要な役割は非線形性を可能にすること</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>Spark MLでの効果的なニューラルネットワークのために：</p>
            <ul>
                <li>隠れ層にはReLU（デフォルト）を使用（高速、勾配消失を回避）</li>
                <li>マルチクラス出力層にはSoftmaxを使用</li>
                <li>回帰以外では線形活性化（例：identity）を避ける</li>
            </ul>

            <p><strong>プロのヒント:</strong> トレーニング中にSpark UIでデッドニューロン（ReLU出力が0）を監視します。</p>
        `
    },
    {
        number: 44,
        domain: "Model Training and Tuning",
        question: "機械学習プロジェクトが、データセット内の不均衡クラスを処理することを含んでいます。より正確なモデルトレーニングのために、Databricks MLlibがサポートする、クラス不均衡に対処するのに役立つ技術はどれですか？",
        keyPoint: "MLlibでのクラス不均衡への対処",
        choices: [
            "クラス重み付け",
            "特徴量スケーリング",
            "データ拡張",
            "外れ値検出"
        ],
        correctIndex: 0,
        explanation: `
            <h3>正解: クラス重み付け</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>Spark MLlibのクラス重み付け（例：<code>weightCol</code>または<code>classWeight</code>）は、少数派クラスの誤分類により重いペナルティを科すように損失関数を調整します。これにより：</p>
            <ul>
                <li>トレーニング中の各クラスの影響をバランス</li>
                <li>データの複製や合成が不要（オーバーサンプリングとは異なる）</li>
            </ul>

            <h5>例</h5>
            <pre><code class="language-python">
from pyspark.ml.classification import LogisticRegression

# DataFrameに重みを追加
balanced_df = df.withColumn("weight", when(df["label"] == 1, 5.0).otherwise(1.0))

# 重みを使用してトレーニング
lr = LogisticRegression(weightCol="weight")
model = lr.fit(balanced_df)
            </code></pre>

            <h4>主なメリット</h4>
            <ul>
                <li><strong>データ変更なし:</strong> 元のデータセットサイズを保持</li>
                <li><strong>ネイティブSparkサポート:</strong> 大規模データセットに効率的</li>
            </ul>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>特徴量スケーリング:</strong> 特徴量を正規化（例：MinMaxScaler）するが、クラス不均衡に対処しない</li>
                <li><strong>データ拡張:</strong> 合成サンプルを作成（例：SMOTE）するが、Spark MLlibでネイティブサポートされていない</li>
                <li><strong>外れ値検出:</strong> 異常を特定するが、クラスバランスとは無関係</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>Spark MLでの不均衡データのために：</p>
            <ul>
                <li>分類器で<code>weightCol</code>を使用（LogisticRegression、RandomForest）</li>
                <li>深刻な不均衡の場合、オーバーサンプリング（カスタムPySparkコード）と組み合わせる</li>
            </ul>

            <p><strong>プロのヒント:</strong> Spark 3.0+で<code>classWeight="balanced"</code>経由で重みを自動計算します。</p>
        `
    },
    {
        number: 45,
        domain: "Databricks ML",
        question: "機械学習エンジニアが、MLflowで多数の実験を管理しており、効率的に整理したいと考えています。関連する実験をグループ化するための階層構造を作成したいと考えています。この目的のためにどのMLflow機能を使用すべきですか？",
        keyPoint: "MLflowでの実験の階層的な整理",
        choices: [
            "mlflow.create_experiment",
            "mlflow.log_param",
            "mlflow.set_experiment",
            "mlflow.start_run"
        ],
        correctIndex: 2,
        explanation: `
            <h3>正解: mlflow.set_experiment</h3>

            <h4>解説</h4>
            <p>MLflowでは、<code>mlflow.set_experiment()</code>を使用して実験を構造化された階層形式に整理します。これにより、機械学習エンジニアは関連する実行を特定の実験名の下にグループ化でき、複数の実験の管理と比較が容易になります。</p>
            <p><code>mlflow.set_experiment("experiment_name")</code>を使用して実験を設定すると、その後のすべてのMLflow実行がその特定の実験の下にログされます。これは、複数のモデル、ハイパーパラメータチューニング、または異なるデータセット間での実験の追跡を扱う場合に便利です。</p>

            <h5>使用例：MLflowでの実験の整理</h5>
            <pre><code class="language-python">
import mlflow

# 関連する実行をグループ化するために実験を設定
mlflow.set_experiment("fraud_detection_experiment")

# MLflow実行を開始
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.92)
            </code></pre>
            <p>これにより、「fraud_detection_experiment」の下のすべての実行がMLflowでグループ化されます。後でMLflow UIでこの実験の下の異なる実行を比較できます。</p>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>mlflow.create_experiment:</strong> プログラムで新しいMLflow実験を作成するために使用されますが、実行をログするためのアクティブな実験を設定しません</li>
                <li><strong>mlflow.log_param:</strong> 特定の実行のハイパーパラメータをログするために使用されますが、実験を整理しません</li>
                <li><strong>mlflow.start_run:</strong> 個別のMLflow実行を開始しますが、<code>mlflow.set_experiment()</code>を使用しない限り実験に割り当てません</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>構造化された実験の下で複数のMLflow実行を効率的に整理および管理するには、実行を開始する前に<code>mlflow.set_experiment("experiment_name")</code>を使用します。</p>
        `
    },
    {
        number: 46,
        domain: "AutoML",
        question: "回帰AutoMLタスクで、モデルのパフォーマンスを評価およびランク付けするために使用されるメトリックをカスタマイズしたいと考えています。主要メトリックを指定できるパラメータはどれですか？",
        keyPoint: "AutoMLでの主要メトリックのカスタマイズ",
        choices: [
            "primary_metric",
            "time_col",
            "exclude_cols",
            "max_trials"
        ],
        correctIndex: 0,
        explanation: `
            <h3>正解: primary_metric</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>Databricks AutoMLの<code>primary_metric</code>パラメータは、トレーニング中にモデルをランク付けするために使用される評価メトリックを明示的に定義します。</p>
            <p>サポートされる回帰メトリックには以下が含まれます：</p>
            <ul>
                <li>rmse（Root Mean Squared Error）</li>
                <li>mse（Mean Squared Error）</li>
                <li>r2（R-squared）</li>
                <li>mae（Mean Absolute Error）</li>
            </ul>

            <h5>例</h5>
            <pre><code class="language-python">
from databricks import automl

automl.regress(
    dataset=df,
    target_col="price",
    primary_metric="rmse"  # 最小RMSEに最適化
)
            </code></pre>

            <h4>主な影響</h4>
            <p>AutoMLリーダーボードで「最良」としてフラグ付けされるモデルバージョンを決定します。</p>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>time_col:</strong> 時系列予測のタイムスタンプ列を指定するが、メトリック選択ではない</li>
                <li><strong>exclude_cols:</strong> トレーニング中に無視する列をリスト</li>
                <li><strong>max_trials:</strong> ハイパーパラメータトライアルの数を制御するが、メトリックではない</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>カスタマイズされた回帰評価のために：</p>
            <ul>
                <li>ビジネス目標に合わせて<code>primary_metric</code>を設定（例：外れ値に堅牢な<code>mae</code>）</li>
                <li>AutoML UIでリーダーボードメトリックを確認（<code>display(automl_results)</code>）</li>
            </ul>

            <p><strong>プロのヒント:</strong> AutoML出力の<code>eval_metrics</code>を使用して、すべてのメトリックを並べて比較します。</p>
        `
    },
    {
        number: 47,
        domain: "ML Workflows",
        question: "機械学習モデルがトレーニングデータに対してアンダーフィッティングしており、複雑性が不足していることを示しています。Databricks MLlibがサポートする、アンダーフィッティングに対処するためにどのアプローチを採用しますか？",
        keyPoint: "アンダーフィッティングへの対処方法",
        choices: [
            "モデルの複雑性を増加させる",
            "モデルの複雑性を減少させる",
            "正則化を増加させる",
            "正則化を減少させる"
        ],
        correctIndex: 0,
        explanation: `
            <h3>正解: モデルの複雑性を増加させる</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>アンダーフィッティングは、モデルがデータのパターンをキャプチャするには単純すぎる場合に発生します。これに対処するには：</p>
            <ul>
                <li>複雑性を増加させる方法：
                    <ul>
                        <li>より強力なアルゴリズムを使用（例：線形回帰からランダムフォレストに切り替え）</li>
                        <li>より多くの特徴量または相互作用項を追加</li>
                        <li>正則化を減少（例：Spark MLlibで<code>regParam</code>を低く）</li>
                    </ul>
                </li>
            </ul>

            <h5>Spark MLlibでの例</h5>
            <pre><code class="language-python">
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(
    numTrees=100,  # より多くの木 = より複雑
    maxDepth=10    # より深い木 = より複雑
)
            </code></pre>

            <h4>主なメリット</h4>
            <p>より高い複雑性により、モデルがトレーニングデータによりよくフィットします（ただし、過学習を監視）。</p>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>複雑性を減少:</strong> アンダーフィッティングを悪化させる（既に単純すぎる）</li>
                <li><strong>正則化を増加:</strong> モデル容量をさらに制限（例：高い<code>regParam</code>が係数を縮小）</li>
                <li><strong>正則化を減少:</strong> 役立つ（複雑性を増加させるサブセット）が、他の方法より直接的でない</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>Spark MLlibでのアンダーフィッティングのために：</p>
            <ul>
                <li>複雑性を向上：木（<code>numTrees</code>）、深さ（<code>maxDepth</code>）、または特徴量を追加</li>
                <li>非線形モデルに切り替え（例：GBT、ニューラルネット）</li>
                <li>検証：テストパフォーマンスが改善するかチェック（過学習を回避）</li>
            </ul>

            <p><strong>プロのヒント:</strong> <code>CrossValidator</code>を使用して複雑性のチューニングを自動化します。</p>
        `
    },
    {
        number: 48,
        domain: "Databricks ML",
        question: "データエンジニアリングチームが、Databricksを使用して外部データベースからデータを読み取り、「external_data」という名前のDatabricks Deltaテーブルに書き込んでいます。Deltaテーブルの既存データが上書きされないようにしたいと考えています。どのコードスニペットを使用すべきですか？",
        keyPoint: "Deltaテーブルへのデータ追加（上書きなし）",
        choices: [
            "external_data.write.format(\"delta\").mode(\"overwrite\").saveAsTable(\"external_data\")",
            "external_data.write.format(\"delta\").mode(\"append\").saveAsTable(\"external_data\")",
            "external_data.write.format(\"delta\").mode(\"ignore\").saveAsTable(\"external_data\")",
            "spark.sql(\"INSERT INTO external_data SELECT * FROM external_data_source\")"
        ],
        correctIndex: 1,
        explanation: `
            <h3>正解: external_data.write.format("delta").mode("append").saveAsTable("external_data")</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p><code>mode("append")</code>は、既存のDeltaテーブルの以前のレコードを削除せずに新しいデータを追加することを保証します。</p>
            <ul>
                <li><strong>冪等性:</strong> 再実行に対して安全（データがユニークであれば重複なし）</li>
            </ul>

            <h5>例</h5>
            <pre><code class="language-python">
(external_data.write
    .format("delta")
    .mode("append")  # キーパラメータ
    .saveAsTable("external_data")
)
            </code></pre>

            <h4>主なメリット</h4>
            <ul>
                <li><strong>履歴の保持:</strong> Delta Lakeのタイムトラベルがすべてのバージョンを保持</li>
                <li><strong>ACIDコンプライアンス:</strong> データ整合性を保証</li>
            </ul>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>overwrite:</strong> 既存データを削除（要件に違反）</li>
                <li><strong>ignore:</strong> テーブルが存在する場合、書き込みをスキップ（データが追加されない）</li>
                <li><strong>SQL INSERT INTO:</strong> 機能するが、バッチ書き込みに対して<code>append</code>より明示的でない</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>Delta Lakeへの増分書き込みのために：</p>
            <ul>
                <li>既存データを保持するために<code>append</code>を使用</li>
                <li>重複排除には、<code>MERGE</code>ロジックを追加</li>
            </ul>

            <p><strong>プロのヒント:</strong> 成長するテーブルでクエリパフォーマンスを維持するために、<code>OPTIMIZE</code>と<code>ZORDER</code>を使用します。</p>
        `
    },
    {
        number: 49,
        domain: "AutoML",
        question: "AutoMLはそのプロセス中に何を実行し、記録しますか？",
        keyPoint: "AutoMLのプロセスと記録内容",
        choices: [
            "手動データ準備を実行する",
            "サマリー統計のみを記録する",
            "複数のモデルを作成、調整、評価する一連のトライアルを記録する",
            "モデル評価のための単一トライアルを記録する"
        ],
        correctIndex: 2,
        explanation: `
            <h3>正解: 複数のモデルを作成、調整、評価する一連のトライアルを記録する</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>Databricks AutoMLは、以下によりエンドツーエンドのMLワークフローを自動化します：</p>
            <ul>
                <li><strong>複数のトライアルを生成:</strong> 各トライアルは、ユニークなハイパーパラメータで異なるモデル（例：Random Forest、XGBoost）をトレーニング</li>
                <li><strong>メトリクス/アーティファクトをログ:</strong> MLflowで各トライアルのパラメータ、パフォーマンスメトリクス、ノートブックを記録</li>
                <li><strong>最良のモデルを選択:</strong> 主要メトリック（例：accuracy、RMSE）に基づいてトライアルをランク付け</li>
            </ul>

            <h5>出力例</h5>
            <pre><code class="language-python">
from databricks import automl
summary = automl.classify(df, target_col="label")
display(summary.trials)  # メトリクス付きのすべてのトライアルを表示
            </code></pre>

            <h4>主な機能</h4>
            <ul>
                <li><strong>手動介入なし:</strong> 特徴量エンジニアリング、分割、チューニングを処理</li>
                <li><strong>再現性:</strong> 各トライアルのコードをノートブックとして保存</li>
            </ul>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>「手動データ準備」:</strong> AutoMLはこれを自動化（例：欠損値補完、エンコーディング）</li>
                <li><strong>「サマリー統計のみ」:</strong> サマリーだけでなく、完全なトライアル詳細を記録</li>
                <li><strong>「単一トライアル」:</strong> AutoMLは最良のモデルを見つけるために数十のトライアルを実行</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>AutoMLは、モデル比較のためのトライアルのリーダーボードを提供します。最良のモデルを使用するか、生成されたノートブックをカスタマイズします。</p>

            <p><strong>プロのヒント:</strong> 大規模データセットの場合、<code>timeout_minutes</code>を設定してランタイムを制限します。</p>
        `
    },
    {
        number: 50,
        domain: "Hyperopt and SparkTrials",
        question: "ハイパーパラメータチューニングにおいて、グリッドサーチやランダムサーチよりも、Hyperopt Tree of Parzen Estimators（TPE）アルゴリズムなどのベイズアプローチを使用する利点は何ですか？",
        keyPoint: "ベイズ最適化の利点",
        choices: [
            "ベイズアプローチはより高速だが精度が低い",
            "ベイズアプローチはより少ないハイパーパラメータを探索する",
            "ベイズアプローチは一般的により効率的で、より多くのハイパーパラメータとより広い範囲を探索できる",
            "ベイズアプローチは小規模データセットにのみ適している"
        ],
        correctIndex: 2,
        explanation: `
            <h3>正解: ベイズアプローチは一般的により効率的で、より多くのハイパーパラメータとより広い範囲を探索できる</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>ベイズ最適化（例：HyperoptのTPE）は、過去のトライアル結果に基づいて検索を適応させます：</p>
            <ul>
                <li><strong>有望な領域に焦点:</strong> パフォーマンスを向上させるハイパーパラメータにより多くのトライアルを割り当て</li>
                <li><strong>無駄な努力を回避:</strong> 性能の低い組み合わせをスキップ（グリッド/ランダムサーチとは異なる）</li>
            </ul>

            <h4>効率性</h4>
            <ul>
                <li>より少ないトライアルでより広い範囲を探索</li>
                <li>高次元空間に最適（例：ニューラルネットワーク）</li>
            </ul>

            <h5>例</h5>
            <pre><code class="language-python">
from hyperopt import fmin, tpe, hp

best = fmin(fn=objective, space=hp.uniform('lr', 0.001, 0.1), algo=tpe.suggest, max_evals=50)
            </code></pre>

            <h4>グリッド/ランダムサーチに対する主な利点</h4>
            <table border="1" cellpadding="5">
                <tr>
                    <th>方法</th>
                    <th>長所</th>
                    <th>短所</th>
                </tr>
                <tr>
                    <td>グリッドサーチ</td>
                    <td>網羅的</td>
                    <td>計算コストが高い</td>
                </tr>
                <tr>
                    <td>ランダムサーチ</td>
                    <td>高次元でグリッドより優れている</td>
                    <td>過去のトライアルから学習しない</td>
                </tr>
                <tr>
                    <td>ベイズ（TPE）</td>
                    <td>スマートサンプリング、より高速な収束</td>
                    <td>逐次トライアルが必要</td>
                </tr>
            </table>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>「より高速だが精度が低い」:</strong> ベイズ手法は適応的サンプリングにより、より高速かつより正確</li>
                <li><strong>「より少ないハイパーパラメータを探索」:</strong> より少なくではなく、より効率的に探索</li>
                <li><strong>「小規模データセットにのみ適している」:</strong> 大規模にスケール（例：SparkTrialsで）</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>ハイパーパラメータチューニングのために：</p>
            <ul>
                <li>効率が重要な場合はベイズ（TPE）を使用（大きな空間、コストのかかる評価）</li>
                <li>単純な問題のベースラインとしてランダムサーチを使用</li>
                <li>低次元空間を除き、グリッドサーチを避ける</li>
            </ul>

            <p><strong>プロのヒント:</strong> 最適な結果を得るために、TPEと早期停止（例：<code>max_evals=100</code>）を組み合わせます。</p>
        `
    },
    {
        number: 51,
        domain: "Databricks ML",
        question: "データサイエンティストがFeature Storeを使用しています。1つの特徴量テーブルで、各特徴量変数のメディアン値で欠損値を置き換えたいと考えています。同僚は、この方法では貴重な情報を捨てていると指摘しています。特徴量セットにできるだけ多くの情報を含めるために、どのアプローチを取ることができますか？",
        keyPoint: "欠損値補完時の情報保持",
        choices: [
            "欠損値を含む各特徴量に対して、各行の値が補完されたかどうかを示すバイナリ特徴量変数を作成する",
            "機械学習アルゴリズムに欠損値の処理方法を決定させるため、欠損値の補完を控える",
            "欠損値を含む各特徴量に対して、元々欠損していた特徴量の行の割合を示す定数特徴量変数を作成する",
            "メディアン値の代わりに各特徴量変数の平均値を使用して欠損値を補完する",
            "元々欠損値を含んでいたすべての特徴量変数を特徴量セットから削除する"
        ],
        correctIndex: 0,
        explanation: `
            <h3>正解: 欠損値を含む各特徴量に対して、各行の値が補完されたかどうかを示すバイナリ特徴量変数を作成する</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>バイナリインジケータ変数は、どの値が欠損していたかについての情報を保持します（欠損性に意味のあるパターンが存在することが多い）。</p>

            <h5>ワークフロー例</h5>
            <ol>
                <li>メディアンで欠損値を補完</li>
                <li>補完された行をマークするバイナリ列（例：<code>age_was_missing</code>）を追加</li>
            </ol>

            <pre><code class="language-python">
from pyspark.sql.functions import col, when, lit

median_age = df.approxQuantile("age", [0.5], 0.01)[0]  # メディアンを計算
df = df.withColumn("age_imputed", when(col("age").isNull(), median_age).otherwise(col("age")))
df = df.withColumn("age_was_missing", when(col("age").isNull(), 1).otherwise(0))
            </code></pre>

            <h4>利点</h4>
            <p>モデルは補完された値と欠損パターンの両方から学習します。</p>

            <h4>主なメリット</h4>
            <p>欠損データを堅牢に処理しながら、情報保持を最大化します。</p>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>「補完を控える」:</strong> ほとんどのMLアルゴリズムはnullを処理できない（例：Spark MLはエラーをスロー）</li>
                <li><strong>「欠損率の定数特徴量」:</strong> 行レベルのインジケータほど実用的でない（グローバル統計は行レベルの予測に役立たない）</li>
                <li><strong>「平均で補完」:</strong> メディアンと似ているが、コアの問題（欠損情報の喪失）に対処しない</li>
                <li><strong>「特徴量を削除」:</strong> 有用な可能性のあるデータを破棄</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>Feature Storeでの欠損データのために：</p>
            <ul>
                <li>補完（メディアン/平均）+ バイナリフラグを追加</li>
                <li><code>FeatureStoreClient.log_feature_stats()</code>を使用して欠損性を追跡</li>
            </ul>

            <p><strong>プロのヒント:</strong> 時系列特徴量の場合、フォワードフィル補完 + フラグを検討します。</p>
        `
    },
    {
        number: 52,
        domain: "ML Workflows",
        question: "過学習を減らす方法は？",
        keyPoint: "過学習を防ぐ技術",
        choices: [
            "早期停止（Early Stopping）- 勾配降下法などの反復法でモデルをトレーニングする際の正則化の形式",
            "データ拡張（Data Augmentation）- トレーニングデータのみの情報を使用してトレーニングデータの量を増やす。例：画像内の犬を見つけるための画像のスケーリング、回転",
            "正則化（Regularization）- モデルの複雑性を減らす技術",
            "ドロップアウト（Dropout）- 過学習を防ぐ正則化技術",
            "上記のすべて"
        ],
        correctIndex: 4,
        explanation: `
            <h3>正解: 上記のすべて</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>リストされたすべての技術は、過学習を減らすための実証済みの方法であり、それぞれ異なる角度から対処します：</p>

            <ul>
                <li><strong>早期停止:</strong> 検証パフォーマンスが低下したときにトレーニングを停止（記憶を防ぐ）</li>
                <li><strong>データ拡張:</strong> トレーニングデータの多様性を拡大（例：CNNのための画像回転）</li>
                <li><strong>正則化:</strong> モデルの複雑性にペナルティ（例：線形モデルのL1/L2、ニューラルネットワークのドロップアウト）</li>
                <li><strong>ドロップアウト:</strong> トレーニング中にニューロンをランダムに無効化して、堅牢な特徴学習を強制</li>
            </ul>

            <h5>例</h5>

            <h6>早期停止（TensorFlow/Keras）</h6>
            <pre><code class="language-python">
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=3)
model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[early_stop])
            </code></pre>

            <h6>ドロップアウト（PyTorch）</h6>
            <pre><code class="language-python">
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(100, 50),
    nn.Dropout(0.2),  # 20%ドロップアウト
    nn.ReLU()
)
            </code></pre>

            <h4>重要ポイント</h4>
            <p>過学習に対抗するために：</p>
            <ul>
                <li>これらの技術の組み合わせを使用</li>
                <li>検証メトリクスを注意深く監視</li>
                <li>正則化の選択はモデルに依存（例：ニューラルネットにはドロップアウト、線形モデルにはL2）</li>
            </ul>

            <p><strong>プロのヒント:</strong> 表形式データの場合、アンサンブル手法（例：Random Forests）は平均化により自然に過学習に抵抗します。</p>
        `
    },
    {
        number: 53,
        domain: "Spark ML Basics",
        question: "Spark MLワークフローで次元削減技術を使用する主な目標は何ですか？",
        keyPoint: "次元削減の目的",
        choices: [
            "モデルの複雑性を増加させる",
            "データセットに無関係な特徴量を追加する",
            "入力特徴量の数を削減する",
            "データ前処理を高速化する"
        ],
        correctIndex: 2,
        explanation: `
            <h3>正解: 入力特徴量の数を削減する</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>Spark MLでの次元削減（例：PCA、特徴量選択）は、以下を目的とします：</p>
            <ul>
                <li><strong>冗長/無関係な特徴量を排除:</strong> 最も情報量の多いものに焦点を当てる</li>
                <li><strong>計算コストを削減:</strong> より少ない特徴量により、トレーニングと推論が高速化</li>
                <li><strong>モデルパフォーマンスを向上:</strong> 「次元の呪い」を軽減（無関係な特徴量からのノイズ）</li>
            </ul>

            <h5>例（Spark MLでのPCA）</h5>
            <pre><code class="language-python">
from pyspark.ml.feature import PCA

pca = PCA(k=10, inputCol="features", outputCol="pca_features")
model = pca.fit(scaled_df)  # 100次元の特徴量 → 10次元に削減
            </code></pre>

            <h4>主なメリット</h4>
            <ul>
                <li><strong>より高速なトレーニング:</strong> 処理するデータが少ない</li>
                <li><strong>より良い汎化:</strong> ノイズを除去</li>
            </ul>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>「モデルの複雑性を増加」:</strong> 次元削減はモデルを簡素化します</li>
                <li><strong>「無関係な特徴量を追加」:</strong> 逆効果—削減は無関係性を除去します</li>
                <li><strong>「前処理を高速化」:</strong> 副次的効果であり、主な目標ではない</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>効率的なSpark MLワークフローのために：</p>
            <ul>
                <li>線形依存関係にはPCAを使用</li>
                <li>カテゴリカル特徴量にはChiSqSelectorを使用</li>
                <li>交差検証で最適な次元数を選択するために検証</li>
            </ul>

            <p><strong>プロのヒント:</strong> VectorSlicerと組み合わせて、重要度の低い特徴量を手動で削除します。</p>
        `
    },
    {
        number: 54,
        domain: "Spark ML Basics",
        question: "Databricksで、データエンジニアは毎日実行され、現在の日付をパラメータとして渡すノートブックをスケジュールする必要があります。どの機能を使用し、どのようにパラメータを渡すべきですか？",
        keyPoint: "Databricksでのパラメータ化されたノートブックのスケジューリング",
        choices: [
            "Databricks Jobsを%runコマンドで使用し、dbutils.widgets.get(\"date\")を使用して日付を渡す",
            "Azure Data Factoryでノートブックをスケジュールし、日付をパイプラインパラメータとして渡す",
            "Databricks Jobsを使用し、ジョブ設定で日付パラメータを指定する",
            "Cronジョブでノートブックをスケジュールし、Scalaを使用して現在の日付を取得する",
            "Apache Airflow DAGを実装してノートブックをスケジュールし、Airflowマクロを通じて日付を渡す"
        ],
        correctIndex: 2,
        explanation: `
            <h3>正解: Databricks Jobsを使用し、ジョブ設定で日付パラメータを指定する</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>Databricks Jobsは、パラメータ化されたノートブック実行をネイティブにサポートします。ステップは以下の通りです：</p>

            <h5>1. ノートブックで日付を受け入れるウィジェットを定義</h5>
            <pre><code class="language-python">
dbutils.widgets.text("date", "")  # テキスト入力ウィジェットを作成
current_date = dbutils.widgets.get("date")  # パラメータを読み取る
            </code></pre>

            <h5>2. ジョブをスケジュール</h5>
            <p>ジョブUIで、動的なタイムスタンプ（例：<code>{{ "{{" }} yyyy-MM-dd {{ "}}" }}</code>）を持つパラメータ<code>date</code>を追加します。</p>
            <p>スケジュールを毎日に設定します。</p>

            <h5>ジョブ設定例</h5>
            <ul>
                <li><strong>ノートブックパス:</strong> /Users/email@company/daily_etl</li>
                <li><strong>パラメータ:</strong> {"date": "{{ "{{" }} yyyy-MM-dd {{ "}}" }}"}</li>
                <li><strong>スケジュール:</strong> 0 0 0 * * ?（毎日深夜）</li>
            </ul>

            <h4>メリット</h4>
            <ul>
                <li>外部依存関係なし（例：Airflow/ADF）</li>
                <li>動的値を持つ組み込みのパラメータ化</li>
            </ul>

            <h4>他の選択肢があまり理想的でない理由</h4>
            <ul>
                <li><strong>%runとウィジェット:</strong> 手動でエラーが発生しやすい。スケジューリングには拡張性がない</li>
                <li><strong>Azure Data Factory/Airflow:</strong> シンプルな毎日のジョブには過剰。外部の複雑性を追加</li>
                <li><strong>Cronジョブ:</strong> Databricksパラメータ渡しとのネイティブ統合が欠如</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>Databricksでスケジュールされたパラメータ化ノートブックのために：</p>
            <ul>
                <li>シンプルさのためにJobs + ウィジェットを使用</li>
                <li>動的日付には<code>{{ "{{" }} yyyy-MM-dd {{ "}}" }}</code>を活用</li>
                <li>Job Runs UIを介して監視</li>
            </ul>

            <p><strong>プロのヒント:</strong> 複雑な依存関係の場合、代わりにDelta Live Tablesを使用します。</p>
        `
    },
    {
        number: 55,
        domain: "ML Workflows",
        question: "欠損値を処理するために平均補完が最も適切なのは、次のどのケースですか？",
        keyPoint: "平均補完が適切な条件",
        choices: [
            "データがランダムに欠損している場合（MAR）",
            "データがランダムでなく欠損している場合（MNAR）",
            "データが完全にランダムに欠損している場合（MCAR）",
            "データが体系的に欠損している場合"
        ],
        correctIndex: 2,
        explanation: `
            <h3>正解: データが完全にランダムに欠損している場合（MCAR）</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>平均補完は、欠損値を特徴量の平均で置き換えます。これは、データが<strong>完全にランダムに欠損（MCAR）</strong>している場合に最もバイアスが少なくなります。つまり：</p>
            <ul>
                <li>欠損性が、観測された変数または観測されていない変数との関係がない</li>
                <li>例：ランダムな停電によるセンサーデータのギャップ</li>
            </ul>

            <h5>適切な使用例</h5>
            <pre><code class="language-python">
from pyspark.ml.feature import Imputer

imputer = Imputer(strategy="mean", inputCols=["age"], outputCols=["age_imputed"])
imputer.fit(df).transform(df)
            </code></pre>

            <h4>主要な仮定</h4>
            <p>MCARは、平均が真の分布を代表し続けることを保証します。</p>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>ランダムに欠損（MAR）:</strong> モデルベースの補完を使用（例：回帰）。欠損性が観測データに依存するため</li>
                <li><strong>ランダムでなく欠損（MNAR）/体系的:</strong> 平均補完はバイアスを導入（例：高所得値が意図的に欠損している場合）</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>欠損データ処理のために：</p>
            <ul>
                <li><strong>MCAR:</strong> 平均/メディアン補完</li>
                <li><strong>MAR:</strong> 予測的補完（例：IterativeImputer）</li>
                <li><strong>MNAR:</strong> 根本原因を分析（例：欠損性インジケータを追加）</li>
            </ul>

            <p><strong>プロのヒント:</strong> バイアスをチェックするために、補完前後で常に分布を比較します。</p>
        `
    },
    {
        number: 56,
        domain: "Databricks ML",
        question: "機械学習エンジニアのチームが、データサイエンティストから3つのノートブック（ノートブックA、B、C）を受け取り、機械学習パイプラインを設定します。ノートブックAは探索的データ分析に使用され、ノートブックBとCは特徴量エンジニアリングに使用されます。ノートブックBとCが正常に実行されるには、ノートブックAが最初に完了する必要があります。ただし、ノートブックBとCは互いに独立して動作します。この設定において、Databricksを利用してこのパイプラインをオーケストレーションする最も効率的で信頼性の高い方法は何ですか？",
        keyPoint: "Databricksでのパイプラインオーケストレーション",
        choices: [
            "各タスクが特定のノートブックを実行する3タスクジョブを設定し、各タスクが前のタスクの完了に依存するようにする",
            "各タスクが異なるノートブックを操作する3タスクジョブを確立し、3つのタスクすべてを並列実行する",
            "各ジョブがユニークなノートブックを実行する3つの単一タスクジョブを作成し、すべて同時実行するようスケジュールする",
            "各タスクが特定のノートブックを操作する3タスクジョブを配置する。最後の2つのタスクは同時に実行されるように設定され、それぞれが最初のタスクの完了に依存する"
        ],
        correctIndex: 3,
        explanation: `
            <h3>正解: 各タスクが特定のノートブックを操作する3タスクジョブを配置する。最後の2つのタスクは同時に実行されるように設定され、それぞれが最初のタスクの完了に依存する</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p><strong>依存関係:</strong> ノートブックBとCはノートブックAに依存するが、互いには独立して実行されます。</p>

            <h5>最適なセットアップ</h5>
            <ul>
                <li><strong>タスク1:</strong> ノートブックA（EDA）を実行</li>
                <li><strong>タスク2 & 3:</strong> タスク1が成功した後、ノートブックBとCを並列実行</li>
            </ul>

            <h5>Databricks Jobsの設定</h5>
            <ul>
                <li>タスク2と3の依存関係としてタスク1を設定</li>
                <li>タスク2と3の間には依存関係なし</li>
            </ul>

            <h4>効率性</h4>
            <p>BとCの並列実行により時間を節約します。</p>

            <h5>ジョブセットアップ例</h5>
            <pre><code class="language-python">
# Databricks Jobs UIの疑似コード
Job:
  - Task 1: Notebook A (EDA)
  - Task 2: Notebook B (Feature Engineering) → Depends on Task 1
  - Task 3: Notebook C (Feature Engineering) → Depends on Task 1
            </code></pre>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>「すべてのタスクを順次」:</strong> 非効率的（BとCの間に並列性がない）</li>
                <li><strong>「すべてのタスクを並列」:</strong> Aが終了する前にB/Cが実行されるリスク</li>
                <li><strong>「3つの個別ジョブ」:</strong> 明示的な依存関係管理が欠如。監視が困難</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>Databricksでの効率的なパイプラインオーケストレーションのために：</p>
            <ul>
                <li>依存関係を持つマルチタスクジョブを使用</li>
                <li>可能な限り独立したタスク（B/C）を並列化</li>
                <li>実行ステータスをJobs UIで監視</li>
            </ul>

            <p><strong>プロのヒント:</strong> タスク1が失敗した場合（B/Cをブロック）に通知するアラートを追加します。</p>
        `
    },
    {
        number: 57,
        domain: "Spark ML",
        question: "分散コンピューティング環境で、データエンジニアがApache Spark Structured Streamingを使用してストリーミングデータパイプラインを実装します。パイプラインには、キーごとのイベントの実行カウントを追跡するステートフル操作が含まれます。このステートフル操作の最適なパフォーマンスとフォールトトレランスを保証する設定はどれですか？",
        keyPoint: "Structured Streamingでのステートフル操作の設定",
        choices: [
            "フォールトトレランスを保証するためにHDFSへのチェックポイントを設定する",
            "並列処理を最大化するためにシャッフルパーティション数を増やす",
            "より高速なアクセスのためにローカルメモリに状態を保存するステートフル操作を使用する",
            "ストリーミングアプリケーションのスループットを増加させるために先行書き込みログを無効化する",
            "状態更新中のシャッフルを削減するために、結合テーブルをすべてのエグゼキュータにブロードキャストする"
        ],
        correctIndex: 0,
        explanation: `
            <h3>正解: フォールトトレランスを保証するためにHDFSへのチェックポイントを設定する</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>Spark Structured Streamingでのチェックポイントは、ステートフル操作（例：<code>mapGroupsWithState</code>、<code>flatMapGroupsWithState</code>）に不可欠です。なぜなら：</p>
            <ul>
                <li><strong>フォールトトレランス:</strong> 状態とメタデータをHDFS/S3に保存（障害後に回復可能）</li>
                <li><strong>Exactly-onceセマンティクス:</strong> 正確な状態復元を保証</li>
            </ul>

            <h5>例</h5>
            <pre><code class="language-python">
query = (
    df.writeStream
    .format("delta")
    .outputMode("update")
    .option("checkpointLocation", "/path/to/checkpoint")  # ステートフル操作に必須
    .start()
)
            </code></pre>

            <h4>主なメリット</h4>
            <ul>
                <li><strong>状態復旧:</strong> クラッシュ後にカウントを復元</li>
                <li><strong>パフォーマンス:</strong> メモリから永続ストレージへ状態ストレージをオフロード</li>
            </ul>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>「シャッフルパーティションを増やす」:</strong> 並列処理には役立つが、状態復旧には対処しない</li>
                <li><strong>「ローカルメモリ状態」:</strong> 揮発性。エグゼキュータ障害時に失われる</li>
                <li><strong>「先行書き込みログを無効化」:</strong> リスクが高い—フォールトトレランスを失う（WALは状態変更をログ）</li>
                <li><strong>「結合テーブルをブロードキャスト」:</strong> ステートフルストリーミングには無関係（状態はキーごとで、結合ではない）</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>ステートフルストリーミングパイプラインのために：</p>
            <ul>
                <li>HDFS/S3へのチェックポイントを有効化</li>
                <li>並列処理のために<code>spark.sql.shuffle.partitions</code>をチューニング</li>
                <li>状態ストアメトリクス（例：<code>numUpdatedStateRows</code>）を監視</li>
            </ul>

            <p><strong>プロのヒント:</strong> ACID保証を活用するために、チェックポイントにDelta Lakeを使用します。</p>
        `
    },
    {
        number: 58,
        domain: "Pandas API on Spark",
        question: "Pandas API on Sparkのどのオプションが、ショートカットの制限を設定し、そのスキーマを使用して指定された行数を計算しますか？",
        keyPoint: "pandas-on-Sparkでのショートカット制限設定",
        choices: [
            "display.max_rows",
            "compute.ops_on_diff_frames",
            "compute.shortcut_limit",
            "compute.default_index_type"
        ],
        correctIndex: 2,
        explanation: `
            <h3>正解: compute.shortcut_limit</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>pandas API on Sparkの<code>compute.shortcut_limit</code>は、スキーマを推論したり迅速なチェックを実行するために、ローカル（ドライバー上）で何行計算するかを決定し、完全な分散計算を回避します。</p>
            <ul>
                <li><strong>デフォルト:</strong> 通常1000行</li>
                <li><strong>用途:</strong> <code>df.head()</code>やスキーマ推論などの操作をサンプリングにより高速化</li>
            </ul>

            <h5>例</h5>
            <pre><code class="language-python">
import pyspark.pandas as ps

ps.set_option("compute.shortcut_limit", 500)  # スキーマチェックのために500行のみ計算
            </code></pre>

            <h4>主な影響</h4>
            <ul>
                <li><strong>パフォーマンス:</strong> メタデータ操作のレイテンシを削減</li>
                <li><strong>トレードオフ:</strong> 低すぎる → スキーマ推論エラー。高すぎる → ドライバー操作が遅くなる</li>
            </ul>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>display.max_rows:</strong> 表示/印刷する行数を制御（例：<code>df.display()</code>）</li>
                <li><strong>compute.ops_on_diff_frames:</strong> 無関係なDataFrames間の操作を許可（サンプリングとは無関係）</li>
                <li><strong>compute.default_index_type:</strong> インデックスタイプを設定（例：「distributed」vs「sequence」）</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>pandas-on-Sparkでの効率的なスキーマチェックのために：</p>
            <ul>
                <li>データサイズに基づいて<code>compute.shortcut_limit</code>を調整</li>
                <li>サンプリングなしの完全なスキーマには<code>df.spark.schema()</code>を使用</li>
            </ul>

            <p><strong>プロのヒント:</strong> 大規模DataFramesでスキーマの不一致に遭遇した場合、この値を増やします。</p>
        `
    },
    {
        number: 59,
        domain: "Pandas API on Spark",
        question: "pandas-on-Sparkで作業しているPySparkユーザーが完全なPySpark APIにアクセスできるメソッドは何ですか？",
        keyPoint: "pandas-on-SparkからPySparkへの変換",
        choices: [
            "DataFrame.to_pandas()",
            "DataFrame.to_spark()",
            "ps.to_spark()",
            "ps.pandas_api()"
        ],
        correctIndex: 1,
        explanation: `
            <h3>正解: DataFrame.to_spark()</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p><code>to_spark()</code>メソッドは、pandas-on-Spark DataFrameをネイティブPySpark DataFrameに変換し直し、完全なPySpark API（例：RDD操作、Spark SQL）へのアクセスを可能にします。</p>

            <h5>例</h5>
            <pre><code class="language-python">
import pyspark.pandas as ps

ps_df = ps.DataFrame({"A": [1, 2], "B": [3, 4]})
spark_df = ps_df.to_spark()  # PySpark DataFrameに変換
spark_df.rdd.map(...)        # これでPySpark APIを使用可能
            </code></pre>

            <h4>主な用途</h4>
            <ul>
                <li><strong>高度な変換:</strong> PySparkの低レベルAPI（例：<code>rdd</code>、<code>join</code>）を使用</li>
                <li><strong>統合:</strong> Spark MLlib、Structured Streamingなどと互換性</li>
            </ul>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>to_pandas():</strong> pandas DataFrame（シングルノード、Sparkの並列性を失う）に変換</li>
                <li><strong>ps.to_spark():</strong> 存在しない—正しい構文は<code>DataFrame.to_spark()</code></li>
                <li><strong>ps.pandas_api():</strong> 存在しない。pandas-on-Spark APIは<code>ps</code>経由で直接アクセス</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>pandas-on-SparkとPySparkのシームレスな移行のために：</p>
            <ul>
                <li>PySparkの完全な機能には<code>to_spark()</code>を使用</li>
                <li>小規模なローカル分析のみに<code>to_pandas()</code>を使用</li>
            </ul>

            <p><strong>プロのヒント:</strong> 操作を効率的にチェーン化：</p>
            <pre><code class="language-python">
ps_df = ps.DataFrame(...).to_spark().groupBy(...).count().to_pandas_on_spark()
            </code></pre>
        `
    },
    {
        number: 60,
        domain: "Spark ML Algorithms",
        question: "機械学習プロジェクトで推薦システムを構築しており、協調フィルタリング技術が必要です。Databricks MLlibがサポートする、協調フィルタリングタスクに適したアルゴリズムはどれですか？",
        keyPoint: "協調フィルタリングアルゴリズム",
        choices: [
            "決定木（Decision Trees）",
            "k平均クラスタリング（k-Means Clustering）",
            "行列分解（Matrix Factorization）",
            "ナイーブベイズ（Naive Bayes）"
        ],
        correctIndex: 2,
        explanation: `
            <h3>正解: 行列分解（Matrix Factorization）</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>行列分解（例：ALS - Alternating Least Squares）は、Spark MLlibでの協調フィルタリングの標準アルゴリズムです。これにより：</p>
            <ul>
                <li>ユーザー-アイテム相互作用マトリックスを潜在因子（ユーザーとアイテムの埋め込み）に分解</li>
                <li>これらの因子のドット積を介して欠損エントリ（例：評価）を予測</li>
            </ul>

            <h5>例</h5>
            <pre><code class="language-python">
from pyspark.ml.recommendation import ALS

als = ALS(
    userCol="user_id",
    itemCol="item_id",
    ratingCol="rating",
    coldStartStrategy="drop"
)
model = als.fit(train_df)
            </code></pre>

            <h4>主な機能</h4>
            <ul>
                <li><strong>暗黙的/明示的フィードバック:</strong> 両方をサポート（<code>implicitPrefs=True</code>経由）</li>
                <li><strong>スケーラビリティ:</strong> 分散計算用に最適化</li>
            </ul>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>決定木/ナイーブベイズ:</strong> ユーザー-アイテム相互作用モデリング用に設計されていない</li>
                <li><strong>k平均:</strong> クラスタリングアルゴリズム（教師なし）で、パーソナライズされた推薦には不向き</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>Spark MLlibでの協調フィルタリングのために：</p>
            <ul>
                <li>行列分解にはALSを使用</li>
                <li><code>rank</code>（潜在因子）、<code>regParam</code>（正則化）をチューニング</li>
                <li>サービング用にMLflowでデプロイ</li>
            </ul>

            <p><strong>プロのヒント:</strong> コールドスタート問題の場合、コンテンツベースフィルタリングとブレンドします。</p>
        `
    },
    {
        number: 61,
        domain: "Databricks ML",
        question: "データエンジニアリングチームは、外部データベースからデータを読み取り、「external_data」という名前のDatabricks Deltaテーブルに書き込むタスクを担当しています。Deltaテーブルの既存データが上書きされないようにしたいと考えています。どのコードスニペットを使用すべきですか？",
        keyPoint: "Deltaテーブルへのデータ追加（上書き回避）",
        choices: [
            "external_data.write.format(\"delta\").mode(\"overwrite\").saveAsTable(\"external_data\")",
            "external_data.write.format(\"delta\").mode(\"append\").saveAsTable(\"external_data\")",
            "external_data.write.format(\"delta\").mode(\"ignore\").saveAsTable(\"external_data\")",
            "spark.sql(\"INSERT INTO external_data SELECT * FROM external_data_source\")"
        ],
        correctIndex: 1,
        explanation: `
            <h3>正解: external_data.write.format("delta").mode("append").saveAsTable("external_data")</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p><code>mode("append")</code>は、既存のDeltaテーブルの以前のレコードを削除せずに新しいデータが追加されることを保証します。</p>
            <ul>
                <li><strong>冪等性:</strong> 再実行に対して安全（データがユニークであれば重複なし）</li>
            </ul>

            <h5>例</h5>
            <pre><code class="language-python">
(external_data.write
    .format("delta")
    .mode("append")  # キーパラメータ
    .saveAsTable("external_data")
)
            </code></pre>

            <h4>主なメリット</h4>
            <ul>
                <li><strong>履歴の保持:</strong> Delta Lakeのタイムトラベルがすべてのバージョンを保持</li>
                <li><strong>ACIDコンプライアンス:</strong> データ整合性を保証</li>
            </ul>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>overwrite:</strong> 既存データを削除（要件に違反）</li>
                <li><strong>ignore:</strong> テーブルが存在する場合、書き込みをスキップ（データが追加されない）</li>
                <li><strong>SQL INSERT INTO:</strong> 機能するが、バッチ書き込みに対して<code>append</code>より明示的でない</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>Delta Lakeへの増分書き込みのために：</p>
            <ul>
                <li>既存データを保持するために<code>append</code>を使用</li>
                <li>重複排除には<code>MERGE</code>ロジックを追加</li>
            </ul>

            <p><strong>プロのヒント:</strong> 成長するテーブルでクエリパフォーマンスを維持するために、<code>OPTIMIZE</code>と<code>ZORDER</code>を使用します。</p>
        `
    },
    {
        number: 62,
        domain: "Spark ML Algorithms",
        question: "機械学習プロジェクトが、予測タスクのための時系列データの処理を含んでいます。Databricks MLlibがサポートする、時系列予測に適したアルゴリズムはどれですか？",
        keyPoint: "時系列予測アルゴリズム",
        choices: [
            "線形回帰（Linear Regression）",
            "決定木（Decision Trees）",
            "長短期記憶（LSTM - Long Short-Term Memory）",
            "サポートベクターマシン（Support Vector Machines）"
        ],
        correctIndex: 2,
        explanation: `
            <h3>正解: 長短期記憶（LSTM - Long Short-Term Memory）</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>LSTM（再帰型ニューラルネットワークの一種）は、時系列予測に理想的です。なぜなら：</p>
            <ul>
                <li><strong>時間的依存関係をキャプチャ:</strong> シーケンシャルデータの長期パターンを記憶</li>
                <li><strong>可変長シーケンスを処理:</strong> 不規則な時間ステップに適応</li>
            </ul>

            <h5>DatabricksでのサポートはTensorFlow/PyTorch統合経由</h5>
            <pre><code class="language-python">
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([LSTM(50, input_shape=(n_steps, n_features)), Dense(1)])
            </code></pre>

            <h5>Spark MLlibの代替</h5>
            <p>古典的アプローチにはARIMAまたはProphetを使用します。</p>

            <h4>主な機能</h4>
            <ul>
                <li><strong>シーケンシャルモデリング:</strong> 時間ステップを反復的に処理</li>
                <li><strong>非線形パターン:</strong> 複雑なトレンド/季節性を学習</li>
            </ul>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>線形回帰/決定木/SVM:</strong> 時間的順序を無視。時系列を独立した点として扱う</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>Databricksでの時系列予測のために：</p>
            <ul>
                <li>ディープラーニングにはLSTMを使用（GPU加速）</li>
                <li>表形式の時系列には、<code>mlflow.prophet</code>またはSpark MLのARIMAを試す</li>
            </ul>

            <p><strong>プロのヒント:</strong> LSTMパフォーマンスを向上させるために、ラグ特徴量（<code>pyspark.sql.window</code>）で前処理します。</p>
        `
    },
    {
        number: 63,
        domain: "Databricks ML",
        question: "機械学習チームが、Databricks MLパイプラインでカスタムPyTorchモデルを使用する必要があるプロジェクトに取り組んでいます。PyTorchライブラリとカスタムモデルがワークスペース内のすべてのノートブックでトレーニングに使用できるようにしたいと考えています。推奨されるアプローチは何ですか？",
        keyPoint: "クラスター全体でのPyTorchライブラリの利用可能性確保",
        choices: [
            "Databricks Runtime for MLflowを使用するようにクラスターを編集する",
            "クラスター設定でMLFLOW_PYTORCH_VERSION変数を設定する",
            "クラスターに接続された任意のノートブックで一度%pip install torchを実行する",
            "torchとカスタムモデルをクラスターのライブラリ依存関係に追加する"
        ],
        correctIndex: 3,
        explanation: `
            <h3>正解: torchとカスタムモデルをクラスターのライブラリ依存関係に追加する</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>クラスターライブラリは、クラスター上のすべてのノートブックが同じPyTorchバージョンとカスタムモデルコードにアクセスできることを保証します。</p>

            <h5>ステップ</h5>
            <h6>1. PyTorchをインストール</h6>
            <p>クラスター → ライブラリ → 新規インストール → PyPI → <code>torch</code>を入力</p>

            <h6>2. カスタムモデルをアップロード</h6>
            <p>モデルコードを<code>whl</code>/<code>egg</code>ファイルまたはGitリポジトリとして追加</p>

            <h4>メリット</h4>
            <ul>
                <li><strong>一貫性:</strong> すべてのユーザーが同じ環境を取得</li>
                <li><strong>再現性:</strong> ライブラリがセッション間で持続</li>
            </ul>

            <h5>ワークフロー例</h5>
            <pre><code class="language-python">
# クラスターライブラリにtorchを追加後、任意のノートブックで使用：
import torch
from custom_model import MyModel  # ライブラリとしてアップロード
model = MyModel()
            </code></pre>

            <h4>他の選択肢があまり理想的でない理由</h4>
            <ul>
                <li><strong>Databricks Runtime for MLflow:</strong> MLflowを含むが、PyTorchの利用可能性を保証しない</li>
                <li><strong>MLFLOW_PYTORCH_VERSION:</strong> 有効な設定ではない。MLflowはバージョンを追跡するが、インストールはしない</li>
                <li><strong>%pip install torch:</strong> 一時的。ノートブック/クラスター再起動ごとに再実行が必要</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>チーム全体のPyTorchプロジェクトのために：</p>
            <ul>
                <li>コア依存関係（PyTorch）にはクラスターライブラリ</li>
                <li>カスタムコード/スクリプトにはInitスクリプト</li>
                <li>モデル追跡にはMLflow</li>
            </ul>

            <p><strong>プロのヒント:</strong> カスタムモデルセットアップを自動化するために、クラスタースコープのinitスクリプトを使用します。</p>
        `
    },
    {
        number: 64,
        domain: "Databricks ML",
        question: "機械学習チームが、numpyライブラリの特定のバージョンを必要とするプロジェクトに取り組んでいます。このバージョンがDatabricksワークスペース内のすべてのノートブックで使用されることを保証したいと考えています。推奨されるアプローチは何ですか？",
        keyPoint: "クラスター全体での特定numpyバージョンの強制",
        choices: [
            "Databricks Runtime for Machine Learningを使用するようにクラスターを編集する",
            "クラスター設定でPYTHON_VERSION変数を必要なバージョンに設定する",
            "クラスターに接続された任意のノートブックで一度%pip install numpy==<desired_version>を実行する",
            "numpy==<desired_version>をクラスターのライブラリ依存関係に追加する",
            "クラスターでnumpyの特定バージョンを強制する方法はない"
        ],
        correctIndex: 3,
        explanation: `
            <h3>正解: numpy==&lt;desired_version&gt;をクラスターのライブラリ依存関係に追加する</h3>

            <h4>解説</h4>
            <p><strong>なぜこれが正解か？</strong></p>
            <p>クラスターライブラリは、指定されたnumpyバージョンがクラスターに接続されたすべてのノートブックで一貫して利用可能であることを保証します。</p>

            <h5>ステップ</h5>
            <ol>
                <li>クラスター → ライブラリ → 新規インストール → PyPIに移動</li>
                <li><code>numpy==&lt;desired_version&gt;</code>を入力（例：<code>numpy==1.21.0</code>）</li>
            </ol>

            <h4>メリット</h4>
            <ul>
                <li><strong>再現性:</strong> すべてのユーザーが同じnumpyバージョンを取得</li>
                <li><strong>永続性:</strong> クラスター再起動後も残存</li>
            </ul>

            <h5>例</h5>
            <pre><code class="language-python">
# クラスターライブラリに追加後、任意のノートブックで使用：
import numpy as np
print(np.__version__)  # 指定されたバージョンであることが保証
            </code></pre>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>Databricks Runtime for ML:</strong> numpyを含むが、バージョン管理を保証しない</li>
                <li><strong>PYTHON_VERSION:</strong> Pythonバージョンを設定するが、ライブラリバージョンではない</li>
                <li><strong>ノートブックで%pip install:</strong> 一時的。セッションごとに再実行が必要</li>
                <li><strong>「強制する方法はない」:</strong> 誤り—クラスターライブラリがバージョンを強制</li>
            </ul>

            <h4>重要ポイント</h4>
            <p>チーム全体の依存関係管理のために：</p>
            <ul>
                <li>コアパッケージ（numpy、pandas）にはクラスターライブラリを使用</li>
                <li>アドホックなニーズには<code>%pip install</code>で補完（ただし、共有ノートブックに文書化）</li>
            </ul>

            <p><strong>プロのヒント:</strong> 複雑な環境セットアップの場合、initスクリプトと組み合わせます。</p>
        `
    },
    {
        number: 65,
        domain: "Pandas API on Spark",
        question: "大規模なデータセットで作業しており、Column.isin(list)によるフィルタリングの効率を向上させたいと考えています。どのオプションを調整することを検討すべきですか？",
        keyPoint: "isin()フィルタリングパフォーマンスの最適化",
        choices: [
            "compute.default_index_type",
            "compute.isin_limit",
            "compute.ordered_head",
            "compute.default_index_cache"
        ],
        correctIndex: 1,
        explanation: `
            <h3>正解: compute.isin_limit</h3>

            <h4>解説</h4>
            <p>pandas-on-Spark（<code>pyspark.pandas</code>）では、<code>isin()</code>関数は、列の値が指定されたリスト内に存在するかどうかに基づいてDataFrameをフィルタリングするために使用されます。デフォルトでは、pandas-on-Sparkはブロードキャスト結合を使用してこの操作を最適化しますが、リストが大きすぎるとパフォーマンスが低下する可能性があります。</p>

            <p><code>compute.isin_limit</code>オプションは、<code>isin()</code>がブロードキャスト結合から標準結合に切り替える大規模データセットのしきい値を制御します。</p>

            <ul>
                <li><code>compute.isin_limit</code>が低すぎる → Sparkは標準結合にデフォルト設定される可能性があり、小さなリストでは遅くなる</li>
                <li><code>compute.isin_limit</code>が高すぎる → ブロードキャスト結合が失敗し、計算時間が増加する可能性がある</li>
            </ul>

            <p><code>compute.isin_limit</code>を調整することで、<code>Column.isin(list)</code>フィルタリングのパフォーマンスを最適化できます。</p>

            <h5>例：大規模リストのためのcompute.isin_limitの調整</h5>
            <pre><code class="language-python">
import pyspark.pandas as ps

# isin()最適化のしきい値を調整
ps.set_option("compute.isin_limit", 5000)

# pandas-on-Spark DataFrameのサンプル
df = ps.DataFrame({"id": range(1, 100000)})

# フィルタリングする大規模リスト
large_list = list(range(1, 6000))

# 最適化されたフィルタリング
filtered_df = df[df["id"].isin(large_list)]
            </code></pre>

            <p><code>compute.isin_limit = 5000</code>を設定すると、リストに最大5000要素が含まれる場合の効率的なフィルタリングが保証されます。</p>
            <p>リストが制限を超える場合、Sparkは自動的により拡張性の高い結合ベースのアプローチに切り替わります。</p>

            <h4>他の選択肢が誤りである理由</h4>
            <ul>
                <li><strong>compute.default_index_type:</strong> インデックスメカニズムを制御（例：分散インデックス vs シーケンスインデックス）するが、フィルタリングパフォーマンスには影響しない</li>
                <li><strong>compute.ordered_head:</strong> DataFramesのプレビューで順序付けされた結果を保証するために<code>head()</code>操作の順序付けに影響するが、<code>isin()</code>フィルタリングとは無関係</li>
                <li><strong>compute.default_index_cache:</strong> インデックスベースの操作でのパフォーマンス向上のためにpandas-on-Sparkがインデックス列をキャッシュするかどうかを制御するが、<code>Column.isin(list)</code>最適化とは無関係</li>
            </ul>

            <h4>重要ポイント</h4>
            <p><code>Column.isin(list)</code>フィルタリング効率を向上させるには、以下を調整します：</p>
            <pre><code class="language-python">
ps.set_option("compute.isin_limit", &lt;new_threshold&gt;)
            </code></pre>
            <p>これにより、大規模リストに対してSparkがブロードキャスト結合から標準結合に切り替わるタイミングを制御し、パフォーマンスを最適化します。</p>
        `
    }
];

// グローバル変数として公開
if (typeof window !== 'undefined') {
    window.questions3 = questions3;
}
