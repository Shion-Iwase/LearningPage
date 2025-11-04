# DBMLA 試験問題 - ビルドスクリプト
# 分割されたファイルを1つのHTMLファイルに結合します

$outputFile = "../DBMLA_pre_exam1.html"
$cssFile = "exam_style.css"
$dataFile = "exam_data.js"
$appFile = "exam_app.js"
$indexFile = "index.html"

Write-Host "ビルド開始..." -ForegroundColor Green

# ファイルの存在確認
$files = @($cssFile, $dataFile, $appFile, $indexFile)
foreach ($file in $files) {
    if (-not (Test-Path $file)) {
        Write-Host "エラー: $file が見つかりません" -ForegroundColor Red
        exit 1
    }
}

# CSSファイルを読み込み
$css = Get-Content $cssFile -Raw

# JavaScriptファイルを読み込み
$dataJs = Get-Content $dataFile -Raw
$appJs = Get-Content $appFile -Raw

# HTMLファイルを読み込み
$html = Get-Content $indexFile -Raw

# CSSリンクをインライン化
$html = $html -replace '<link rel="stylesheet" href="exam_style.css">', "<style>`n$css`n    </style>"

# JavaScriptファイルをインライン化
$html = $html -replace '<script src="exam_data.js"></script>', "<script>`n$dataJs`n    </script>"
$html = $html -replace '<script src="exam_app.js"></script>', "<script>`n$appJs`n    </script>"

# 出力ファイルに書き込み
$html | Out-File -FilePath $outputFile -Encoding UTF8

Write-Host "ビルド完了!" -ForegroundColor Green
Write-Host "出力ファイル: $outputFile" -ForegroundColor Cyan
Write-Host ""
Write-Host "配布用ファイルが生成されました。" -ForegroundColor Yellow
Write-Host "このファイルをSharePoint/OneDriveで共有できます。" -ForegroundColor Yellow
