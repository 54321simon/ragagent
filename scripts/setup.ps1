$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $repo
$env:UV_CACHE_DIR = Join-Path $root 'cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $root 'runtime\python'
$env:TEMP = Join-Path $root 'tmp'
$env:TMP = $env:TEMP
$env:OLLAMA_MODELS = Join-Path $root 'models\ollama'
$env:OLLAMA_HOST = '127.0.0.1:11434'
New-Item -ItemType Directory -Path $env:TEMP,(Join-Path $root 'runtime') -Force | Out-Null
$uvExe = Join-Path $root 'runtime\uv\uv.exe'
if (!(Test-Path -LiteralPath $uvExe)) {
    $release = Invoke-RestMethod 'https://api.github.com/repos/astral-sh/uv/releases/latest'
    $asset = $release.assets | Where-Object name -eq 'uv-x86_64-pc-windows-msvc.zip'
    $archive = Join-Path $root 'runtime\uv.zip'
    Invoke-WebRequest $asset.browser_download_url -OutFile $archive
    Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $root 'runtime\uv') -Force
}
Set-Location -LiteralPath $repo
& $uvExe python install 3.12
if ($LASTEXITCODE -ne 0) { throw 'Python installation failed' }
& $uvExe sync --locked --python 3.12
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
if (!(Test-Path -LiteralPath '.env')) { Copy-Item -LiteralPath '.env.example' -Destination '.env' }
$ollamaExe = Join-Path $root 'runtime\ollama\ollama.exe'
if (!(Test-Path -LiteralPath $ollamaExe)) {
    $release = Invoke-RestMethod 'https://api.github.com/repos/ollama/ollama/releases/latest'
    $asset = $release.assets | Where-Object name -eq 'ollama-windows-amd64.zip'
    $archive = Join-Path $root 'runtime\ollama-windows-amd64.zip'
    Invoke-WebRequest $asset.browser_download_url -OutFile $archive
    Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $root 'runtime\ollama') -Force
}
New-Item -ItemType Directory -Path 'data\logs' -Force | Out-Null
try { $null = Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2 }
catch {
    Start-Process -FilePath $ollamaExe -ArgumentList 'serve' -WindowStyle Hidden -RedirectStandardOutput 'data\logs\ollama.out.log' -RedirectStandardError 'data\logs\ollama.err.log'
    Start-Sleep -Seconds 3
}
& $ollamaExe pull qwen2.5:3b
if ($LASTEXITCODE -ne 0) { throw 'Qwen model download failed' }
& $ollamaExe pull bge-m3:567m
if ($LASTEXITCODE -ne 0) { throw 'Embedding model download failed' }
Write-Host 'Ready. Double click start_app.cmd.'
