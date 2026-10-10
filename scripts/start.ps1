$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $repo
Set-Location -LiteralPath $repo
$env:PYTHONIOENCODING = 'utf-8'
$env:TEMP = Join-Path $root 'tmp'
$env:TMP = $env:TEMP
$env:OLLAMA_MODELS = Join-Path $root 'models\ollama'
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '2'
$env:HF_HOME = Join-Path $root 'models\huggingface'
New-Item -ItemType Directory -Path $env:TEMP,'data\logs' -Force | Out-Null
$ollamaExe = Join-Path $root 'runtime\ollama\ollama.exe'
$pythonExe = Join-Path $repo '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $pythonExe)) { throw 'Please run scripts/setup.ps1 first.' }
try { $null = Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2 }
catch {
    if (!(Test-Path -LiteralPath $ollamaExe)) { throw 'Ollama is missing. Please run scripts/setup.ps1.' }
    Start-Process -FilePath $ollamaExe -ArgumentList 'serve' -WindowStyle Hidden -RedirectStandardOutput 'data\logs\ollama.out.log' -RedirectStandardError 'data\logs\ollama.err.log'
    for ($attempt=0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Seconds 1
        try { $null = Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2; break } catch {}
    }
}
$running = $false
try { $r = Invoke-WebRequest 'http://127.0.0.1:8501/_stcore/health' -TimeoutSec 2; $running = $r.StatusCode -eq 200 } catch {}
if (!$running) {
    Start-Process -FilePath $pythonExe -ArgumentList '-m','streamlit','run','ui/app.py','--server.address=127.0.0.1','--server.port=8501','--server.headless=true','--browser.gatherUsageStats=false' -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput 'data\logs\streamlit.out.log' -RedirectStandardError 'data\logs\streamlit.err.log'
}
Write-Host 'Research assistant: http://127.0.0.1:8501'
