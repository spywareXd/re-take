# Start the retake service and the Retake Editor editor together.
#   .\run_dev.ps1              real service (LLM keys from .env, Groq by default), editor connected to it
#   .\run_dev.ps1 -Llm gemini  use another provider (groq | gemini | openai | auto)
#   .\run_dev.ps1 -MockLlm     "Ask AI" answers from a mock, no key or network
#   .\run_dev.ps1 -Release     run the optimised editor build (slower to build, smoother to use)
#   .\run_dev.ps1 -ServiceOnly just the service (open the editor yourself with RETAKE_EDITOR_SERVICE_URL=1)
# Closing the editor stops the service.
param(
    [ValidateSet('groq', 'gemini', 'openai', 'auto')][string]$Llm = 'groq',   # which LLM answers "Ask AI" (auto = first key found: gemini, groq, openai)
    [switch]$MockLlm,
    [switch]$Release,
    [switch]$ServiceOnly,
    [int]$Port = 8765
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "No Python environment at $python. Run 'uv sync' in $root first." }
$env:PATH += ";$env:USERPROFILE\.cargo\bin"
$url = "http://127.0.0.1:$Port"

$running = $false
try { Invoke-RestMethod "$url/v1/capabilities" -TimeoutSec 2 | Out-Null; $running = $true } catch { }

$service = $null
if ($running) {
    Write-Host "A retake service is already answering at $url; using it (it keeps whatever LLM setting it was started with; stop it to change that)."
} else {
    # set or clear it explicitly: an earlier -MockLlm run leaves the variable in this terminal session
    if ($MockLlm) { $env:RETAKE_LLM_MOCK = '1' } else { Remove-Item Env:RETAKE_LLM_MOCK -ErrorAction SilentlyContinue }
    if ($Llm -eq 'auto') { Remove-Item Env:RETAKE_LLM_PROVIDER -ErrorAction SilentlyContinue } else { $env:RETAKE_LLM_PROVIDER = $Llm }
    $env:RETAKE_PORT = "$Port"
    Write-Host "Starting the retake service on $url ..."
    $service = Start-Process -FilePath $python -ArgumentList '-m', 'service' -WorkingDirectory $root -PassThru -NoNewWindow
    $deadline = (Get-Date).AddSeconds(180)
    $up = $false
    while ((Get-Date) -lt $deadline -and -not $service.HasExited) {
        try { Invoke-RestMethod "$url/v1/capabilities" -TimeoutSec 2 | Out-Null; $up = $true; break } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $up) {
        if (-not $service.HasExited) { Stop-Process -Id $service.Id -Force }
        throw "The retake service did not start. Run '$python -m service' to see why."
    }
}

try {
    $caps = Invoke-RestMethod "$url/v1/capabilities"
    Write-Host ("Service ready: adjudicate={0}, refine={1}" -f $caps.adjudicate, $caps.refine)
    Write-Host "Service log (full errors): $(Join-Path $root 'service\data\logs\service.log')"
    Write-Host "  last lines:  Get-Content '$(Join-Path $root 'service\data\logs\service.log')' -Tail 60"
    if ($ServiceOnly) {
        Write-Host "Service only. Press Ctrl+C to stop."
        if ($service) { $service.WaitForExit() }
    } else {
        $env:RETAKE_EDITOR_SERVICE_URL = $url
        Push-Location (Join-Path $root 'retake-editor')
        try {
            if ($Release) { cargo run --release -p retake-editor } else { cargo run -p retake-editor }
        } finally { Pop-Location }
    }
} finally {
    if ($service -and -not $service.HasExited) {
        Write-Host 'Stopping the retake service.'
        Stop-Process -Id $service.Id -Force
    }
}
