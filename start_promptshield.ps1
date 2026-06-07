$Python = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$env:PYTHONPATH = Join-Path (Get-Location) ".python_packages"
& $Python -B "server.py" --host "127.0.0.1" --port 3000

