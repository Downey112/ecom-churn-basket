# Full pipeline: local PostgreSQL -> data -> load -> segment -> basket -> churn -> tests.
# For Docker instead: docker compose up -d; $env:DATABASE_URL='postgresql://postgres:retail@localhost:5433/retail'
# and skip the local_pg line.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$py = '.\.venv\Scripts\python.exe'
if (-not $env:DATABASE_URL) { .\scripts\local_pg.ps1 start }
foreach ($s in '01_fetch_data', '02_load_db', '03_segment', '04_basket', '05_churn') {
    Write-Host "== $s" -ForegroundColor Cyan
    & $py "scripts\$s.py"
    if ($LASTEXITCODE) { throw "$s failed" }
}
& $py -m pytest -q tests
