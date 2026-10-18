# Runs a project-local PostgreSQL cluster (no Docker, no admin rights, no password).
# Usage:  .\scripts\local_pg.ps1 start | stop | status
# Uses the binaries of an installed PostgreSQL (default: newest under C:\Program Files\PostgreSQL).
param([ValidateSet('start','stop','status')][string]$Action = 'start', [int]$Port = 5433)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$data = Join-Path $root '.pgdata'
$bin = if ($env:PG_BIN) { $env:PG_BIN } else {
    (Get-ChildItem 'C:\Program Files\PostgreSQL\*\bin' | Sort-Object { [int]$_.Parent.Name } | Select-Object -Last 1).FullName }
if (-not (Test-Path $data)) {
    & "$bin\initdb.exe" -D $data -U postgres -A trust -E UTF8 --locale=C | Out-Null
    Write-Host "initialised cluster in $data"
}
switch ($Action) {
    'start'  { & "$bin\pg_ctl.exe" -D $data -o "-p $Port" -l (Join-Path $data 'server.log') -w start
               & "$bin\psql.exe" -h localhost -p $Port -U postgres -tAc "SELECT 1 FROM pg_database WHERE datname='retail'" | Set-Variable exists
               if (-not $exists) { & "$bin\createdb.exe" -h localhost -p $Port -U postgres retail; Write-Host 'created database retail' } }
    'stop'   { & "$bin\pg_ctl.exe" -D $data -w stop }
    'status' { & "$bin\pg_ctl.exe" -D $data status }
}
