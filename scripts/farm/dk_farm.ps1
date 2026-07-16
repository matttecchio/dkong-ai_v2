# DK MAME farm supervisor — run on the Windows box (192.168.20.59).
# Launches 8 headless MAME instances with the training bridge on ports
# 5016-5023 and restarts any that die. Requires MAME 0.264 EXACTLY.
# Usage:  powershell -ExecutionPolicy Bypass -File dk_farm.ps1
$MAME   = "C:\mame\mame.exe"          # adjust to your install
$BRIDGE = "C:\mame\bridge.lua"        # copy from the repo's scripts/
$STATES = "C:\mame\dkstates"          # SHARE THIS FOLDER as \\<host>\dkstates
$PORTS  = 5016..5023
New-Item -ItemType Directory -Force -Path $STATES | Out-Null
$procs = @{}
while ($true) {
  foreach ($p in $PORTS) {
    if (-not $procs[$p] -or $procs[$p].HasExited) {
      $env:DK_BRIDGE_PORT = "$p"
      $env:DK_BRIDGE_BIND = "0.0.0.0"
      New-Item -ItemType Directory -Force -Path "C:\mame\logs" | Out-Null
      $procs[$p] = Start-Process -FilePath $MAME -WorkingDirectory "C:\mame" -ArgumentList @(
        "dkong", "-rompath", "C:\mame\roms",
        "-state_directory", $STATES,
        "-autoboot_script", $BRIDGE,
        "-video", "none", "-sound", "none", "-nothrottle"
      ) -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput "C:\mame\logs\mame_$p.out" `
        -RedirectStandardError  "C:\mame\logs\mame_$p.err"
      Write-Host "$(Get-Date -f HH:mm:ss) started MAME on port $p (pid $($procs[$p].Id))"
      Start-Sleep -Milliseconds 800
    }
  }
  Start-Sleep -Seconds 5
}
