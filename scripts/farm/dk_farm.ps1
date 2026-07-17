# DK MAME farm supervisor — run on the Windows box (192.168.20.59).
# Launches 8 headless MAME instances with the training bridge on ports
# 5016-5023 and restarts any that die. Requires MAME 0.264 EXACTLY.
# Usage:  powershell -ExecutionPolicy Bypass -File dk_farm.ps1
$MAME   = "C:\mame\mame.exe"          # adjust to your install
$BRIDGE = "C:\mame\bridge.lua"        # copy from the repo's scripts/
$STATES = "C:\mame\dkstates"          # SHARE THIS FOLDER as \\<host>\dkstates
$PORTS  = 5016..5023
New-Item -ItemType Directory -Force -Path $STATES | Out-Null
# SINGLE-SUPERVISOR TAKEOVER (2026-07-17): forgotten farm windows kept
# auto-restarting dead MAMEs with era-old bridge copies, racing the new
# window for ports — half the farm randomly ran fragile bridges. The
# newest supervisor owns the lock; older ones see a foreign PID, kill
# their own children, and exit.
$host.UI.RawUI.WindowTitle = "DK FARM supervisor"
$LOCK = "$STATES\farm_supervisor.pid"
"$PID" | Out-File -Force $LOCK
$procs = @{}
# Ctrl+C in this window kills all MAMEs cleanly (finally block).
# If the window is CLOSED instead, children survive — use stop_farm.bat.
try {
while ($true) {
  $owner = Get-Content $LOCK -ErrorAction SilentlyContinue
  if ("$owner" -ne "$PID") {
    Write-Host "newer supervisor ($owner) took over -> stopping my MAMEs and exiting"
    break
  }
  foreach ($p in $PORTS) {
    if (-not $procs[$p] -or $procs[$p].HasExited) {
      $env:DK_BRIDGE_PORT = "$p"
      $env:DK_BRIDGE_BIND = "0.0.0.0"
      # logs live in the SHARED folder so the Linux side can read them
      New-Item -ItemType Directory -Force -Path "$STATES\logs" | Out-Null
      $procs[$p] = Start-Process -FilePath $MAME -WorkingDirectory "C:\mame" -ArgumentList @(
        "dkong", "-rompath", "C:\mame\roms",
        "-state_directory", $STATES,
        "-autoboot_script", $BRIDGE,
        "-video", "none", "-sound", "none", "-nothrottle"
      ) -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput "$STATES\logs\mame_$p.out" `
        -RedirectStandardError  "$STATES\logs\mame_$p.err"
      Write-Host "$(Get-Date -f HH:mm:ss) started MAME on port $p (pid $($procs[$p].Id))"
      Start-Sleep -Milliseconds 800
    }
  }
  Start-Sleep -Seconds 5
}
} finally {
  Write-Host "shutting down farm..."
  Get-Process mame -ErrorAction SilentlyContinue | Stop-Process -Force
}
