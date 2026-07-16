# Remote MAME farm (Windows box)

1. Install **MAME 0.264 exactly** (save-states are version-locked) to C:\mame.
2. Place the dkong ROM set in C:\mame\roms.
3. Copy `scripts/bridge.lua` from this repo to C:\mame\bridge.lua.
4. Create C:\mame\dkstates and SHARE it as `dkstates` (read/write).
5. Firewall (admin PowerShell):
   New-NetFirewallRule -DisplayName "DK Farm" -Direction Inbound -Protocol TCP -LocalPort 5016-5023 -Action Allow
6. Run: powershell -ExecutionPolicy Bypass -File dk_farm.ps1

Trainer side: mount the share (sudo mount -t drvfs '\\192.168.20.59\dkstates' /mnt/dkfarm
or cifs with credentials), run scripts/farm/sync_states.sh, then launch with
remote envs enabled (see launcher notes). Gate on the WiFi probe verdict:
p99 must stay under ~20ms or the lock-step batch will stall on spikes.

## Activation (trainer side)
Create `artifacts/farm.json`:
{"hosts": [{"host": "192.168.20.59", "ports": [5016,5017,5018,5019,5020,5021,5022,5023], "statedir": "/mnt/dkfarm"}]}
Delete the file to return to the single-machine default. Unreachable
ports are skipped automatically at launch.
