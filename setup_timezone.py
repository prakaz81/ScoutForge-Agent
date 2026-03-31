#!/usr/bin/env python3
"""Auto-detect system timezone and update .env for docker-compose"""
import subprocess
import os
import sys
import platform

def get_system_timezone():
    """Detect system timezone - works on macOS, Windows, and Linux"""
    system = platform.system()
    
    try:
        if system == "Darwin":  # macOS
            result = subprocess.run(
                ["systemsetup", "-gettimezone"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.split(": ")[1].strip()
        
        elif system == "Windows":
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-TimeZone).Id"],
                capture_output=True,
                text=True,
                timeout=5,
                shell=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        
        else:  # Linux
            # Try /etc/timezone first
            try:
                with open("/etc/timezone", "r") as f:
                    return f.read().strip()
            except:
                pass
            
            # Try timedatectl
            result = subprocess.run(
                ["timedatectl", "show", "-p", "Timezone", "--value"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
    
    except Exception as e:
        print(f"Warning: Could not detect timezone: {e}", file=sys.stderr)
    
    return "UTC"

def update_env_file(tz):
    """Update .env file with TZ variable"""
    env_file = ".env"
    
    if not os.path.exists(env_file):
        with open(env_file, "w") as f:
            f.write(f"TZ={tz}\n")
        print(f"Created {env_file} with TZ={tz}")
        return
    
    # Read existing .env
    with open(env_file, "r") as f:
        lines = f.readlines()
    
    # Update or add TZ line
    found = False
    for i, line in enumerate(lines):
        if line.startswith("TZ="):
            lines[i] = f"TZ={tz}\n"
            found = True
            break
    
    if not found:
        lines.append(f"TZ={tz}\n")
    
    # Write back
    with open(env_file, "w") as f:
        f.writelines(lines)
    
    print(f"Updated {env_file}: TZ={tz}")

if __name__ == "__main__":
    tz = get_system_timezone()
    print(f"Detected system timezone: {tz}")
    update_env_file(tz)
    print("Ready to run: docker-compose up -d")
