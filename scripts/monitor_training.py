#!/usr/bin/env python3
"""
monitor_training.py

Monitor de logs en tiempo real para el sistema de entrenamiento Multi-GPU.
Muestra el estado de cada GPU y las últimas líneas de sus logs.
"""

import os
import time
import glob
import curses
from datetime import datetime

LOG_DIR = "logs/multi_gpu"

def get_log_files():
    return sorted(glob.glob(os.path.join(LOG_DIR, "*.log")))

def tail(filename, n=1):
    """Devuelve las últimas n líneas de un archivo."""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            return lines[-n:]
    except:
        return []

def draw_screen(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(1)
    stdscr.timeout(1000)  # Refresh cada 1 segundo

    # Colores
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)

    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        
        # Header
        header = f"🚀 MONITOR DE ENTRENAMIENTO MULTI-GPU | {datetime.now().strftime('%H:%M:%S')}"
        stdscr.addstr(0, 0, header, curses.color_pair(4) | curses.A_BOLD)
        stdscr.addstr(1, 0, "-" * width)

        logs = get_log_files()
        
        if not logs:
            stdscr.addstr(3, 2, "⚠️  No se encontraron logs en logs/multi_gpu/", curses.color_pair(3))
        else:
            row = 3
            for log_file in logs:
                filename = os.path.basename(log_file)
                # Parse filename: SYMBOL_TIMEFRAME_TYPE_ID.log
                try:
                    parts = filename.replace(".log", "").split("_")
                    symbol = parts[0]
                    tf = parts[1]
                    gpu_type = parts[2]
                    gpu_id = parts[3]
                    gpu_info = f"GPU {gpu_id} ({gpu_type})"
                except:
                    gpu_info = filename

                # Get last line
                last_lines = tail(log_file, n=2)
                last_line = last_lines[-1].strip() if last_lines else "Esperando output..."
                
                # Check for errors
                status_color = curses.color_pair(1)
                status_icon = "RUNNING"
                
                if "Traceback" in last_line or "Error" in last_line:
                    status_color = curses.color_pair(3)
                    status_icon = "ERROR"
                elif "KeyboardInterrupt" in last_line:
                    status_color = curses.color_pair(2)
                    status_icon = "STOPPED"
                elif "Saving model" in last_line:
                    status_color = curses.color_pair(1) | curses.A_BOLD
                    status_icon = "SAVING"

                # Draw info
                if row < height - 2:
                    stdscr.addstr(row, 2, f"[{gpu_info}]", curses.color_pair(4))
                    stdscr.addstr(row, 25, f"{symbol} {tf}", curses.A_BOLD)
                    stdscr.addstr(row, 45, status_icon, status_color)
                    
                    # Draw last log line (truncated)
                    log_preview = (last_line[:width-60] + '...') if len(last_line) > width-60 else last_line
                    stdscr.addstr(row+1, 4, f"└─ {log_preview}", curses.A_DIM)
                    
                    row += 3

        stdscr.addstr(height-1, 0, "Presiona 'q' para salir", curses.A_REVERSE)
        
        # Input handling
        c = stdscr.getch()
        if c == ord('q'):
            break

def main():
    try:
        curses.wrapper(draw_screen)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
