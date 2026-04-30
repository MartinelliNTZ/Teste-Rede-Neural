#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de debug para testar a UI
"""
import sys
import traceback

def main():
    print("[DEBUG] Iniciando aplicação...")
    
    try:
        from PySide6.QtWidgets import QApplication
        from ui_main import MainWindow
        print("[DEBUG] Imports bem-sucedidos")
        
        app = QApplication(sys.argv)
        print("[DEBUG] QApplication criado")
        
        window = MainWindow()
        print("[DEBUG] MainWindow criado")
        print(f"[DEBUG] Controller existe: {hasattr(window, 'controller')}")
        print(f"[DEBUG] Worker existe: {hasattr(window.controller, 'worker')}")
        
        window.show()
        print("[DEBUG] Janela exibida. Aguardando entrada do usuário...")
        print("[DEBUG] Execute q() para sair")
        
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"[ERROR] Exceção: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
