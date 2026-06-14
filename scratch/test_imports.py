import sys
import traceback

def test_imports():
    try:
        import config
        print("[OK] config.py")
        
        from core import attacks, security, video_utils, watermark_hybrid
        print("[OK] core modules")
        
        from app import dashboard
        print("[OK] app/dashboard.py")
        
        import importlib
        importlib.import_module('ml_pipeline.2_train_detector')
        importlib.import_module('ml_pipeline.3_train_attack_specialist')
        importlib.import_module('ml_pipeline.4_train_logo_specialist')
        print("[OK] ml_pipeline scripts imports")
        
    except Exception as e:
        print("[ERROR] Something failed during import:")
        traceback.print_exc()

if __name__ == "__main__":
    test_imports()
