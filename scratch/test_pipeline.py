import sys
import os
import cv2
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TRAINING_PASSWORD, WATERMARK_BITS, VIDEO_FEATURE_COLUMNS
from core.security import generate_zodiak_index
from core.watermark_hybrid import embed_bitstream, extract_bitstream
from core.video_utils import extract_features
from core.attacks import apply_attack

def test():
    print("Membuat frame uji...")
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    zodiak_bits = generate_zodiak_index("Aries", WATERMARK_BITS)
    
    print("Testing embed_bitstream...")
    try:
        stego = embed_bitstream(frame, zodiak_bits, TRAINING_PASSWORD)
        print("Embed OK!")
    except Exception as e:
        print("Embed Error:")
        traceback.print_exc()
        return

    print("Testing extract_features...")
    try:
        feats = extract_features(stego)
        print("Extract Features OK!")
        print("Missing columns?", [c for c in VIDEO_FEATURE_COLUMNS if c not in feats])
    except Exception as e:
        print("Extract Features Error:")
        traceback.print_exc()
        return
        
    print("Testing extract_bitstream...")
    try:
        extracted = extract_bitstream(stego, TRAINING_PASSWORD, num_bits=WATERMARK_BITS)
        print("Extract Bitstream OK!")
    except Exception as e:
        print("Extract Bitstream Error:")
        traceback.print_exc()
        return

if __name__ == "__main__":
    test()
