#!/usr/bin/env python3
"""
Test numeric templates by generating exactly 1 sample for each defined text template.
Saves outputs to Processed_DATA/TestTemplates/ for listening and quality verification.
Supports dual-text formats: raw numbers in Test_*.txt and written-out words in Test_*_raw.txt.
"""

import os
import sys
import random
import soundfile as sf
from tqdm import tqdm
from typing import List, Tuple

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from components.tts.F5_TTS import F5TTSVietnamese
from Preprocess.generate_numeric_data import NumericTextGenerator, RefAudioSelector, digits_to_words

# Constants
REF_AUDIO_DIR = os.path.join(BASE_DIR, "Processed_DATA", "PhoAudioBook")
TEST_OUTPUT_DIR = os.path.join(BASE_DIR, "Processed_DATA", "TestTemplates")


def main():
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    print("=== Step 1: Initialize Generators ===")
    generator = NumericTextGenerator()
    selector = RefAudioSelector(REF_AUDIO_DIR)
    
    # We will gather exactly one (num_text, word_text) tuple for each specific template
    test_cases = []  # List of Tuple[category, template_index, Tuple[num_text, word_text]]

    # 1. Year Templates
    for idx, t in enumerate(generator.year_templates):
        num_n, word_n = generator.gen_year()
        test_cases.append(("year", idx, (t.format(n=num_n), t.format(n=word_n))))

    # 2. Date Templates
    for idx, t in enumerate(generator.date_templates):
        (thu_num, ngay_num, thang_num, nam_num), (thu_word, ngay_word, thang_word, nam_word) = generator.gen_date()
        num_text = t.format(thu=thu_num, ngay=ngay_num, thang=thang_num, nam=nam_num)
        word_text = t.format(thu=thu_word, ngay=ngay_word, thang=thang_word, nam=nam_word)
        test_cases.append(("date", idx, (num_text, word_text)))

    # 3. Phone Templates
    for idx, t in enumerate(generator.phone_templates):
        num_n, word_n = generator.gen_phone()
        test_cases.append(("phone", idx, (t.format(n=num_n), t.format(n=word_n))))

    # 4. Percentage Templates
    for idx, t in enumerate(generator.percentage_templates):
        num_n, word_n = generator.gen_percentage()
        test_cases.append(("percentage", idx, (t.format(n=num_n), t.format(n=word_n))))

    # 5. Price Templates
    for idx, t in enumerate(generator.price_templates):
        num_n, word_n = generator.gen_price()
        num_sentence = t.format(n=num_n)
        word_sentence = t.replace("{n} VNĐ", "{n} đồng").replace("{n}đ", "{n} đồng").format(n=word_n)
        test_cases.append(("price", idx, (num_sentence, word_sentence)))

    # 6. Count Templates
    for idx, t in enumerate(generator.count_templates):
        num_n, word_n = generator.gen_large_count()
        test_cases.append(("count", idx, (t.format(n=num_n), t.format(n=word_n))))

    # 7. Simple Templates
    for idx, t in enumerate(generator.simple_templates):
        num_n, word_n = generator.gen_simple_num()
        test_cases.append(("simple", idx, (t.format(n=num_n), t.format(n=word_n))))

    # 8. Plain Digits (1 sample)
    num_n, word_n = generator.gen_large_count()
    test_cases.append(("plain_digits", 0, (num_n, word_n)))

    # 9. Plain Single Digits (1 sample)
    digits = [str(random.randint(0, 9)) for _ in range(5)]
    num_n = " ".join(digits)
    word_n = " ".join([digits_to_words(c) for c in digits])
    test_cases.append(("plain_single", 0, (num_n, word_n)))

    print(f"Prepared {len(test_cases)} total test cases (1 sample for every single template).")
    
    print("\n=== Step 2: Initialize F5-TTS Model ===")
    model = F5TTSVietnamese(
        ckpt_file="/home/reg/TTS_DATA/models/f5-tts-v0/model.pt",
        vocoder_name="vocos",
        speed=1.0,
    )
    
    # We will pick a single high-quality reference audio to use for all tests
    # to make listening and comparisons consistent
    ref_wav, _, ref_text = selector.get_random_ref()
    print(f"Using reference audio: {os.path.basename(ref_wav)}")
    print(f"Reference text: '{ref_text}'")

    print("\n=== Step 3: Synthesizing Test Samples ===")
    
    # Create metadata index file to easily review what text matches which audio
    index_path = os.path.join(TEST_OUTPUT_DIR, "index.txt")
    index_lines = []

    for category, idx, (num_sentence, word_sentence) in tqdm(test_cases, desc="Synthesizing templates"):
        filename_base = f"Test_{category}_{idx:02d}"
        wav_path = os.path.join(TEST_OUTPUT_DIR, f"{filename_base}.wav")
        txt_path = os.path.join(TEST_OUTPUT_DIR, f"{filename_base}.txt")
        raw_txt_path = os.path.join(TEST_OUTPUT_DIR, f"{filename_base}_raw.txt")
        
        try:
            # Synthesize using the word sentence (written-out format) for perfect pronunciation
            audio, sr = model.synthesize(
                gen_text=word_sentence,
                ref_audio_path=ref_wav,
                ref_text=ref_text
            )
            
            # Save audio
            sf.write(wav_path, audio, sr)
            
            # Save number label (for training reference)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(num_sentence)
                
            # Save written-out words (used in TTS synthesis)
            with open(raw_txt_path, "w", encoding="utf-8") as f:
                f.write(word_sentence)
                
            index_lines.append(f"{filename_base}.wav | NUM: {num_sentence} | WORD: {word_sentence}")
        except Exception as e:
            print(f"\n[ERROR] Failed to synthesize category '{category}' template {idx}: {e}")
            
    # Write index file
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines))
        
    print(f"\n=== Test Synthesis Completed! ===")
    print(f"Synthesized files saved in: {TEST_OUTPUT_DIR}")
    print(f"Metadata index file written to: {index_path}")
    print("You can open index.txt to read both versions and play the corresponding wav files to test!")


if __name__ == "__main__":
    main()
