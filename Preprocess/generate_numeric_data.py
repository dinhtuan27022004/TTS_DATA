#!/usr/bin/env python3
"""
Generate Synthetic Numeric TTS Training Data using F5-TTS.
Synthesizes speech for sentences containing numbers to prevent model forgetting of digits.
Automatically mixes the generated dataset with PhoAudioBook via symbolic links.
Supports generating two text versions: one with raw numbers and one with written-out words for F5-TTS synthesis.
"""

import os
import sys
import glob
import json
import random
import argparse
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from components.tts.F5_V0 import F5TTSVietnamese

# Constants & Default Paths
REF_AUDIO_DIR = os.path.join(BASE_DIR, "Processed_DATA", "PhoAudioBook")
OUTPUT_DIR = os.path.join(BASE_DIR, "Processed_DATA", "NumericData")
MIXED_DIR = os.path.join(BASE_DIR, "Processed_DATA", "FineTune_Mixed")
STATS_PATH = os.path.join(BASE_DIR, "Processed_DATA", "stats_NumericData.json")

# Generation thresholds
REF_MIN_DURATION = 3.0      # seconds
REF_MAX_DURATION = 10.0     # seconds
MIN_OUTPUT_DUR = 0.5        # seconds
RMS_THRESHOLD = 0.0001      # threshold for silent audio check


def num_to_vi(n: int) -> str:
    """Converts any integer n to spoken Vietnamese words."""
    if n == 0:
        return "không"
    if n < 0:
        return "âm " + num_to_vi(abs(n))
    
    units = ["", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
    
    def read_three_digits(num: int, show_hundred: bool = True) -> str:
        h = num // 100
        t = (num % 100) // 10
        u = num % 10
        
        res = []
        if h > 0:
            res.append(units[h] + " trăm")
        elif show_hundred:
            res.append("không trăm")
            
        if t > 0:
            if t == 1:
                res.append("mười")
            else:
                res.append(units[t] + " mươi")
        elif (h > 0 or show_hundred) and u > 0:
            res.append("lẻ")
            
        if u > 0:
            if t > 0 and u == 1:
                if t == 1:
                    res.append("một")
                else:
                    res.append("mốt")
            elif t > 0 and u == 5:
                res.append("lăm")
            elif t > 1 and u == 4:
                res.append("tư")
            else:
                res.append(units[u])
        return " ".join(res)

    groups = []
    temp = n
    while temp > 0:
        groups.append(temp % 1000)
        temp //= 1000
        
    group_names = ["", "nghìn", "triệu", "tỷ"]
    
    res_groups = []
    for i, g in enumerate(groups):
        if g == 0:
            if len(groups) == 1:
                res_groups.append("không")
            continue
            
        show_hundred = (i < len(groups) - 1)
        g_words = read_three_digits(g, show_hundred=show_hundred)
        
        g_name = ""
        if i > 0:
            if i % 3 == 0:
                g_name = "tỷ" * (i // 3)
            else:
                g_name = group_names[i % 3] + " " + "tỷ" * (i // 3)
        
        res_groups.append(g_words + (" " + g_name.strip() if g_name else ""))
        
    return ", ".join(reversed(res_groups)).strip()


def digits_to_words(digits_str: str) -> str:
    """Spells out digits one by one in Vietnamese."""
    digit_names = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
    return " ".join([digit_names[int(c)] for c in digits_str if c.isdigit()])


class NumericTextGenerator:
    """Generates varied and rich Vietnamese sentences containing numbers (both digit and spoken formats)."""
    
    def __init__(self):
        # 1. Templates for Years
        self.year_templates = [
            "Năm {n} là một năm đầy những sự kiện đáng nhớ.",
            "Sự kiện lịch sử hào hùng đó diễn ra vào năm {n}.",
            "Đến năm {n}, công trình này dự kiến sẽ hoàn thành xuất sắc.",
            "Nhà thơ sinh năm {n} tại một làng quê nghèo miền Trung.",
            "Bộ phim truyền hình nổi tiếng được ra mắt công chúng lần đầu vào năm {n}.",
            "Kể từ năm {n}, doanh nghiệp đã liên tục phát triển bứt phá.",
            "Vào năm {n}, một phát minh khoa học vĩ đại đã thay đổi thế giới.",
            "Hiệp định song phương được chính thức ký kết vào năm {n}.",
            "Trường học của chúng tôi được khánh thành vào năm {n}.",
            "Bản báo cáo thống kê tình hình phát triển tính từ năm {n} trở đi."
        ]

        # 2. Templates for Dates
        self.date_templates = [
            "Hôm nay là thứ {thu}, ngày {ngay} tháng {thang}.",
            "Cuộc họp hội đồng quản trị sẽ diễn ra vào ngày {ngay} tháng {thang} năm {nam}.",
            "Lễ kỷ niệm truyền thống được long trọng tổ chức vào ngày {ngay} tháng {thang}.",
            "Gia đình chúng tôi sẽ đi du lịch từ ngày {ngay} tháng {thang} năm {nam}.",
            "Bản hợp đồng thương mại này có hiệu lực bắt đầu từ ngày {ngay} tháng {thang}.",
            "Quyết định chính thức được ban hành vào ngày {ngay} tháng {thang} năm {nam}.",
            "Sự kiện tri ân khách hàng sẽ bắt đầu lúc tám giờ sáng ngày {ngay} tháng {thang}.",
            "Tôi sinh ra vào một ngày mưa tầm tã, chính xác là ngày {ngay} tháng {thang} năm {nam}.",
            "Nhà trường sẽ tổ chức hội thao từ ngày {ngay} tháng {thang}.",
            "Hạn cuối để nộp hồ sơ đăng ký là ngày {ngay} tháng {thang} năm {nam}."
        ]

        # 3. Templates for Phone Numbers
        self.phone_templates = [
            "Mọi chi tiết thắc mắc xin vui lòng liên hệ số điện thoại {n}.",
            "Hãy gọi ngay cho tôi qua số hotline {n} khi bạn đến nơi.",
            "Số điện thoại hỗ trợ kỹ thuật trực tuyến là {n}.",
            "Vui lòng gửi tin nhắn xác nhận vào số thuê bao {n} để đăng ký.",
            "Số điện thoại di động cá nhân của cô ấy là {n}.",
            "Nếu có sự cố khẩn cấp, hãy gọi trực tiếp đến số {n}.",
            "Tổng đài tư vấn chăm sóc khách hàng miễn phí là {n}.",
            "Vui lòng ghi lại số điện thoại liên lạc của phòng ban chúng tôi: {n}.",
            "Bạn có thể nhắn tin Zalo qua số điện thoại di động {n}.",
            "Số hotline tiếp nhận phản ánh phản hồi từ người dân là {n}."
        ]

        # 4. Templates for Percentages
        self.percentage_templates = [
            "Tỷ lệ tăng trưởng kinh tế năm nay đạt mức kỷ lục {n}.",
            "Khoảng {n} nhân viên bày tỏ sự đồng ý hoàn toàn với chính sách phúc lợi mới.",
            "Sản phẩm công nghệ cao được giảm giá sâu đến {n} trong dịp mua sắm cuối năm.",
            "Cơ hội thành công của dự án khởi nghiệp này được đánh giá lên tới {n}.",
            "Nước ngọt chiếm tỷ trọng khoảng {n} tổng lượng nước trên toàn thế giới.",
            "Tần suất xuất hiện lỗi hệ thống đã giảm xuống dưới {n} sau đợt cập nhật.",
            "Hơn {n} ý kiến khảo sát đánh giá dịch vụ đạt tiêu chuẩn chất lượng cao.",
            "Mức thuế VAT áp dụng cho mặt hàng này dự kiến sẽ giảm còn {n}.",
            "Chỉ số hài lòng của khách hàng trong quý này tăng vọt lên {n}.",
            "Hiệu suất hoạt động của máy nén khí đạt tối đa {n} công suất thiết kế."
        ]

        # 5. Templates for Prices
        self.price_templates = [
            "Giá vé xem phim định dạng 3D vào ngày thường là {n} đồng.",
            "Món ăn đặc sản này hiện có mức giá bán là {n} VNĐ tại cửa hàng.",
            "Họ đã đầu tư tổng số tiền lên đến {n} đồng cho chuyến phiêu lưu.",
            "Sản phẩm dưỡng da cao cấp này đang được bán công khai với giá {n}đ.",
            "Tổng quỹ quyên góp từ thiện nhận được số tiền ủng hộ là {n} đồng.",
            "Chi phí sửa chữa động cơ xe máy dự kiến tốn khoảng {n} đồng.",
            "Hóa đơn thanh toán tiền điện tháng này của nhà tôi lên tới {n} đồng.",
            "Bạn có thể sở hữu chiếc áo khoác thời trang này chỉ với giá {n} VNĐ.",
            "Gói dịch vụ internet tốc độ cao có cước phí thuê bao {n} đồng một tháng.",
            "Mức giá sàn cho mỗi cổ phiếu của tập đoàn được ấn định là {n} đồng."
        ]

        # 6. Templates for Counts/Large numbers
        self.count_templates = [
            "Có khoảng {n} người hâm mộ đã đổ về sân vận động để tham gia sự kiện.",
            "Nhà máy hiện đại đã hoàn thành sản xuất thành công {n} sản phẩm.",
            "Quãng đường di chuyển từ trung tâm đến ngoại ô dài khoảng {n} mét.",
            "Trong chiến dịch xanh, các tình nguyện viên đã trồng được {n} cây xanh.",
            "Khu đô thị mới được xây dựng quy mô có hơn {n} hộ gia đình sinh sống.",
            "Dự án đã ghi nhận tổng cộng {n} lượt tải ứng dụng trên toàn cầu.",
            "Thư viện trung tâm lưu trữ hơn {n} đầu sách quý hiếm thuộc nhiều lĩnh vực.",
            "Bức tường thành kiên cố có chiều dài đo được là {n} km.",
            "Nhiệt độ đo được ngoài trời trưa nay đã tăng cao đến {n} độ C.",
            "Công ty vừa tuyển dụng bổ sung thêm {n} nhân sự chất lượng cao."
        ]

        # 7. Templates for mixed simple contexts
        self.simple_templates = [
            "Nhà tôi nằm ở số {n} đường Lê Lợi, quận một.",
            "Hiện tại có {n} chiếc xe ô tô đang đỗ hàng dọc bên lề đường.",
            "Cuốn sách hướng dẫn lập trình này bao gồm tổng cộng {n} trang viết.",
            "Lớp học tiếng Anh giao tiếp buổi tối hôm nay có sĩ số {n} học sinh.",
            "Bây giờ đã là {n} giờ rưỡi chiều rồi, chúng ta nên chuẩn bị đi thôi.",
            "Cửa hàng mở cửa phục vụ khách từ {n} giờ sáng mỗi ngày.",
            "Bài kiểm tra cuối kỳ của tôi đạt điểm số {n} trên thang điểm mười.",
            "Mục tiêu tiếp theo nằm ở dòng số {n} trong tài liệu tham khảo.",
            "Em bé đã tự đếm được từ một đến {n} mà không cần ai giúp đỡ.",
            "Căn hộ chung cư cao cấp của gia đình anh ấy ở tầng thứ {n}.",
            "Tọa độ địa lý của vị trí này được xác định chính xác là {n}.",
            "Hằng số vật lý đặc biệt này có giá trị đo đạc khoảng {n}.",
            "Kết quả đo sai lệch điện tử ghi nhận mức độ {n}.",
            "Chỉ số hiệu chuẩn của thiết bị hiển thị chính xác số {n}.",
            "Tỷ lệ sai số tuyệt đối được ước tính rơi vào khoảng {n}."
        ]

    def format_decimal(self, val: float, decimals: int) -> str:
        """Formats a float using only comma as decimal separator."""
        s = f"{val:.{decimals}f}"
        return s.replace(".", ",")

    def format_large_int(self, val: int) -> str:
        """Formats an integer using NO separator (no dots or commas)."""
        return str(val)

    def gen_phone(self) -> Tuple[str, str]:
        """Generates a random phone number starting with 0."""
        digits = [str(random.randint(0, 9)) for _ in range(9)]
        fmt = random.choice(["plain", "space_3_3_3", "space_4_3_3"])
        
        if fmt == "space_3_3_3":
            num_str = f"0{digits[0]}{digits[1]} {digits[2]}{digits[3]}{digits[4]} {digits[5]}{digits[6]}{digits[7]}{digits[8]}"
        elif fmt == "space_4_3_3":
            num_str = f"0{digits[0]}{digits[1]}{digits[2]} {digits[3]}{digits[4]}{digits[5]} {digits[6]}{digits[7]}{digits[8]}"
        else:
            num_str = "0" + "".join(digits)
            
        spoken_str = "không " + " ".join([digits_to_words(c) for c in digits])
        return num_str, spoken_str

    def gen_year(self) -> Tuple[str, str]:
        """Generates a year string (maximally wide: 1 to 9999)."""
        val = random.randint(1, 9999)
        return str(val), num_to_vi(val)

    def gen_date(self) -> Tuple[Tuple[str, str, str, str], Tuple[str, str, str, str]]:
        """Generates date components (thu, ngay, thang, nam - nam up to 9999)."""
        thu = random.choice(["hai", "ba", "tư", "năm", "sáu", "bảy", "chủ nhật"])
        ngay = random.randint(1, 31)
        thang = random.randint(1, 12)
        nam = random.randint(1, 9999)
        
        return (
            (thu, str(ngay), str(thang), str(nam)),
            (thu, num_to_vi(ngay), num_to_vi(thang), num_to_vi(nam))
        )

    def gen_percentage(self) -> Tuple[str, str]:
        """Generates a percentage string (strictly < 12 digits)."""
        if random.random() < 0.3:
            # Decimal percentage
            before = random.randint(0, 9999)
            after_len = random.randint(1, 3)
            after_digits = "".join([str(random.randint(0, 9)) for _ in range(after_len)])
            sep = ","
            
            num_str = f"{before}{sep}{after_digits}%"
            sep_word = "phẩy"
            spoken_str = f"{num_to_vi(before)} {sep_word} {digits_to_words(after_digits)} phần trăm"
            return num_str, spoken_str
        else:
            val = random.randint(0, 99999)
            num_str = self.format_large_int(val) + "%"
            spoken_str = f"{num_to_vi(val)} phần trăm"
            return num_str, spoken_str

    def gen_price(self) -> Tuple[str, str]:
        """Generates a currency string without any separators (up to 9.9 billion)."""
        val = random.randint(1, 9999999999)
        return str(val), num_to_vi(val)

    def gen_large_count(self) -> Tuple[str, str]:
        """Generates general integers using only commas or no separator (up to 9.9 billion)."""
        val = random.randint(1, 9999999999)
        return self.format_large_int(val), num_to_vi(val)

    def gen_simple_num(self) -> Tuple[str, str]:
        """Generates any numerical value capped strictly at less than 12 digits."""
        r = random.random()
        if r < 0.15:
            val = random.randint(-9, 9)
            return str(val), num_to_vi(val)
        elif r < 0.35:
            # Highly negative/positive integers
            val = random.randint(-999999, 999999)
            return self.format_large_int(val), num_to_vi(val)
        elif r < 0.55:
            # Very large integers
            val = random.randint(1000000, 9999999999)
            return self.format_large_int(val), num_to_vi(val)
        elif r < 0.75:
            # Floating point
            sign = "-" if random.random() < 0.3 else ""
            before = random.randint(0, 999999)
            after_len = random.randint(1, 3)
            after_digits = "".join([str(random.randint(0, 9)) for _ in range(after_len)])
            sep = ","
            
            num_str = f"{sign}{before}{sep}{after_digits}"
            sign_word = "âm " if sign else ""
            sep_word = "phẩy"
            spoken_str = f"{sign_word}{num_to_vi(before)} {sep_word} {digits_to_words(after_digits)}"
            return num_str, spoken_str
        elif r < 0.9:
            # Fractions
            n = random.randint(-100, 100)
            d = random.randint(1, 100)
            num_str = f"{n}/{d}"
            spoken_str = f"{num_to_vi(n)} phần {num_to_vi(d)}"
            return num_str, spoken_str
        else:
            # Scientific notation
            val = random.uniform(-1e5, 1e5)
            s = f"{val:e}"
            parts = s.split("e")
            base_val = float(parts[0])
            exp_val = int(parts[1])
            
            sep = ","
            base_str = f"{base_val:.2f}".replace(".", sep)
            exp_str = f"{exp_val:+03d}"
            num_str = f"{base_str}e{exp_str}"
            
            base_before = int(abs(base_val))
            base_after = f"{abs(base_val):.2f}".split(".")[1]
            sign_word = "âm " if base_val < 0 else ""
            sep_word = "phẩy"
            base_spoken = f"{sign_word}{num_to_vi(base_before)} {sep_word} {digits_to_words(base_after)}"
            
            spoken_str = f"{base_spoken} nhân mười mũ {num_to_vi(exp_val)}"
            return num_str, spoken_str

    def generate_single_sentence(self, category: str) -> Tuple[str, str]:
        """Generates a sentence containing a number of the specified category.
        Returns a tuple of (number_sentence, spoken_sentence).
        """
        if category == "year":
            num_n, word_n = self.gen_year()
            template = random.choice(self.year_templates)
            return template.format(n=num_n), template.format(n=word_n)
        elif category == "date":
            (thu_num, ngay_num, thang_num, nam_num), (thu_word, ngay_word, thang_word, nam_word) = self.gen_date()
            template = random.choice(self.date_templates)
            return (
                template.format(thu=thu_num, ngay=ngay_num, thang=thang_num, nam=nam_num),
                template.format(thu=thu_word, ngay=ngay_word, thang=thang_word, nam=nam_word)
            )
        elif category == "phone":
            num_n, word_n = self.gen_phone()
            template = random.choice(self.phone_templates)
            return template.format(n=num_n), template.format(n=word_n)
        elif category == "percentage":
            num_n, word_n = self.gen_percentage()
            template = random.choice(self.percentage_templates)
            return template.format(n=num_n), template.format(n=word_n)
        elif category == "price":
            num_n, word_n = self.gen_price()
            template = random.choice(self.price_templates)
            num_sentence = template.format(n=num_n)
            word_sentence = template.replace("{n} VNĐ", "{n} việt nam đồng").replace("{n}đ", "{n} đồng").format(n=word_n)
            return num_sentence, word_sentence
        elif category == "count":
            num_n, word_n = self.gen_large_count()
            template = random.choice(self.count_templates)
            return template.format(n=num_n), template.format(n=word_n)
        elif category == "simple":
            num_n, word_n = self.gen_simple_num()
            template = random.choice(self.simple_templates)
            return template.format(n=num_n), template.format(n=word_n)
        elif category == "plain_digits":
            num_n, word_n = self.gen_large_count()
            return num_n, word_n
        elif category == "plain_single":
            digits = [str(random.randint(0, 9)) for _ in range(random.randint(2, 6))]
            num_n = " ".join(digits)
            word_n = " ".join([digits_to_words(c) for c in digits])
            return num_n, word_n
        else:
            num_n, word_n = self.gen_year()
            template = random.choice(self.year_templates)
            return template.format(n=num_n), template.format(n=word_n)

    def generate_batch(self, count: int) -> List[Tuple[str, str]]:
        """Generates a balanced batch of sentences containing numbers."""
        categories = ["year", "date", "phone", "percentage", "price", "count", "simple", "plain_digits", "plain_single"]
        weights = [0.12, 0.12, 0.15, 0.12, 0.12, 0.15, 0.12, 0.05, 0.05]
        
        generated = []
        for _ in range(count):
            cat = random.choices(categories, weights=weights)[0]
            generated.append(self.generate_single_sentence(cat))
            
        # Shuffle to mix all categories randomly
        random.shuffle(generated)
        return generated


class RefAudioSelector:
    """Scans and selects valid reference audios and transcripts from the dataset."""
    
    def __init__(self, ref_dir: str):
        self.ref_dir = ref_dir
        self.valid_refs: List[Tuple[str, str, str]] = []  # (wav_path, txt_path, transcript)
        
        if not os.path.isdir(ref_dir):
            raise FileNotFoundError(f"Reference directory does not exist: {ref_dir}")
            
        self._scan_and_filter()

    def _scan_and_filter(self):
        """Scans directory for .wav files and matches transcripts without duration filtering."""
        print(f"Scanning '{self.ref_dir}' for reference audios...")
        wav_files = glob.glob(os.path.join(self.ref_dir, "*.wav"))
        
        if not wav_files:
            print(f"Warning: No .wav files found in '{self.ref_dir}'. Checking subdirectories...")
            wav_files = glob.glob(os.path.join(self.ref_dir, "**", "*.wav"), recursive=True)
            
        print(f"Found {len(wav_files)} total wav files. Loading transcripts...")
        
        for wav_path in tqdm(wav_files):
            txt_path = os.path.splitext(wav_path)[0] + ".txt"
            if not os.path.exists(txt_path):
                continue
                
            try:
                # Load transcript
                with open(txt_path, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()
                    
                if transcript:
                    self.valid_refs.append((wav_path, txt_path, transcript))
            except Exception as e:
                # Silently ignore files with read errors during scan
                continue
                
        print(f"Successfully loaded {len(self.valid_refs)} reference audio/text pairs.")
        if not self.valid_refs:
            raise ValueError(f"No valid reference audio files found in {self.ref_dir}.")

    def get_random_ref(self) -> Tuple[str, str, str]:
        """Returns a random valid reference (wav_path, txt_path, transcript)."""
        return random.choice(self.valid_refs)


class SyntheticDataGenerator:
    """Manages the generation of synthetic audio using F5-TTS model."""
    
    def __init__(self, output_dir: str, stats_path: str):
        self.output_dir = output_dir
        self.stats_path = stats_path
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.processed_files = set()
        self.stats = {
            "total_requested": 0,
            "generated_successfully": 0,
            "failed_duration": 0,
            "failed_silence": 0,
            "failed_errors": 0,
            "resumed_count": 0
        }
        
        self._load_state()

    def _load_state(self):
        """Loads execution state for resume capability."""
        # Find existing files in output directory
        existing_wavs = [f for f in os.listdir(self.output_dir) if f.endswith(".wav")]
        for f in existing_wavs:
            self.processed_files.add(f)
            
        self.stats["resumed_count"] = len(existing_wavs)
        self.stats["generated_successfully"] = len(existing_wavs)
        print(f"Resume state: found {len(existing_wavs)} already generated wav files.")
        
        # Load existing stats if available
        if os.path.exists(self.stats_path):
            try:
                with open(self.stats_path, "r", encoding="utf-8") as f:
                    old_stats = json.load(f)
                    # Update counts while keeping resumed counts accurate
                    for k, v in old_stats.items():
                        if k in self.stats and k != "resumed_count":
                            self.stats[k] = v
            except Exception:
                print("Warning: Could not read old stats file, starting fresh stats.")

    def save_stats(self):
        """Writes current statistics to file."""
        with open(self.stats_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=4, ensure_ascii=False)

    def is_valid_audio(self, audio: np.ndarray, sr: int) -> Tuple[bool, str]:
        """Checks if synthesized audio is valid (not too short, not silent)."""
        duration = len(audio) / sr
        if duration < MIN_OUTPUT_DUR:
            return False, f"too_short ({duration:.2f}s < {MIN_OUTPUT_DUR}s)"
            
        # Calculate root-mean-square (RMS) energy to detect silence or collapse
        rms = np.sqrt(np.mean(audio**2))
        if rms < RMS_THRESHOLD:
            return False, f"silent (RMS {rms:.6f} < {RMS_THRESHOLD})"
            
        return True, "ok"

    def generate(self, texts: List[Tuple[str, str]], selector: RefAudioSelector, num_samples: int):
        """Main synthesis loop."""
        self.stats["total_requested"] = num_samples
        
        needed = num_samples - len(self.processed_files)
        if needed <= 0:
            print("All requested samples are already generated! Skipping synthesis.")
            return
            
        print(f"Initializing F5-TTS model...")
        # Initialize F5-TTS
        model = F5TTSVietnamese(vocoder_name="vocos", speed=1.0)
        
        print(f"Generating {needed} synthetic numeric speech files...")
        
        # Take the slice of texts needed
        texts_to_process = texts[:needed]
        
        # We determine the starting index based on existing files in directory
        existing_indices = []
        for fn in self.processed_files:
            try:
                # Format: NumericData_000123.wav
                idx = int(fn.split("_")[1].split(".")[0])
                existing_indices.append(idx)
            except Exception:
                continue
        
        next_idx = max(existing_indices) + 1 if existing_indices else 0
        
        pbar = tqdm(total=needed, desc="Synthesizing")
        
        for num_sentence, word_sentence in texts_to_process:
            filename_base = f"NumericData_{next_idx:06d}"
            wav_name = f"{filename_base}.wav"
            txt_name = f"{filename_base}.txt"
            
            wav_path = os.path.join(self.output_dir, wav_name)
            txt_path = os.path.join(self.output_dir, txt_name)
            
            # Select random reference
            ref_wav, _, ref_text = selector.get_random_ref()
            
            success = False
            attempts = 0
            max_attempts = 3
            
            while not success and attempts < max_attempts:
                attempts += 1
                try:
                    # Synthesize using F5-TTS with the fully written-out spoken text
                    audio, sr = model.synthesize(
                        gen_text=word_sentence,
                        ref_audio_path=ref_wav,
                        ref_text=ref_text
                    )
                    
                    # Validate
                    valid, reason = self.is_valid_audio(audio, sr)
                    if valid:
                        # Save audio
                        sf.write(wav_path, audio, sr)
                        
                        # Save standard label: number version
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(num_sentence)
                            
                        # Save spoken raw text label: word version
                        raw_txt_path = os.path.splitext(txt_path)[0] + "_raw.txt"
                        with open(raw_txt_path, "w", encoding="utf-8") as f:
                            f.write(word_sentence)
                            
                        self.stats["generated_successfully"] += 1
                        self.processed_files.add(wav_name)
                        success = True
                        next_idx += 1
                    else:
                        if "too_short" in reason:
                            self.stats["failed_duration"] += 1
                        else:
                            self.stats["failed_silence"] += 1
                        # Get a new reference for the next attempt
                        ref_wav, _, ref_text = selector.get_random_ref()
                except Exception as e:
                    self.stats["failed_errors"] += 1
                    # Get a new reference for the next attempt
                    ref_wav, _, ref_text = selector.get_random_ref()
            
            pbar.update(1)
            self.save_stats()
            
        pbar.close()
        print(f"Synthesis finished! Success: {self.stats['generated_successfully']}/{num_samples}")
        print(f"Stats saved to: {self.stats_path}")


def create_mixed_dataset_symlinks():
    """
    Creates a virtual mixed dataset directory containing symbolic links to files in both
    PhoAudioBook and NumericData. This avoids duplicating audio data on disk.
    """
    print(f"Merging datasets via symlinks into: '{MIXED_DIR}'")
    os.makedirs(MIXED_DIR, exist_ok=True)
    
    # 1. Clear existing dead or old symlinks in the mixed directory
    print("Cleaning up old links...")
    for item in os.listdir(MIXED_DIR):
        item_path = os.path.join(MIXED_DIR, item)
        if os.path.islink(item_path) or os.path.exists(item_path):
            try:
                os.unlink(item_path)
            except Exception as e:
                print(f"Error removing {item_path}: {e}")
                
    # 2. Add symlinks for PhoAudioBook
    print("Linking PhoAudioBook dataset...")
    pho_files = glob.glob(os.path.join(REF_AUDIO_DIR, "*.*"))
    for f in tqdm(pho_files):
        base = os.path.basename(f)
        target_link = os.path.join(MIXED_DIR, base)
        # Use absolute path for safety
        os.symlink(os.path.abspath(f), target_link)
        
    # 3. Add symlinks for NumericData
    # Note: We only symlink .wav and .txt (raw numbers), we skip _raw.txt to keep the fine-tuning clean!
    print("Linking NumericData dataset...")
    num_files = glob.glob(os.path.join(OUTPUT_DIR, "*.*"))
    for f in tqdm(num_files):
        base = os.path.basename(f)
        # Skip _raw.txt in mixed dataset to keep training clean
        if "_raw.txt" in base:
            continue
        target_link = os.path.join(MIXED_DIR, base)
        # Use absolute path for safety
        os.symlink(os.path.abspath(f), target_link)
        
    # Verify counts
    total_links = len([item for item in os.listdir(MIXED_DIR)])
    print(f"Dataset merge complete. Created {total_links} total symbolic links in '{MIXED_DIR}'.")
    print(f"You can now configure F5-TTS to train directly using this directory.")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic numeric speech data.")
    parser.add_argument("--num-samples", type=int, default=1500, help="Number of synthetic samples to generate.")
    parser.add_argument("--create-symlink", action="store_true", help="Create the mixed dataset virtual directory via symlinks.")
    parser.add_argument("--only-symlink", action="store_true", help="Only run the symlink merger and exit.")
    args = parser.parse_args()
    
    if args.only_symlink:
        create_mixed_dataset_symlinks()
        return

    print("=== Step 1: Initialize Numeric Text Generator ===")
    generator = NumericTextGenerator()
    # Generate more than needed to allow room for filtering
    raw_texts = generator.generate_batch(args.num_samples * 2)
    print(f"Generated {len(raw_texts)} rich sentences with numbers.")

    print("\n=== Step 2: Initialize Reference Audio Selector ===")
    selector = RefAudioSelector(REF_AUDIO_DIR)

    print("\n=== Step 3: Run Synthetic Data Generation ===")
    generator_engine = SyntheticDataGenerator(OUTPUT_DIR, STATS_PATH)
    generator_engine.generate(raw_texts, selector, args.num_samples)

    if args.create_symlink:
        print("\n=== Step 4: Merge Datasets via Symlinks ===")
        create_mixed_dataset_symlinks()
    else:
        print("\n=== Dataset merging via symlinks skipped (use --create-symlink to run) ===")
    
    print("\n=== Completed Successfully! ===")


if __name__ == "__main__":
    main()
