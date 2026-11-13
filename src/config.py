"""全局配置：路径与预算常量。

单位约定（与项目说明一致，勿改）：
    MB = 1,000,000 字节（十进制），不是 1,048,576
    项目 PDF 表格里的 "63.90 MB" 就是 63,899,977 字节 / 1e6
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw" / "CS5185_Project_Video_Dataset"
META_DIR = RAW_DIR / "metadata"
OUT_DIR = ROOT / "data" / "out"
BASELINE_DIR = ROOT / "data" / "out_baseline"
LOG_DIR = ROOT / "logs"
RESULT_DIR = ROOT / "results"
FEATURE_DIR = RESULT_DIR / "features"

MB = 1_000_000

# 附录 B2：留 2% 给容器开销（moov 头等）
OVERHEAD = 0.98

# 两个目标压缩比例
RATIOS = (0.05, 0.02)

# 附录 A：一次任务最多尝试轮数（重编码次数 = MAX_ATTEMPTS - 1）
MAX_ATTEMPTS = 3

# 开发集 / 最终测试集（项目说明 p3 指定）
DEV_IDS = ("01_FourPeople", "02_Beauty", "03_Bosphorus", "04_HoneyBee",
           "07_Mobcal", "08_Shields")
TEST_IDS = ("05_Jockey", "06_ShakeNDry", "09_Parkrun", "10_Stockholm")
RESERVE_IDS = ("11_Sintel", "12_ControlledBurn")


def ensure_dirs() -> None:
    for d in (OUT_DIR, BASELINE_DIR, LOG_DIR, RESULT_DIR, FEATURE_DIR):
        d.mkdir(parents=True, exist_ok=True)
