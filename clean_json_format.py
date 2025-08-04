
import json

# 載入原始 JSON
with open("customer_llm_summaries.json", "r", encoding="utf-8") as f:
    original = json.load(f)

# 建立轉換後的新字典
cleaned = {}

for k, v in original.items():
    try:
        # 將 float 字串轉為 int，再轉回字串
        new_key = str(int(float(k)))
        cleaned[new_key] = v
    except:
        # 若轉換失敗，就保留原樣
        cleaned[k] = v

# 存成新的乾淨檔案
with open("customer_llm_summaries_clean.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)

print("✅ Clean JSON saved as customer_llm_summaries_clean.json")