from datasets import load_dataset
from collections import Counter
import re

def run_total_scan():
    print("📡 INITIALIZING FULL DATASET SCAN (35,334 Rows)...")
    print("⏳ This will take 2-5 minutes depending on your internet. Please wait.")
    
    # Streaming the full training split
    ds = load_dataset("CIRCL/vulnerability-cwe-patch", split="train", streaming=True)

    # Counters
    sources = Counter()
    cwe_raw_counts = Counter()
    total_rows = 0
    missing_cwe = 0

    for row in ds:
        total_rows += 1
        
        # 1. Source Analysis
        vuln_id = str(row.get('id', 'Unknown'))
        prefix = vuln_id.split('-')[0] if '-' in vuln_id else 'Other'
        sources[prefix] += 1

        # 2. CWE Normalization Logic
        cwe_data = row.get('cwe')
        if not cwe_data or str(cwe_data).lower() in ['n/a', 'none', 'nan', '']:
            missing_cwe += 1
        else:
            # Clean strings like "CWE-79: Title" -> "CWE-79"
            # Using regex to grab just the "CWE-NUMBER" part
            found_cwes = re.findall(r'CWE-\d+', str(cwe_data))
            if found_cwes:
                for c in found_cwes:
                    cwe_raw_counts[c] += 1
            else:
                # If no ID found, keep the original string but cleaned
                clean_cwe = str(cwe_data).split(':')[0].strip()
                cwe_raw_counts[clean_cwe] += 1

        if total_rows % 5000 == 0:
            print(f"  > 📥 Progress: {total_rows} / 35,334 rows processed...")

    print("\n" + "═"*60)
    print("🏁 FULL DATASET AUDIT COMPLETE")
    print("═"*60)
    
    print(f"\n📊 TOTAL RECORDS SCANNED: {total_rows}")
    print(f"❌ RECORDS MISSING CWE: {missing_cwe} ({ (missing_cwe/total_rows)*100:.2f}%)")

    print("\n🆔 SOURCE DISTRIBUTION:")
    for src, count in sources.most_common():
        print(f"  - {src:8} : {count:6} ({ (count/total_rows)*100:.2f}%)")

    print(f"\n🏷️  UNIQUE CWE IDs FOUND: {len(cwe_raw_counts)}")
    
    print("\n🏆 TOP 15 CWE CATEGORIES (NORMALIZED):")
    print(f"{'CWE ID':<12} | {'Occurrences':<12} | {'% of Data':<10}")
    print("-" * 40)
    for cwe, count in cwe_raw_counts.most_common(15):
        perc = (count / total_rows) * 100
        print(f"{cwe:<12} | {count:<12} | {perc:>8.2f}%")
    print("═"*60)

if __name__ == "__main__":
    run_total_scan()