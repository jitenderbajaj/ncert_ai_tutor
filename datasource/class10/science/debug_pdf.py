import sys
import re

def analyze_pdf(pdf_path):
    print(f"--- ANALYZING: {pdf_path} ---")
    
    raw_text = ""
    
    # Try PyMuPDF (fitz) first - Best quality
    try:
        import fitz
        print("[INFO] Using PyMuPDF (fitz)")
        doc = fitz.open(pdf_path)
        for page in doc:
            raw_text += page.get_text() + "\n"
    except ImportError:
        # Fallback to pypdf
        try:
            from pypdf import PdfReader
            print("[INFO] Using pypdf (fitz not found)")
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    raw_text += text + "\n"
        except ImportError:
            print("[ERROR] No PDF library found. Please install pymupdf or pypdf.")
            print("pip install pymupdf")
            return

    print(f"[INFO] Extracted {len(raw_text)} characters.")
    
    lines = raw_text.split('\n')
    
    # Define targets to find context for
    targets = ["1.2.2", "Decomposition", "1.2.3", "Displacement"]
    
    print("\n--- CONTEXT DUMP ---")
    print("Looking for lines containing: 1.2.2, Decomposition, 1.2.3, Displacement")
    
    for i, line in enumerate(lines):
        clean_line = line.strip()
        
        # Check if line matches any target
        hit = False
        for t in targets:
            if t in clean_line:
                hit = True
                break
        
        if hit:
            print(f"\n[LINE {i}] >>> '{line}'")
            # Print surrounding lines for context
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            for j in range(start, end):
                marker = ">>>" if j == i else "   "
                print(f"{marker} {j}: '{repr(lines[j])}'")
                
            # Hex dump of the specific line to see hidden chars
            print(f"    HEX: {line.encode('utf-8').hex()}")

if __name__ == "__main__":
    analyze_pdf("ch1.pdf")
