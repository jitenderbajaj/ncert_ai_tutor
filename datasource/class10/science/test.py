import fitz # pip install pymupdf

doc = fitz.open("ch1.pdf")
text = ""
for page in doc:
    text += page.get_text()

# Find the problematic area
start = text.find("Combination Reaction")
end = text.find("Double Displacement")

if start != -1 and end != -1:
    print("--- RAW TEXT DUMP ---")
    print(text[start:end+100])
    print("---------------------")
else:
    print("Could not find context markers.")
