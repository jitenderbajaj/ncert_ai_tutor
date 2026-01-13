import re

line = '1.2.2 Decomposition Reaction'
# The regex I gave you
tight_pattern = re.compile(r'^(\d+(?:\s*\.\s*\d+)+)\.?\s*(.*)$')

match = tight_pattern.match(line)
if match:
    print(f"MATCH: {match.group(1)} | {match.group(2)}")
else:
    print("NO MATCH")
