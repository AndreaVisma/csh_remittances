import PyPDF2
import re

# Open and read the PDF file
file_path = "C:\\Users\\Andrea Vismara\\Downloads\\Accordi per il 25 Aprile-1.pdf"
lyrics_only = []


def is_chord_line(line):
    """
    Heuristic to determine if a line is a chord line.
    We assume chord lines contain mostly chord patterns like letters A-G, optional # or b,
    and possibly numbers or slashes, and spaces.
    We'll consider a line to be a chord line if most tokens match chord pattern.
    """
    # Remove extra whitespace
    line = line.strip()
    if not line:
        return False
    tokens = line.split()
    if not tokens:
        return False

    chord_pattern = re.compile(r'^[A-G](#|b)?(?:maj|min|dim|aug|sus\d*|add\d*|[0-9]*)?$')
    chord_tokens = 0
    for token in tokens:
        # remove punctuation that might be at the end
        token = token.strip(",.;")
        if chord_pattern.match(token):
            chord_tokens += 1
    # if majority of tokens are chords, consider it chord line
    return chord_tokens > len(tokens) / 2


with open(file_path, "rb") as file:
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text = page.extract_text()
        lines = text.splitlines()
        for line in lines:
            # Skip lines that are chord lines (if a majority of tokens are chords)
            if is_chord_line(line):
                continue
            else:
                lyrics_only.append(line)

# Save the lyrics_only to a file for review
output_text = "\n".join(lyrics_only)
with open("C:\\Users\\Andrea Vismara\\Downloads\\Accordi_per_il_25_Aprile_lyrics_only.txt", "w", encoding="utf-8") as out_file:
    out_file.write(output_text)

output_text[:2000]  # output first 2000 characters for preview
