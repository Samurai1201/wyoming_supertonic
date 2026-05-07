If you have specific text processing needs for your language, you can create a new normalization file and add the necessary actions to it that are applied to each sentence.

### Overview
The engine uses a modular architecture. To add a new language, you need to create a standalone normalizer script and register it in the main engine.

---

### Step 1: Create the Normalizer Script
Create a new file in the same directory as your engine, for example: `de_norm.py` (for German). 

Your class must implement a `normalize` method. It is highly recommended to use the `num2words` library for consistency.

**Template (`de_norm.py`):**
```python
import re
from num2words import num2words

class GermanTextNormalizer:
    def __init__(self):
        # Add your character cleanup rules here
        self._chars_to_delete = "=#$“”„«»*\""
        self._translation_table = str.maketrans("", "", self._chars_to_delete)

    def normalize(self, text: str) -> str:
        if not text.strip():
            return text
            
        # 1. Basic cleanup
        text = text.translate(self._translation_table)
        
        # 2. Convert numbers to words (example)
        text = re.sub(r'\b\d+\b', lambda m: num2words(int(m.group(0)), lang='de'), text)
        
        # 3. Final cleanup (ensure multiple spaces are removed)
        return " ".join(text.split())
```

---

### Step 2: Register the Normalizer in `supertonic_engine.py`

You need to tell the engine to use your new script when the language code matches.

#### 1. Import the class
At the top of `supertonic_engine.py`, add an import block with a safety check:

```python
try:
    from .de_norm import GermanTextNormalizer
    DE_NORMALIZER_AVAILABLE = True
except ImportError:
    DE_NORMALIZER_AVAILABLE = False
```

#### 2. Initialize in `__init__`
Inside the `__init__` method of `SupertonicEngine`, add your new normalizer to the `self.normalizers` dictionary:

```python
def __init__(self, ...):
    # ... existing code ...
    self.normalizers = {}

    # Register Russian
    if RU_NORMALIZER_AVAILABLE:
        self.normalizers["ru"] = RussianTextNormalizer()

    # Register German (New)
    if DE_NORMALIZER_AVAILABLE:
        self.normalizers["de"] = GermanTextNormalizer()
```

---

### Step 3: Test the Integration
1.  **Install dependencies**: If your new normalizer uses a specific library, ensure it's installed (`pip install num2words`).
2.  **Restart the server**: The engine will now automatically route any synthesis request with `lang_code="de"` through your `GermanTextNormalizer`.
3.  **Check logs**: Because of the logic in `synthesize`, you will see a debug log whenever the text is modified by your new normalizer:
    `DEBUG: [de] Text prepared for synthesis: einhundert ...`

### Key Requirements for New Normalizers:
*   **Language Codes**: Use 2-letter ISO codes (e.g., `es`, `fr`, `it`). The engine automatically converts incoming codes like `en-US` to `en`.
*   **Safety**: Always use `try-except` for imports so the server can still start even if a specific language dependency is missing.
*   **Tags**: If you use expression tags (like `<laugh>`), ensure your normalizer does not delete the `<` and `>` brackets.
