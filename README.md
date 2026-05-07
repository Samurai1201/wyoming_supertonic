# Wyoming [Supertonic](https://github.com/supertone-inc/supertonic)

Wyoming server for Supertonic TTS (V3).

## Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/mitrokun/wyoming_supertonic.git
cd wyoming_supertonic

python3 -m venv venv
source venv/bin/activate
pip install supertonic wyoming sentence-stream num2words onnxruntime numpy
```


## Usage

Run the server pointing to your model directory:

```bash
python3 -m wyoming_supertonic --uri 'tcp://0.0.0.0:10209'
```

### Arguments

*   `--language` Default voice language (default: `en`).      
*   `--uri`: Server URI (default: `tcp://0.0.0.0:10209`).
*   `--speed`: Speech speed, 0.5 to 2.0 (default: `1.0`).
*   `--steps`: Denoising steps. Higher is better quality but slower (default: `5`).
*   `--threads`: Number of CPU threads to use (default: `4`).
*   `--no-streaming`: Disable sentence-by-sentence streaming.
*   `--debug`: Enable debug logging.

The initial synthesized audio for each sentence is framed by silence on both sides. I chose a default trim value of 300ms; if the pauses between sentences are unsatisfactory for you, try adjusting `--crop-silence` parameter.

### Supported Languages:
| Code | Language | Code | Language | Code | Language |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ar` | Arabic | `fr` | French | `pt` | Portuguese |
| `bg` | Bulgarian | `hi` | Hindi | `ro` | Romanian |
| `cs` | Czech | `hr` | Croatian | `ru` | Russian |
| `da` | Danish | `hu` | Hungarian | `sk` | Slovak |
| `de` | German | `id` | Indonesian | `sl` | Slovenian |
| `el` | Greek | `it` | Italian | `sv` | Swedish |
| `en` | English | `ja` | Japanese | `tr` | Turkish |
| `es` | Spanish | `ko` | Korean | `uk` | Ukrainian |
| `et` | Estonian | `lt` | Lithuanian | `vi` | Vietnamese |
| `fi` | Finnish | `lv` | Latvian | `nl` | Dutch |

## Quick start with uv

```
git clone https://github.com/mitrokun/wyoming_supertonic.git
cd wyoming_supertonic
UV_CACHE_DIR=.uv_cache uv run -m wyoming_supertonic  --uri 'tcp://0.0.0.0:10209'
```

