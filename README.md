# Pykedex - A Pokédox made with python
Small computer vision project that recognizes the enemy Pokémon in **Pokémon Red/Blue (Gen 1)** from a game screenshot or photo of the device screen on which you are gaming.  
Given an input image, the program isolates the battle screen, extracts the enemy sprite and predicts which Pokémon it is using Zernike moments and template matching.

## Requirements

- Python 3.x
- `opencv-python`
- `numpy`
- `mahotas`

Install with:

```bash
pip install opencv-python numpy mahotas
```

## Usage

Build the sprite index (run once):

```bash
python indexer.py --sprites path/to/sprites --output index.pkl
```

Run recognition on a screenshot:

```bash
python poke_search.py --index index.pkl --image path/to/screenshot.png
```

The script prints the predicted Pokémon and can optionally show intermediate debug images.
