# CPLfold_inter

RNA secondary structure prediction tool based on LinearFold algorithm with base pair bonus matrix support.

## Installation

```bash
pip install numpy numba
```

Optional (for Vienna energy model validation):
```bash
pip install ViennaRNA
```

## Usage

### Basic Usage

```bash
# CONTRAfold mode (default)
python3 CPLfold_inter.py GCGCAAAAGCGC

# Vienna energy model
python3 CPLfold_inter.py GCGCAAAAGCGC --V
```

### With Bonus Matrix

```bash
# Load bonus matrix
python3 CPLfold_inter.py GCGCAAAAGC --bonus base_pair.txt --V

# Adjust bonus weight (alpha)
python3 CPLfold_inter.py GCGCAAAAGC --bonus base_pair.txt --alpha 2.0 --V
```

### Read from stdin

```bash
echo "GCGCAAAAGCGC" | python3 CPLfold_inter.py --V
```

## Parameters

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `seq` | - | - | RNA sequence (positional, optional) |
| `--bonus` | `-p` | None | Bonus matrix file path |
| `--alpha` | `-a` | 1.0 | Bonus weight multiplier |
| `--beamsize` | `-b` | 100 | Beam search size |
| `--verbose` | `-v` | False | Verbose output |
| `--V` | - | False | Use Vienna energy model |
| `--zuker` | - | False | Output suboptimal structures |
| `--delta` | - | 5.0 | Energy delta for suboptimal structures |

## Energy Models

### CONTRAfold Mode (default)

- Machine learning-based scoring function
- Output is unitless score
- Enabled when `--V` is not specified

### Vienna Mode

- Uses Turner 2004 thermodynamic parameters
- Output unit: kcal/mol
- Enabled with `--V` flag

## Bonus Matrix File Format

Tab-separated file with three columns:

```
i	j	score
```

- `i`: First position (1-based index)
- `j`: Second position (1-based index)
- `score`: Pairing support score (float)

### Example (base_pair.txt)

```
1	10	0.85
1	9	0.32
2	9	0.91
2	8	0.45
3	8	0.78
3	7	0.56
4	7	0.67
```

Notes:
- Position 1 and 10 have pairing support of 0.85
- Position 2 and 9 have pairing support of 0.91
- Higher scores indicate higher pairing probability

### Important Notes

1. Index starts from 1 (1-based)
2. Matrix is automatically symmetrized (i-j and j-i are treated as identical)
3. Score range is recommended to be 0-1, adjustable via `--alpha`

## Output Format

```
GCGCAAAAGCGC
((((....)))) (-5.10)
```

- Line 1: Input sequence
- Line 2: Dot-bracket notation with energy in parentheses

## Examples

```bash
# Simple prediction
$ python3 CPLfold_inter.py GGGGAAAACCCC --V
GGGGAAAACCCC
((((....)))) (-5.40)

# With bonus guidance
$ python3 CPLfold_inter.py GCGCAAAAGC --bonus base_pair.txt --alpha 1.5 --V
GCGCAAAAGC
((......)) (-0.30)
```

## Author

Ke Wang
