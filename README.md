All 45 tests work. This was made to serve as a response to Google's TurboVec.

This watches the quantization coordinate distribution as a leading indicator of recall degradation.

Every design decision traces back to a real constraint. The reservoir is capped at 5,000 vectors not arbitrarily but because that's the crossover point where statistical utility and memory cost balance for typical embedding dimensions. 
