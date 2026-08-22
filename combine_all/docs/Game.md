# Game Module (osu!)

This module implements an osu!mania style rhythm game interface using `pygame`. It serves as the primary stimulus presentation and user interaction environment for the BCI system.

## Components

### `game_view.py`
Handles all the rendering logic for the game.
- Draws notes, lanes, hit effects, score, and accuracy.
- Manages different screens like the login screen, main menu, pause menu, and the actual gameplay screen.
- Manages Pygame display initialization and frame rate control.

### `osu_interface.py`
The core logic handler for the rhythm game mechanics.
- **Hit Detection**: Checks if the predicted key or manually pressed key matches the notes in the correct time window.
- **Marker Calculation**: Calculates the current game state to determine the `marker` for the EEG system, synchronizing game events with EEG recordings.
- Plays music and handles game timing using LSL (Lab Streaming Layer) clock for high-precision synchronization.

### `osu_parser.py`
Parses `.osu` beatmap files.
- Extracts metadata (title, artist, etc.).
- Parses hit objects to determine the timing (`time`), type, and lane of each note.

### `songs/` & `sound_effect/`
Directories storing the `.osu` beatmaps, music files, and sound effects used during gameplay.

## Integration

The game module continuously syncs with the EEG system via LSL clocks. As notes approach the hit line, the system uses predictions from the `EEG_DL` module to simulate keypresses, or allows manual override. The results (hits/misses) are recorded and displayed to the user.
