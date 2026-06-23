# AirDrawVocab face-camera drawing feature

This build adds face-aware camera drawing for the game.

## What was added

1. Camera tool switch
   - Hand/finger mode keeps the existing MediaPipe Hands drawing flow.
   - Face/nose mode uses MediaPipe FaceMesh in the browser.

2. Face/nose pen
   - The nose controls the drawing cursor.
   - Open mouth lightly to draw.
   - Close mouth to lift the pen.
   - Blink both eyes to clear the drawing canvas.
   - Saved samples and game sessions use mode `camera-face`.

3. Face sketch capture
   - The `Face sketch` button captures the current webcam frame.
   - Frontend posts the frame to `POST /camera/face-strokes`.
   - Backend uses the DeepShieldAI-Pro style OpenCV pipeline: Haar face detection, largest face selection, padded face crop, edge extraction, and semantic face-template strokes.
   - The generated strokes are injected into the same draw canvas and stored in `state.strokes` with source `camera-face-sketch`.

4. AI integration
   - Realtime prediction sends `source=camera-face` when the face tool is active.
   - If the drawing contains face-sourced strokes, the realtime panel appends `+face-strokes` to the AI source label.
   - Stroke samples saved from this mode can be exported and used by the self-improving train loop.

5. Privacy behavior
   - The backend processes webcam frames in memory.
   - Frames are not saved to disk by `/camera/face-strokes`.
   - Only normalized drawing strokes are added to the game canvas/dataset.

## Main files changed

- `frontend/app.js`
- `frontend/index.html`
- `frontend/style.css`
- `backend/app.py`
- `camera_face_strokes.py`

## Verification performed

- `node --check frontend/app.js`
- `python3 -m py_compile backend/app.py camera_face_strokes.py`
- Basic no-face endpoint helper test using a non-face screenshot.

Runtime webcam testing and TensorFlow model execution were not performed in the container.
