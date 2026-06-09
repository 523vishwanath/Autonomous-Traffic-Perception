# assets/

Place your demo media here:

- `demo.mp4`  — the final pipeline output video (browser-playable H.264)
- `demo.gif`  — GIF preview (optional, converts automatically from demo.mp4)

## Generate a GIF preview from your output video

```bash
# Requires ffmpeg
ffmpeg -i demo.mp4 \
    -vf "fps=10,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
    -loop 0 demo.gif
```

A 960px wide, 10 FPS GIF from the first ~30 seconds makes a great README header.
