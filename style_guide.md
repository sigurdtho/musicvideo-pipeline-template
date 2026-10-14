# Style guide — __SLUG__

Status: **template**. The user approves this file before any still is generated. Set `status` in `style_guide.json` to `approved` only after that yes. A draft is not a yes.

The numbers live in `style_guide.json`. This page is the same gate in sentences.

## Palette

Eight colours, plus one accent. The accent may cover at most `max_frame_pct` of the frame and at most `max_elements` things in that frame.

| Name | Hex | Role |
| --- | --- | --- |
| paper | #C8C2B6 | ground |
| ink | #1A1917 | darkest |
| graphite | #5E5A54 | mid |
| chalk | #F4F1E8 | highlight |
| ash | #8E887E | dust |
| shadow | #3E4650 | cool shadow |
| mist | #D9D6CE | haze |
| night | #262B30 | deep |
| accent | #8C5A3C | at most 8% of the frame, at most 2 elements |

Replace every hex from the film's master before the first still.

## Type

On-screen text: none. If a later decision adds type, the face is Noto Serif. Orbitron, a terminal HUD and karaoke are out.

## Motion

- Picture rate 24 fps.
- Push-in 6–10%, ease-in-out.
- Parallax: 2–3 layers, a shift of 2–4%.
- A hard cut lands within ±1 frame of a beat (`pipeline/cutaudit.py`).
- Verse cuts follow phrase ends. They do not land on the snare. At most one new picture per sentence pair.
- Chorus cuts are hard cuts on the downbeat.
- Generated motion stays off until a two-shot test is approved.
- Grain, if the film uses it, is a fixed overlay.

## Banned

Text inside an image or video prompt. A negation in a prompt. A real person's name, voice or recording. Book covers and screen stills. Extra hands or extra limbs. A second grade over the whole frame. Saturated colour outside the accent.

## Storyboard

One row per lyric line in master.md. Columns: shot, framing, motion, on-screen text.

| Section | Line | Shot | Framing | Motion | Text |
| --- | --- | --- | --- | --- | --- |
| Verse 1 | One quiet line to start the song. | v1_open | wide, one figure, empty room | push-in 6%, ease-in-out | none |
| Verse 1 | Another line, still unhurried. | v1_open | same frame, held | push-in continues to 8% | none |
| Chorus | Hold the picture. | c_hold | medium, the figure still | hard cut on the downbeat, then a 6% push | none |
| Chorus | Let it breathe. | c_hold | same frame | hold | none |
