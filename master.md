# __SLUG__ (Draft 1)

master.md is the source of truth. The pipeline reads the BPM, the key, DURATION, the STYLE and NEGATIVE lines, the CHARACTER lines, and the generator-lyrics block.

## Concept

One paragraph: the hook, who speaks or sings, the world, the tone.

- **Device:** the one recurring picture idea. It is not the previous film's device.
- **Arc:** where the video starts and where it ends.
- **Length:** about 4:00.

## Lyrics

Stage 0, before any take. A frontier model (Opus 5.5, or an equal) researches the source, then drafts. Fill
`qc/lyrics_brief.md`: rhyme scheme, syllable count per line, words per second per section (slow-genre verses at
or under about 2.3), and phonetic spellings for tricky names in the generator feed. Then one or two revision
rounds with the user. Lines stay short enough to sing or speak. The lines below keep the real spelling.

**[Verse 1]**
One quiet line to start the song.
Another line, still unhurried.

**[Chorus]**
Hold the picture.
Let it breathe.

## Genre

Name the genre, the vocal, and the tempo in one paragraph. Include a number of BPM and a key such as C minor.

120 BPM. C minor.

DURATION = 180

## Song production

| Section | Bars | Vocal | Arrangement |
| --- | --- | --- | --- |
| Verse 1 | 8 | lead | bed only |
| Chorus | 8 | lead | the lift |

### Generator lyrics

```text
[Verse 1]
One quiet line to start the song.
Another line, still unhurried.

[Chorus]
Hold the picture.
Let it breathe.
```

### Audio toolchain

Song model (ACE-Step 1.5 XL turbo and YuE2), stems (BS-RoFormer), alignment (WhisperX), voices (Qwen3-TTS), mix (pedalboard).

## Characters

### Lead (narrator)
CHARACTER: A person standing in a plain room, calm face, plain grey clothes, empty hands.
LOOK: an original design. No official art and no real person.
VOICE: calm, close to the mic, unhurried.

## Music video beats

Section by section: what we see. One picture idea per line goes in style_guide.md before any still is generated.
Run qc/coherence.md when the concept and the lyric exist, and again before the style guide is approved. That
check is information, not a gate.

## Video pipeline

### Style
STYLE: Muted cinematic still, soft daylight, grey walls, warm wood, shallow depth of field.
NEGATIVE: text, letters, watermark, logo, extra limbs, extra people.

### Overlay box spec

Fill this only if the style guide allows on-screen text. Default for a new film is no text in the picture and no overlay until the guide says otherwise.

### Video toolchain

Stills (Flux.2 Klein), clips (Wan 2.2 I2V, only after a two-shot test), lip-sync off until a later decision, overlays (headless Chrome), upscale (SeedVR2 7B), interpolation (RIFE).

## Orchestration

Stages 0-10. Gates: song_take on, looks off, final_cut on. The style guide is approved before stage 6.

### CLAUDE.md

The agent rules live in CLAUDE.md.
