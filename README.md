# Fleming 9 Training Course Model

A coherent training-plan draft based on the Fleming 9 / Parameter Golf metaphor.

## Core rules

The objective is **par**, achieved with three simple rules:

1. **Par 4 holes** use **shot 1 for setup** and **shot 2 to reach the green**.
2. **Par 3 holes** try to **reach the green in shot 1**.
3. **Every hole is finished with 2 putts**.

This creates a compact control grammar:

- **Par 3** = `on green` + `putt 1` + `putt 2`
- **Par 4** = `setup` + `on green` + `putt 1` + `putt 2`

## Layer / hole ordering

Handicap-transposed layer order:

- 1-based: `7, 5, 1, 9, 6, 2, 3, 4, 8`
- 0-based: `6, 4, 0, 8, 5, 1, 2, 3, 7`

Pars in that order:

`4, 3, 4, 3, 3, 3, 3, 3, 4` = **Par 30**

Blue-course hole lengths in that order:

`425, 235, 405, 165, 200, 140, 195, 140, 260` would be the strict scorecard order.

For the working training course below, the hole sequence used is:

`405, 140, 195, 140, 235, 200, 425, 260, 165`

Total: **2165 steps**

## Reach / play assumptions

Useful abstractions:

- **250-ish** is near maximum full tee-shot reach.
- **235** is the safer control threshold for deterministic planning.
- Some holes are reachable in one by raw distance, but geometry still makes the correct play a setup shot.
- Six holes are effectively **straight/direct**.
- Three holes are **setup-first** or geometry-sensitive.

## Club / shot abstraction

Starter bag:

- **D** = Driver
- **3** = Long iron / driving-iron class
- **9** = Short iron / lofted precision club
- **P** = Putter

Likely expanded bag:

- **D, 3, 6, 9, P**

Shot qualities:

- **Line-drive** = flatter, longer, more roll
- **Flyball / lofted** = higher, softer landing, less roll
- **Putt 1** = rim finder / prep - get within 3 ft
- **Putt 2** = hole shot - close hole

## Warmup idea

Before the course:

- regular stretch / compile warmup
- bucket-at-the-range style `LR_WARMUP_STEPS` such as **35 or 50**
- objective is to calm the early step-2 spike before scoring begins

---

# 2165-Step Course

## Hole 1 — 405 steps — h3 — long hinge with setup — Par 4

**Intent:** long setup hole with a meaningful hinge.

- Shot 1 — **255** — all-out driver / setup push
- Shot 2 — **145** — mid-iron style approach to green edge
- Shot 3 — **4** — putt 1, get within 3 ft
- Shot 4 — **1** — putt 2, finish

Cumulative end: **405**

## Hole 2 — 140 steps — h6 — straight technical hole — Par 3

**Intent:** direct attack, little drama.

- Shot 1 — **135** — on green
- Shot 2 — **4** — putt 1
- Shot 3 — **1** — putt 2

Cumulative end: **545**

## Hole 3 — 195 steps — h7 — medium direct hole — Par 3

**Intent:** long iron / heavy controlled drive to green.

- Shot 1 — **190** — on green
- Shot 2 — **4** — putt 1
- Shot 3 — **1** — putt 2

Cumulative end: **740**

## Hole 4 — 140 steps — h8 — short direct hole — Par 3

**Intent:** short drive / mid iron / heavy short iron to green.

- Shot 1 — **135** — on green
- Shot 2 — **4** — putt 1
- Shot 3 — **1** — putt 2

Cumulative end: **880**

## Hole 5 — 235 steps — h2 — threshold direct hole — Par 3

**Intent:** control-threshold hole; reachable in one under the planning grammar.

- Shot 1 — **230** — regular drive / heavy long iron to green
- Shot 2 — **4** — putt 1
- Shot 3 — **1** — putt 2

Cumulative end: **1115**

## Hole 6 — 200 steps — h5 — straight carry hole — Par 3

**Intent:** 80% drive / long iron to green.

- Shot 1 — **195** — on green
- Shot 2 — **4** — putt 1
- Shot 3 — **1** — putt 2

Cumulative end: **1315**

## Hole 7 — 425 steps — h1 — long setup hole — Par 4

**Intent:** long fairway setup, then approach over protected landing.

- Shot 1 — **255** — fairway setup / hinge
- Shot 2 — **165** — long-iron or heavy mid-iron approach to green
- Shot 3 — **4** — putt 1
- Shot 4 — **1** — putt 2

Cumulative end: **1740**

## Hole 8 — 260 steps — h9 — dogleg / chip setup hole — Par 4

**Intent:** positional shot first, then short chip to green.

- Shot 1 — **235** — setup drive to position
- Shot 2 — **20** — short-iron / chip to green
- Shot 3 — **4** — putt 1
- Shot 4 — **1** — putt 2

Cumulative end: **2000**

## Hole 9 — 165 steps — h4 — short lofted finisher — Par 3

**Intent:** short straight shot over trees; loft and stick.

- Shot 1 — **160** — on green
- Shot 2 — **4** — putt 1
- Shot 3 — **1** — putt 2

Cumulative end: **2165**

---

# 10× Course — 21650 Steps

Same hole order, same ratios, every shot length multiplied by **10**.

## Hole 1 — 4050 steps — h3 — Par 4

- Shot 1 — **2550**
- Shot 2 — **1450**
- Shot 3 — **40**
- Shot 4 — **10**

Cumulative end: **4050**

## Hole 2 — 1400 steps — h6 — Par 3

- Shot 1 — **1350**
- Shot 2 — **40**
- Shot 3 — **10**

Cumulative end: **5450**

## Hole 3 — 1950 steps — h7 — Par 3

- Shot 1 — **1900**
- Shot 2 — **40**
- Shot 3 — **10**

Cumulative end: **7400**

## Hole 4 — 1400 steps — h8 — Par 3

- Shot 1 — **1350**
- Shot 2 — **40**
- Shot 3 — **10**

Cumulative end: **8800**

## Hole 5 — 2350 steps — h2 — Par 3

- Shot 1 — **2300**
- Shot 2 — **40**
- Shot 3 — **10**

Cumulative end: **11150**

## Hole 6 — 2000 steps — h5 — Par 3

- Shot 1 — **1950**
- Shot 2 — **40**
- Shot 3 — **10**

Cumulative end: **13150**

## Hole 7 — 4250 steps — h1 — Par 4

- Shot 1 — **2550**
- Shot 2 — **1650**
- Shot 3 — **40**
- Shot 4 — **10**

Cumulative end: **17400**

## Hole 8 — 2600 steps — h9 — Par 4

- Shot 1 — **2350**
- Shot 2 — **200**
- Shot 3 — **40**
- Shot 4 — **10**

Cumulative end: **20000**

## Hole 9 — 1650 steps — h4 — Par 3

- Shot 1 — **1600**
- Shot 2 — **40**
- Shot 3 — **10**

Cumulative end: **21650**

---

## Notes

- This document keeps the golf-course abstraction coherent without forcing a literal implementation.
- The 2165-step version is the working compact course.
- The 21650-step version is the same course at 10× scale, preserving hole and shot ratios exactly.
- This is best treated as a **training grammar / control scaffold**, not yet as the final competition branch.
