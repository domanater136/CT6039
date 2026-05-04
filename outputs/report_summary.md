# CS2 Dissertation Analysis - Summary

---

## 1. Dataset

- Maps retained: **16,695**
- Maps excluded: **21** (count: 21, nulls: 0, bad rounds: 0)
- Players: **2,310**
- Matches: **7,242**

---

## 2. Context Regressions

- **rating**: teammate b = 0.0064  (adj R2 = 0.0001)
- **adr**: teammate b = 0.0082  (adj R2 = 0.0001)
- **kd_log**: teammate b = 0.0143 . (adj R2 = 0.0001)

---

## 3. RAPM

- Ridge alpha: **30.0**
- Active players: **850**

| label     |   rapm_score |
|:----------|-------------:|
| sunpayus  |       2.325  |
| pr1metapz |       2.3246 |
| ropz      |       2.2871 |
| m0nesy    |       2.1624 |
| niko      |       2.1353 |
| yekindar  |       2.079  |
| vsm       |       2.0777 |
| kisserek  |       2.0678 |
| zont1x    |       2.0428 |
| molodoy   |       1.9945 |

---

## 4. Rank Shift

- Rating overestimates vs RAPM: **25**
- Rating underestimates vs RAPM: **30**

---

## 5. Teammate Influence

Full results in `teammate_influence/report_teammate_influence.md`
