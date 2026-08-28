# Οδηγός: δωρεές + διάδοση (για τον Κωνσταντίνο)

Όλα εδώ θέλουν συνολικά ~30 λεπτά δικού σου χρόνου. Τα υπόλοιπα είναι έτοιμα.

## Α. Δωρεές — στήσιμο σε 10 λεπτά

Πρότεινα τρεις δρόμους, με σειρά ευκολίας:

### 1. Ko-fi (το πιο γρήγορο — ξεκίνα από εδώ)
1. Πήγαινε στο **ko-fi.com** → Sign up (δωρεάν, χωρίς προμήθεια στο βασικό).
2. Όνομα σελίδας: `kosred` (ή `neoethos`). Σύνδεσε **PayPal** ή Stripe.
3. Γράψε 2 γραμμές: «Χτίζω το NeoEthos, ανοιχτό εργαλείο trading για μικρούς
   traders. Οι δωρεές πάνε σε ρεύμα, hardware και τη συνδρομή AI που το χτίζει.»
4. Άνοιξε το αρχείο `.github/FUNDING.yml` στο repo (μπορείς από το site του
   GitHub → Edit) και ξε-σχολίασε τη γραμμή `ko_fi: kosred` (βάλε το δικό σου
   username). Commit. → Εμφανίζεται κουμπί **Sponsor** πάνω-πάνω στο repo.

### 2. GitHub Sponsors (πιο «επίσημο», 0% προμήθεια)
1. **github.com/sponsors** → Join the waitlist / Set up sponsorship.
2. Θέλει Stripe Connect (λειτουργεί στην Ελλάδα) + φορολογικά στοιχεία.
   Παίρνει 1-2 μέρες έγκριση.
3. Μετά ξε-σχολίασε το `github: [kosred]` στο FUNDING.yml.

### 3. Liberapay (φιλικό στο open source, ΕΕ)
1. **liberapay.com** → λογαριασμός → σύνδεση τράπεζας/PayPal.
2. Ξε-σχολίασε `liberapay: kosred`.

> Συμβουλή: ΜΗΝ υποσχεθείς ανταλλάγματα με «κέρδη» ή σήματα στους δωρητές —
> δωρεά για την ανάπτυξη του λογισμικού, τίποτα επενδυτικό (δες
> docs/p2p-mesh-design §4 για το γιατί).

## Β. Διάδοση — 15 λεπτά, έτοιμα κείμενα

Στον φάκελο `docs/announcements/` υπάρχουν **έτοιμα προς επικόλληση** κείμενα:

| Αρχείο | Πού το ποστάρεις |
|---|---|
| `reddit-algotrading.md` | reddit.com/r/algotrading → New post (κείμενο) |
| `hackernews.md` | news.ycombinator.com → submit → «Show HN» |
| `twitter-thread.md` | X/Twitter — νήμα 5 tweets |

Σειρά που προτείνω: **r/algotrading πρώτα** (η πιο σχετική κοινότητα, δέχεται
open-source projects), μετά Show HN (Τρίτη-Πέμπτη, 15:00-17:00 ώρα Ελλάδας =
πρωί ΗΠΑ), μετά το νήμα στο X. Απάντα στα σχόλια την πρώτη ώρα — αυτό καθορίζει
αν θα ανέβει. Να είσαι αυτός που είσαι: ειλικρινής για το τι κάνει ΚΑΙ τι δεν
υπόσχεται. Αυτό ξεχωρίζει από τα χίλια «bot που πλουτίζεις».

Άλλα μέρη όταν βρεις χρόνο: r/Forex (πιο αυστηροί με self-promo — δώσε βάρος
στο «open source, no signals sold»), Rust subreddit (r/rust αγαπάει
production Rust projects — γωνία: «pure-Rust trading engine, Tauri, Burn»),
Discord servers για algotrading.

## Γ. Τι έκανα ήδη εγώ (μην το ξανακάνεις)

- README: ενότητα **Support** + σύνδεσμοι σε PRIVACY/PRINCIPLES.
- `PRIVACY.md` (πλήρης έλεγχος δικτύου/δεδομένων), `PRINCIPLES.md` (το ήθος),
  `CONTRIBUTING.md`.
- `BUILDING.md`: πίνακας hardware (τι χρειάζεται ανά χρήση, GPU οδηγίες).
- GitHub: περιγραφή repo + topics για να βγαίνει στις αναζητήσεις.
- Release v0.5.2 δημοσιευμένο με installers.
