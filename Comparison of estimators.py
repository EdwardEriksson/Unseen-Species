#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TOTAL distinct-token prediction on either

    (a) Hamlet from Project Gutenberg,
    (b) SNAP CollegeMsg (users in messages),
    (c) genome/genes (SAM -> per-fragment base sets),
    (d) Butterfly subset (days × species at one site).

Unified viewpoint:
    We work with a sequence of *groups*:
        groups = [g_0, g_1, g_2, ...]
    where each g_i is an iterable of 'tokens' seen at step i.

    - Hamlet:       g_i = { word_i }
    - CollegeMsg:   g_i = { src_i, dst_i }
    - Genes:        g_i = set of base indices in one interval
    - Butterflies:  g_i = set of species observed that day

Prediction model (unchanged at top level):
    total_hat = S_n + increment_hat
"""

import os
import re
import math
import random
import urllib.request
import gzip
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import json
import pdfplumber
import pandas as pd

# === Local imports (your existing modules) ===
from QSIP_solver_neo import solve_qsip_adaptive_scipy
from Estimators_neo import (
    predict_new_SGT,
    predict_new_trivial,
    predict_new_ratio_multiplicative,
    predict_new_favaro,
    predict_new_padegt_plain,
    predict_new_padegt_boot,
    predict_new_optimistic,
    predict_new_chebyshev_increment,
)

# ============================================================
# CONFIG
# ============================================================

# choose: "genes","hamlet","college_msg","butterflies","synthetic_uniform","snap_email","mtg"
DATA_SOURCE = "mtg"

# CollegeMsg
COLLEGEMSG_URL   = "https://snap.stanford.edu/data/CollegeMsg.txt.gz"
LOCAL_COLLEGEMSG = "./CollegeMsg.txt.gz"

# Butterfly subset (Corbet-style days × species)
BUTTERFLY_CSV_PATH = r"C:\Users\Edwar\Downloads\Butterflydata\0169126-240321170329656.csv"
BUTTERFLY_ROUND_COORD = 2
BUTTERFLY_SITE_LAT = -5.25
BUTTERFLY_SITE_LON = 145.27
BUTTERFLY_START_DATE = "2008-05-30"
BUTTERFLY_END_DATE   = "2008-07-29"

# Estimator toggles
INCLUDE_QSIP    = True
INCLUDE_SGT     = True
INCLUDE_TRIV    = True
INCLUDE_RATIO   = True
INCLUDE_FAVARO  = True
INCLUDE_PADEGT_PLAIN = True
INCLUDE_PADEGT_BOOT  = False
INCLUDE_OPTIMISTIC   = False
INCLUDE_CHEBYSHEV    = True

# Padé parameters
PADE_P = 2
PADE_Q = 3

# QSIP parameters
K_QSIP = 10

# Chebyshev parameters
CHEB_K = None  # will set later
c_0 = 0.45
c_1 = 0.5

# Experiment knobs
NUM_SHUFFLES = 10
TRUE_ORDER = False
NUM_N        = 50
ALPHA        = 1.0
SEED         = 1

GROUP_SUBSAMPLE_K = 1
FILTER_EMPTY_INTERVALS = True


# Synthetic uniform config
SYNTHETIC_N = 2000   # number of samples
SYNTHETIC_M = 5000   # number of symbols

#Mtg param
N_boosters = 14

# Result saving (for later multi-run plotting)
SAVE_RESULTS = True
RESULTS_DIR  = "./comparison_results"
RUN_TAG      = "run"   # change this between experiments if you like



CHRONO_DATASETS = {"hamlet", "college_msg", "butterflies", "snap_email"}

# ============================================================
# Hamlet loader -> groups
# ============================================================

def load_hamlet_words():
    url = "https://www.gutenberg.org/files/1524/1524-0.txt"
    with urllib.request.urlopen(url) as resp:
        raw = resp.read().decode("utf-8")
    words = re.findall(r"[A-Za-z]+", raw.lower())
    print(f"[Hamlet] total tokens={len(words)} | unique={len(set(words))}")
    return words

def groups_from_hamlet(words):
    # each occurrence is its own group
    return [{w} for w in words]


# ============================================================
# Synthetic uniform loader -> groups
# ============================================================

def groups_from_synthetic_uniform(N, M):
    """
    Draw N samples i.i.d. from the uniform distribution on {0, 1, ..., M-1}
    and return them as singleton groups [{x_0}, {x_1}, ...].
    """
    samples = np.random.randint(0, M, size=N)
    groups = [{int(x)} for x in samples]
    return groups, samples


# ============================================================
# CollegeMsg loader -> groups
# ============================================================

def ensure_collegemsg(local_path=LOCAL_COLLEGEMSG, url=COLLEGEMSG_URL):
    if not os.path.exists(local_path):
        print(f"[CollegeMsg] Downloading from {url}")
        urllib.request.urlretrieve(url, local_path)
        print("[CollegeMsg] Download done")
    else:
        print(f"[CollegeMsg] Using existing {local_path}")
    return local_path

def groups_from_collegemsg(local_path=LOCAL_COLLEGEMSG):
    import pandas as pd
    df = pd.read_csv(
        local_path,
        sep=' ',
        names=['src', 'dst', 't'],
        compression='gzip'
    ).sort_values('t').reset_index(drop=True)

    groups = []
    app = groups.append
    for row in df.itertuples(index=False):
        app({int(row.src), int(row.dst)})
    print(f"[CollegeMsg] messages={len(groups)} | distinct users={len({u for g in groups for u in g})}")
    return groups


# ============================================================
# Butterfly loader -> groups (days × species)
# ============================================================

def load_butterfly_day_sets(
    path=BUTTERFLY_CSV_PATH,
    round_coord=BUTTERFLY_ROUND_COORD,
    site_lat=BUTTERFLY_SITE_LAT,
    site_lon=BUTTERFLY_SITE_LON,
    start_date_str=BUTTERFLY_START_DATE,
    end_date_str=BUTTERFLY_END_DATE,
):
    """
    Load the GBIF butterfly export, filter to one site and time window,
    and return a list of sets: one set of species per day.
    """
    import pandas as pd

    print("[butterflies] reading GBIF export...")
    df = pd.read_csv(path, sep=None, engine="python")
    print(f"[butterflies] raw rows={len(df):,} cols={len(df.columns)}")

    # Parse dates
    df["eventDate_parsed"] = pd.to_datetime(df["eventDate"], errors="coerce")
    df = df[df["eventDate_parsed"].notna()]

    # Keep only rows with coordinates
    df = df[df["decimalLatitude"].notna() & df["decimalLongitude"].notna()]

    # Round coords and define site
    df["lat_round"] = df["decimalLatitude"].round(round_coord)
    df["lon_round"] = df["decimalLongitude"].round(round_coord)

    start_date = pd.to_datetime(start_date_str)
    end_date   = pd.to_datetime(end_date_str)

    mask_site = (df["lat_round"] == site_lat) & (df["lon_round"] == site_lon)
    mask_time = (df["eventDate_parsed"] >= start_date) & (df["eventDate_parsed"] <= end_date)

    df_burst = df[mask_site & mask_time].copy()

    print(
        f"[butterflies] rows at site {(site_lat, site_lon)} in "
        f"window {start_date.date()}–{end_date.date()}: {len(df_burst)}"
    )

    species_col = "species" if "species" in df_burst.columns else "scientificName"
    print(f"[butterflies] distinct species in burst: {df_burst[species_col].nunique()}")

    # Keep only useful columns
    keep = [
        "eventDate_parsed",
        species_col,
        "decimalLatitude",
        "decimalLongitude",
    ]

    for extra in ["locality", "stateProvince", "countryCode", "recordedBy"]:
        if extra in df_burst.columns:
            keep.append(extra)

    dense_df_clean = df_burst[keep].copy()
    dense_df_clean["date"] = dense_df_clean["eventDate_parsed"].dt.date

    # Corbet-style log: (date, species, count)
    log_df = (
        dense_df_clean
        .groupby(["date", species_col])
        .size()
        .reset_index(name="count")
    )

    print(f"[butterflies] log entries={len(log_df)} | "
          f"distinct species={log_df[species_col].nunique()} | "
          f"total individuals={log_df['count'].sum()}")

    # Build list of sets: one set of species per day
    day_sets = []
    day_order = sorted(log_df["date"].unique())  # chronological

    for day in day_order:
        species_that_day = set(
            log_df.loc[log_df["date"] == day, species_col]
        )
        day_sets.append(species_that_day)

    print(f"[butterflies] number of days with observations: {len(day_sets)}")
    if day_sets:
        example_day = next(iter(day_sets))
        print(f"[butterflies] example day set size={len(example_day)}")

    dataset_label = (
        f"Butterflies HK site "
        f"({len(day_sets)} days, {log_df[species_col].nunique()} species)"
    )
    return day_sets, dataset_label


# ============================================================
# MTG boosters loader -> groups (each booster = set of cards)
# ============================================================

from pathlib import Path

def parse_mtg_boosters(path):
    """
    Parse boosters_unfiltered.txt located at Desktop with format:

        === All Packs ===
        Pack 1 (15 cards):
        Card A
        Card B
        ...
        Pack 2 (15 cards):
        ...

    Returns: list of sets, one per booster.
    """
    packs = []
    current = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Pack ") and "cards" in line:
                if current is not None:
                    packs.append(set(current))
                current = []
            elif line and not line.startswith("==="):
                if current is not None:
                    current.append(line)
    if current is not None:
        packs.append(set(current))
    return packs

def load_mtg_groups():
    boosters_file = Path.home() / "Desktop" / "boosters_unfiltered.txt"
    groups = parse_mtg_boosters(boosters_file)
    groups = groups[:N_boosters]
    print(f"[MTG] Loaded {len(groups)} boosters.")
    distinct_cards = len({c for g in groups for c in g})
    print(f"[MTG] Distinct cards: {distinct_cards}")
    return groups, f"MTG boosters ({len(groups)} packs)"


#Email loader

def groups_from_snap_email(path):
    """
    Parse SNAP email-Eu-core-temporal.txt(.gz) into chronologically ordered groups.

    Input format (per SNAP):
        src dst time

    Multiple lines can correspond to the same email if:
        (same src, same time, different dst)

    Output:
        groups = [ {sender, recipient1, recipient2, ...}, ... ]
    in strictly increasing time order.
    """
    import gzip

    # Auto-handle gzipped or plain text
    if str(path).endswith(".gz"):
        opener = lambda p: gzip.open(p, "rt")
    else:
        opener = lambda p: open(p, "r")

    groups = []
    with opener(path) as f:
        current_time = None
        current_sender = None
        current_set = None

        for line in f:
            line = line.strip()
            if not line:
                continue
            s, d, t = line.split()
            s = int(s)
            d = int(d)
            t = int(t)

            # If we are starting a new email event:
            if t != current_time or s != current_sender:
                # Finish previous group
                if current_set is not None:
                    groups.append(current_set)

                # Start new group
                current_time = t
                current_sender = s
                current_set = {s, d}
            else:
                # Same email event → accumulate another recipient
                current_set.add(d)

        # Append final group
        if current_set is not None:
            groups.append(current_set)

    print(f"[SNAP email] Parsed {len(groups)} email groups from {path}")
    distinct_users = len({u for g in groups for u in g})
    print(f"[SNAP email] Distinct users: {distinct_users}")
    return groups

#Bacteria Loader

def groups_from_bacteria_pdf(pdf_path):
    import pdfplumber
    import pandas as pd

    # ----- STEP 1: load all table rows from all pages -----
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                rows.extend(table)

    df = pd.DataFrame(rows)

    # ----- STEP 2: drop junk outer columns (always present) -----
    df = df.drop(df.columns[:3], axis=1)
    df = df.drop(df.columns[-3:], axis=1)

    # ----- STEP 3: remove rows that are completely empty -----
    df = df.dropna(how="all").reset_index(drop=True)

    # ----- STEP 4: find the REAL header row -----
    # It is the first row that contains "SubA"/"SubB"/etc.
    header_idx = None
    for i, row in df.iterrows():
        if any(isinstance(x, str) and x.startswith("Sub") for x in row):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find header row with SubA/SubB/...")

    # Assign that row as header
    df.columns = df.iloc[header_idx]
    df = df.iloc[header_idx+1:].reset_index(drop=True)

    # ----- STEP 5: clean header names -----
    df.columns = [
        "species" if (c is None or str(c).strip()=="") else str(c).strip()
        for c in df.columns
    ]

    # Deduplicate header names
    seen = {}
    clean_cols = []
    for c in df.columns:
        if c not in seen:
            seen[c] = 1
            clean_cols.append(c)
        else:
            seen[c] += 1
            clean_cols.append(f"{c}_{seen[c]}")
    df.columns = clean_cols

    # ----- STEP 6: keep only rows with real species names -----
    df = df[df["species"].notna()].reset_index(drop=True)

    # ----- STEP 7: identify subject columns -----
    subject_cols = [c for c in df.columns if c.startswith("Sub")]

    # ----- STEP 8: build subject → bacteria sets -----
    subject_to_bacteria = {sub: set() for sub in subject_cols}

    for _, row in df.iterrows():
        species = row["species"]
        for sub in subject_cols:
            cell = row[sub]
            if pd.notna(cell) and str(cell).strip() not in ["", "0"]:
                subject_to_bacteria[sub].add(species)

    # Convert to list of sets in subject order
    groups = [subject_to_bacteria[sub] for sub in subject_cols]
    label = f"Bacteria table ({len(groups)} subjects)"

    return groups, label



# ============================================================
# Genome helpers
# ============================================================

# Genome / genes config
GENES_INTERVAL_MANIFEST = r"C:\Users\Edwar\toy_align\sharded_intervals\manifest.json"
# If you change --out-dir in Shardmaker.py, update this path accordingly.

import gzip, pickle, json  # (json already imported above, but harmless)

def iterate_intervals_from_manifest(manifest_path):
    base = os.path.dirname(os.path.abspath(manifest_path))
    with open(manifest_path, "r", encoding="utf-8") as f:
        mani = json.load(f)
    for ent in mani["shards"]:
        shard_file = os.path.join(base, ent["path"])
        with gzip.open(shard_file, "rb") as f:
            obj = pickle.load(f)
        for (start, end) in obj["intervals"]:
            yield (start, end), mani  # interval and (shared) meta


def compute_phi_from_intervals(manifest_path):
    """
    Efficiently compute φ_k (count of bases covered exactly k times) from (start,end) intervals
    using a difference-array line sweep. Returns: phi (list where phi[k] is count), Sn, total_occ.
    """
    intervals = []
    contig_len = None
    with open(manifest_path, "r", encoding="utf-8") as f:
        mani = json.load(f)
    contig_len = int(mani["contig_len"])

    # difference array: +1 at start, -1 at end
    diff = [0]*(contig_len + 1)  # end-exclusive, safe to touch diff[contig_len]
    total_occ = 0
    for ent in mani["shards"]:
        shard_file = os.path.join(os.path.dirname(manifest_path), ent["path"])
        with gzip.open(shard_file, "rb") as f:
            obj = pickle.load(f)
        for (a, b) in obj["intervals"]:
            if a < 0:  # (-1,-1) placeholders if you used --keep-empty
                continue
            if a >= b:
                continue
            a = max(0, min(a, contig_len))
            b = max(0, min(b, contig_len))
            diff[a] += 1
            diff[b] -= 1
            total_occ += (b - a)

    # prefix-sum to coverage array; then histogram to φ
    phi = {}
    Sn = 0
    c = 0
    for i in range(contig_len):
        c += diff[i]
        if c > 0:
            Sn += 1
        phi[c] = phi.get(c, 0) + 1  # includes c=0 bucket too

    # Convert to dense list starting at k=0 (phi[0]=uncovered)
    max_k = max(phi) if phi else 0
    phi_list = [phi.get(k, 0) for k in range(max_k+1)]
    return phi_list, Sn, total_occ, mani


def load_intervals_from_manifest_as_list(manifest_path):
    """Load all (start,end) intervals from a sharded manifest into a flat list.

    This is the interval analogue of building `groups`:
        groups  <->  intervals
        len(groups)  <->  number of intervals
    We do *not* form per-base sets, we just keep the integer endpoints.
    """
    base = os.path.dirname(os.path.abspath(manifest_path))
    with open(manifest_path, "r", encoding="utf-8") as f:
        mani = json.load(f)
    contig_len = int(mani.get("contig_len", 0))

    intervals = []
    for ent in mani["shards"]:
        shard_file = os.path.join(base, ent["path"])
        with gzip.open(shard_file, "rb") as f:
            obj = pickle.load(f)
        for (a, b) in obj["intervals"]:
            # Explicit "empty" intervals from Shardmaker: unaligned reads
            # keep them as (-1, -1) so we can optionally filter later.
            if a < 0 and b < 0:
                intervals.append((a, b))
                continue

            # Discard other malformed intervals
            if a < 0 or a >= b:
                continue

            a = max(0, min(a, contig_len))
            b = max(0, min(b, contig_len))
            intervals.append((a, b))
    return intervals, contig_len, mani


def compute_phi_prefix_from_intervals(intervals, n, contig_len):
    """Interval analogue of compute_phi_prefix_from_groups.

    intervals : list of (start, end) pairs (end-exclusive)
    n         : use the first n intervals
    contig_len: length of the reference/contig

    Returns a dense phi array with phi[k] = #bases covered exactly k times.
    """
    # Difference array over [0, contig_len)
    diff = [0] * (contig_len + 1)
    for (a, b) in intervals[:n]:
        if a < 0 or a >= b:
            continue
        a = max(0, min(a, contig_len))
        b = max(0, min(b, contig_len))
        diff[a] += 1
        diff[b] -= 1

    # Prefix-sum to coverage, then histogram
    cov = 0
    phi = {}
    for i in range(contig_len):
        cov += diff[i]
        phi[cov] = phi.get(cov, 0) + 1

    if not phi:
        return np.zeros(1, dtype=int)

    max_k = max(phi)
    phi_list = np.zeros(max_k + 1, dtype=int)
    for k, v in phi.items():
        phi_list[k] = v
    return phi_list


def win_to_wsl(path: str) -> str:
    path = os.path.abspath(path)
    drive = path[0].lower()
    tail = path[2:].replace("\\", "/").lstrip("/")
    return f"/mnt/{drive}/{tail}"

def run_cmd_capture(cmd_list, desc="command"):
    proc = subprocess.run(cmd_list, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"=== STDOUT ({desc}) ===")
        print(proc.stdout)
        print(f"=== STDERR ({desc}) ===")
        print(proc.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd_list,
                                            output=proc.stdout, stderr=proc.stderr)
    return proc.stdout, proc.stderr

def make_subset_fastq(src_fastq, dst_fastq, n_reads, force=False):
    if (not force) and os.path.exists(dst_fastq):
        print(f"[genes] reusing existing subset {dst_fastq}")
        return
    os.makedirs(os.path.dirname(dst_fastq), exist_ok=True)
    cnt = 0
    with open(src_fastq, "r") as fin, open(dst_fastq, "w") as fout:
        while cnt < n_reads:
            h = fin.readline()
            if not h:
                break
            s = fin.readline()
            p = fin.readline()
            q = fin.readline()
            if not q:
                break
            fout.write(h); fout.write(s); fout.write(p); fout.write(q)
            cnt += 1
    print(f"[genes] wrote {cnt} reads to {dst_fastq}")

def make_small_fasta(src_fasta, dst_fasta, max_bases=100_000, force=False):
    if (not force) and os.path.exists(dst_fasta):
        print(f"[genes] reusing existing slice {dst_fasta}")
        return
    name = None
    seq_chunks = []
    total = 0
    with open(src_fasta, "r") as f:
        for line in f:
            if line.startswith(">"):
                if name is None:
                    name = line[1:].strip().split()[0]
                else:
                    break
            else:
                if name is not None:
                    s = line.strip()
                    need = max_bases - total
                    if need <= 0:
                        break
                    seq_chunks.append(s[:need])
                    total += min(len(s), need)
                    if total >= max_bases:
                        break
    if name is None or total == 0:
        raise RuntimeError("[genes] could not slice FASTA")

    os.makedirs(os.path.dirname(dst_fasta), exist_ok=True)
    with open(dst_fasta, "w") as out:
        out.write(f">{name}\n")
        seq = "".join(seq_chunks)
        for i in range(0, len(seq), 60):
            out.write(seq[i:i+60] + "\n")
    print(f"[genes] wrote {total} bases of {name} to {dst_fasta}")

def get_first_contig_and_len(fasta_path):
    name = None
    length = 0
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                if name is None:
                    name = line[1:].strip().split()[0]
                else:
                    break
            else:
                if name is not None:
                    length += len(line.strip())
    return name, length

def cigar_ref_len(cigar: str) -> int:
    ref_consume = set("MDN=X")
    num = ""
    total = 0
    for ch in cigar:
        if ch.isdigit():
            num += ch
        else:
            if ch in ref_consume:
                total += int(num)
            num = ""
    return total

def run_minimap2_wsl(ref_fasta_win, fastq_win, sam_out_win,
                     wsl_exe=None, minimap2_opts=("-x", "sr", "-a")):
    if wsl_exe is None:
        if "PROCESSOR_ARCHITEW6432" in os.environ:
            wsl_exe = r"C:\Windows\Sysnative\wsl.exe"
        else:
            wsl_exe = r"C:\Windows\System32\wsl.exe"
    ref_wsl = win_to_wsl(ref_fasta_win)
    fq_wsl  = win_to_wsl(fastq_win)
    cmd = [wsl_exe, "minimap2", *minimap2_opts, ref_wsl, fq_wsl]
    print("[genes] running minimap2…")
    sam_txt, _ = run_cmd_capture(cmd, desc="minimap2")
    os.makedirs(os.path.dirname(sam_out_win), exist_ok=True)
    with open(sam_out_win, "w", newline="\n") as f:
        f.write(sam_txt)
    print(f"[genes] wrote SAM to {sam_out_win}")

def groups_from_genome_sam(sam_path, contig, contig_len,
                           max_fragments=None,
                           drop_gappy=True):
    groups = []
    current_qname = None
    any_aligned = False
    fragments_seen = 0

    with open(sam_path, "r") as f:
        for line in f:
            if line.startswith("@"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue

            qname = parts[0]
            flag  = int(parts[1])
            rname = parts[2]
            pos   = int(parts[3])
            cigar = parts[5]

            if qname != current_qname:
                if current_qname is not None and not any_aligned:
                    groups.append(set())
                fragments_seen += 1
                current_qname = qname
                any_aligned = False
                if (max_fragments is not None) and (fragments_seen > max_fragments):
                    break

            if flag & 0x100 or flag & 0x800:
                continue
            if rname == "*" or cigar == "*":
                continue
            if drop_gappy and ("N" in cigar or "D" in cigar):
                continue
            if rname != contig:
                continue

            any_aligned = True
            ref_len = cigar_ref_len(cigar)
            start = pos - 1
            end = min(start + ref_len, contig_len)
            bases = set(range(start, end))

            if len(groups) < fragments_seen:
                groups.append(bases)
            else:
                groups[-1].update(bases)

    if current_qname is not None and not any_aligned:
        groups.append(set())

    print(f"[genes] built {len(groups)} groups from SAM")
    return groups

def load_preprocessed_groups_pickle_gz(path):
    with gzip.open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["groups"], obj.get("meta", {})

def load_genes_groups(
    ref_full,
    ref_slice,
    fastq_path,
    sam_path,
    n_reads=10**4,
    max_bases_in_slice=100_000,
    force_reslice=True,
    force_resubset=True,
    max_fragments=None,
):
    make_subset_fastq(fastq_path, fastq_path + f".subset_{n_reads}.fastq",
                      n_reads=n_reads, force=force_resubset)
    subset_fastq = fastq_path + f".subset_{n_reads}.fastq"

    make_small_fasta(ref_full, ref_slice,
                     max_bases=max_bases_in_slice, force=force_reslice)

    run_minimap2_wsl(ref_slice, subset_fastq, sam_path)

    contig, contig_len = get_first_contig_and_len(ref_slice)
    print(f"[genes] contig={contig} len={contig_len}")

    groups = groups_from_genome_sam(
        sam_path,
        contig,
        contig_len,
        max_fragments=max_fragments,
        drop_gappy=True,
    )
    dataset_label = f"Genome slice {os.path.basename(ref_slice)} ({contig_len} bases)"
    return groups, dataset_label, contig_len


# ============================================================
# Utility
# ============================================================

def abs_pct_err(pred, truth):
    return 100.0 * abs(pred - truth) / truth if truth > 0 else float("nan")

def S_n_from_phi(phi):
    # phi[0] unused
    return float(sum(phi[1:]))

def compute_phi_prefix_from_groups(groups, n):
    """
    groups: list of iterables of tokens
    n: number of groups to take
    returns phi only (no t): phi[k] = #tokens seen exactly k times in first n groups
    """
    counts = Counter()
    for g in groups[:n]:
        for tok in g:
            counts[tok] += 1

    if not counts:
        return np.zeros(1, dtype=int)

    freq_counts = Counter(counts.values())
    max_freq = max(freq_counts)
    phi = np.zeros(max_freq + 1, dtype=int)
    for k, v in freq_counts.items():
        phi[k] = v
    return phi

def apply_estimator(H, phi):
    m = max(len(H), len(phi))
    H_pad = np.zeros(m); H_pad[:len(H)] = H
    phi_pad = np.zeros(m); phi_pad[:len(phi)] = phi
    return float(np.dot(H_pad, phi_pad))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if SEED is not None:
        np.random.seed(SEED)
        random.seed(SEED)

    # 1) load dataset
    if DATA_SOURCE == "hamlet":
        print("[Dataset] Using Hamlet (Gutenberg 1524)")
        words = load_hamlet_words()
        groups = groups_from_hamlet(words)

        if GROUP_SUBSAMPLE_K > 1:
            groups = groups[::GROUP_SUBSAMPLE_K]
            print(f"[Subsample] Keeping every {GROUP_SUBSAMPLE_K}-th group -> {len(groups)} groups")

        dataset_label = "Hamlet (Gutenberg 1524)"

        # Global counts from (possibly subsampled) groups
        G_total = len(groups)                              # number of groups (steps)
        T_total = sum(len(g) for g in groups)              # total token-occurrences
        V_true  = len({tok for g in groups for tok in g})  # distinct tokens
        print(f"Total groups: {G_total} | Total token-occurrences: {T_total} | Distinct tokens: {V_true}")

    elif DATA_SOURCE == "college_msg":
        print("[Dataset] Using CollegeMsg (SNAP)")
        ensure_collegemsg()
        groups = groups_from_collegemsg()

        if GROUP_SUBSAMPLE_K > 1:
            groups = groups[::GROUP_SUBSAMPLE_K]
            print(f"[Subsample] Keeping every {GROUP_SUBSAMPLE_K}-th group -> {len(groups)} groups")

        dataset_label = "CollegeMsg (SNAP)"

        # Counts from (possibly subsampled) groups
        G_total = len(groups)
        T_total = sum(len(g) for g in groups)
        V_true  = len({tok for g in groups for tok in g})
        print(f"Total groups: {G_total} | Total token-occurrences: {T_total} | Distinct tokens: {V_true}")

    elif DATA_SOURCE == "butterflies":
        print("[Dataset] Using Butterfly subset (days × species)")
        groups, dataset_label = load_butterfly_day_sets()

        if GROUP_SUBSAMPLE_K > 1:
            groups = groups[::GROUP_SUBSAMPLE_K]
            print(f"[Subsample] Keeping every {GROUP_SUBSAMPLE_K}-th day -> {len(groups)} days")

        G_total = len(groups)
        T_total = sum(len(g) for g in groups)               # species-day incidences
        V_true  = len({tok for g in groups for tok in g})   # distinct species

        print(
            f"Total days: {G_total} | "
            f"Total species-day incidences: {T_total} | "
            f"Distinct species: {V_true}"
        )
        
    elif DATA_SOURCE == "snap_email":
        print("[Dataset] Using SNAP email-Eu-core-temporal")
    
        path = r"C:\Users\Edwar\Downloads\email-Eu-core-temporal.txt.gz"
        groups = groups_from_snap_email(path)
    
        if GROUP_SUBSAMPLE_K > 1:
            groups = groups[::GROUP_SUBSAMPLE_K]
    
        dataset_label = "SNAP email-Eu-core-temporal"
    
        G_total = len(groups)
        T_total = sum(len(g) for g in groups)
        V_true  = len({tok for g in groups for tok in g})
    
        print(
            f"Total groups: {G_total} | "
            f"Total token-occurrences: {T_total} | "
            f"Distinct tokens: {V_true}"
        )
    elif DATA_SOURCE == "bacteria":
        print("[Dataset] Using bacteria PDF table")
    
        pdf_path = r"C:\Users\Edwar\Downloads\BacteriaTable.pdf"
        groups, dataset_label = groups_from_bacteria_pdf(pdf_path)
    
        if GROUP_SUBSAMPLE_K > 1:
            groups = groups[::GROUP_SUBSAMPLE_K]
            print(f"[Subsample] Keeping every {GROUP_SUBSAMPLE_K}-th subject -> {len(groups)} groups")
    
        G_total = len(groups)
        T_total = sum(len(g) for g in groups)
        V_true  = len({tok for g in groups for tok in g})

        print(
            f"Total subjects: {G_total} | "
            f"Total species-occurrences: {T_total} | "
            f"Distinct species: {V_true}"
        )
        for i, g in enumerate(groups):
            print(i, len(g))
        print("")
        for i, g in enumerate(groups[:3]):
            print("Group", i, ":", sorted(g)[:20])





    elif DATA_SOURCE == "genes":
        print("[Dataset] Using preprocessed genes (intervals manifest)")
        MANIFEST = GENES_INTERVAL_MANIFEST

        # Interval analogue of `groups`: each (start,end) is one group of bases.
        intervals, contig_len, meta = load_intervals_from_manifest_as_list(MANIFEST)

        # Optional: drop (-1,-1) intervals; load_intervals already skips a<0, but keep for safety
        if FILTER_EMPTY_INTERVALS:
            before = len(intervals)
            intervals = [iv for iv in intervals if iv != (-1, -1)]
            removed = before - len(intervals)
            print(f"[Genes] Dropped {removed} empty intervals (-1,-1); keeping {len(intervals)}")

        # Optional: subsample intervals (keep every k-th)
        if GROUP_SUBSAMPLE_K > 1:
            intervals = intervals[::GROUP_SUBSAMPLE_K]
            print(f"[Subsample] Keeping every {GROUP_SUBSAMPLE_K}-th interval -> {len(intervals)} intervals")

        dataset_label = (
            f"Genome slice (preprocessed: {meta.get('contig','?')}, "
            f"{contig_len} bases)"
        )
        
        # Diagnostic: shard read-counts vs manifest total (still on full manifest)
        sum_reads = int(sum(ent.get("reads", 0) for ent in meta.get("shards", [])))
        print(
            f"[check] manifest total_reads={meta.get('total_reads')} | "
            f"sum(shard.reads)={sum_reads}"
        )

        # Truth and global quantities are now computed from the
        # *actual* intervals used by the experiment, not from the
        # full manifest.
        G_total = len(intervals)  # number of interval-groups

        # φ over the whole (filtered + subsampled) interval list
        phi_full = compute_phi_prefix_from_intervals(intervals, G_total, contig_len)

        # True #covered bases
        S_n_full = S_n_from_phi(phi_full)
        V_true   = int(S_n_full)

        # Total coverage counts = sum_k k * φ_k
        total_occ = int(sum(k * int(phi_full[k]) for k in range(1, len(phi_full))))
        T_total   = total_occ

        print(
            f"[Genes] contig={meta.get('contig','?')} | "
            f"contig_len={contig_len} | total_reads={meta.get('total_reads')} | "
            f"num_intervals={G_total} | V_true={V_true} | T_total={T_total}"
        )
    elif DATA_SOURCE == "synthetic_uniform":
        print("[Dataset] Using synthetic uniform")
        groups, samples = groups_from_synthetic_uniform(SYNTHETIC_N, SYNTHETIC_M)
    
        if GROUP_SUBSAMPLE_K > 1:
            groups = groups[::GROUP_SUBSAMPLE_K]
            print(f"[Subsample] Keeping every {GROUP_SUBSAMPLE_K}-th group -> {len(groups)} groups")
    
        dataset_label = f"Synthetic uniform (N={SYNTHETIC_N}, M={SYNTHETIC_M})"
    
        G_total = len(groups)
        T_total = sum(len(g) for g in groups)
        V_true  = len({tok for g in groups for tok in g})
        print(f"Total groups: {G_total} | Total token-occurrences: {T_total} | Distinct tokens: {V_true}")
    
        # *** IMPORTANT: modify DATA_SOURCE only for filename ***
        DATA_SOURCE = f"synthetic_uniform_M{SYNTHETIC_M}"
    elif DATA_SOURCE == "mtg":
        print("[Dataset] Using MTG booster packs")
        groups, dataset_label = load_mtg_groups()

        if GROUP_SUBSAMPLE_K > 1:
            groups = groups[::GROUP_SUBSAMPLE_K]
            print(f"[Subsample] Keeping every {GROUP_SUBSAMPLE_K}-th group -> {len(groups)} boosters")

        G_total = len(groups)
        T_total = sum(len(g) for g in groups)          # total card-occurrences
        V_true  = len({tok for g in groups for tok in g})

        print(
            f"Total boosters: {G_total} | "
            f"Total card-occurrences: {T_total} | "
            f"Distinct cards: {V_true}"
        )

    else:
        raise ValueError("DATA_SOURCE must be 'genes', 'hamlet', 'college_msg', or 'butterflies'")

    # 3) n-grid over GROUPS / READS / INTERVALS
    if DATA_SOURCE == "butterflies":
        # small dataset: start earlier
        n_min = 1
    
    else:
        n_min = 1

    # up to ~half the dataset, but never beyond G_total-1
    if G_total <= 1:
        raise RuntimeError("Not enough groups to run the experiment.")

    n_max_half = G_total // 2
    n_max = max(n_min, min(G_total - 1, n_max_half))

    u = np.linspace(0, 0.999, NUM_N)
    n_vals = (n_min + (n_max - n_min) * (u ** ALPHA)).astype(int)
    n_vals = np.clip(n_vals, n_min, n_max)

    # percentage progress values (same formula for all datasets)
    perc_vals = 100.0 * n_vals / float(G_total)

    # 4) QSIP precompute
    H_cache = {}
    if INCLUDE_QSIP:
        print("Precomputing H* with QSIP solver…")
        x0 = np.zeros(K_QSIP)
        for n in n_vals:
            if n == 0:
                continue
            if DATA_SOURCE == "genes":
                phi_n = compute_phi_prefix_from_intervals(intervals, n, contig_len)
            else:
                phi_n = compute_phi_prefix_from_groups(groups, n)
            r_n = (G_total - n) / n
            if r_n > 1.0:
                H_full = solve_qsip_adaptive_scipy(
                    n, r_n, K=K_QSIP,
                    p_init=161, q_init=161,
                    eps=1e-10, tau1=1e-4, tau2=1e-4,
                    outer_iters=4, x0=x0,
                    maxiter_inner=600, refine_mode="halve"
                )
                x0 = H_full[1:].copy()
                H_cache[n] = H_full

    # 5) experiment
    mean = {
        k: np.zeros_like(n_vals, float)
        for k in ["qsip", "SGT", "triv", "ratio", "fav",
                  "padegt_plain", "padegt_boot", "optim", "cheby"]
    }

    # Decide number of runs based on TRUE_ORDER flag
    if TRUE_ORDER and DATA_SOURCE in CHRONO_DATASETS:
        num_runs = 1
        print("Using true order (no shuffling), single run.")
    else:
        num_runs = NUM_SHUFFLES
        print(f"Averaging |%error| over {NUM_SHUFFLES} shuffles.")
        
    if DATA_SOURCE == "genes":
        base_seq = intervals          # intervals = [(a,b),...]
    else:
        base_seq = groups             # groups = [{...}, {...}, ...]


    for s in range(num_runs):

        if TRUE_ORDER and DATA_SOURCE in CHRONO_DATASETS:
            perm_seq = base_seq
        else:
            perm = np.random.permutation(G_total)
            perm_seq = [base_seq[i] for i in perm]
        
        if DATA_SOURCE == "genes":
            intervals_perm = perm_seq
        else:
            groups_perm = perm_seq


        for j, n in enumerate(n_vals):
            if n == 0:
                continue

            if DATA_SOURCE == "genes":
                phi = compute_phi_prefix_from_intervals(intervals_perm, n, contig_len)
            else:
                phi = compute_phi_prefix_from_groups(groups_perm, n)

            # n-based r
            r = (G_total - n) / n
            if r <= 1.0:
                # horizon too small → skip
                continue

            Sn = S_n_from_phi(phi)

            if INCLUDE_QSIP and n in H_cache:
                pred = Sn + apply_estimator(H_cache[n], phi)
                mean["qsip"][j] += abs_pct_err(pred, V_true)

            if INCLUDE_SGT:
                inc = predict_new_SGT(phi, n, r)  # t := n
                mean["SGT"][j] += abs_pct_err(Sn + inc, V_true)

            if INCLUDE_TRIV:
                inc = predict_new_trivial(phi)
                mean["triv"][j] += abs_pct_err(Sn + inc, V_true)

            if INCLUDE_RATIO:
                inc = predict_new_ratio_multiplicative(phi, r)
                mean["ratio"][j] += abs_pct_err(Sn + inc, V_true)

            if INCLUDE_FAVARO:
                inc = predict_new_favaro(phi, r)
                mean["fav"][j] += abs_pct_err(Sn + inc, V_true)

            if INCLUDE_PADEGT_PLAIN:
                inc = predict_new_padegt_plain(phi, r, P=PADE_P, Q=PADE_Q)
                mean["padegt_plain"][j] += abs_pct_err(Sn + inc, V_true)

            if INCLUDE_PADEGT_BOOT:
                inc = predict_new_padegt_boot(phi, r, P=PADE_P, Q=PADE_Q, B=20)
                mean["padegt_boot"][j] += abs_pct_err(Sn + inc, V_true)

            if INCLUDE_OPTIMISTIC:
                inc = predict_new_optimistic(phi, r)
                mean["optim"][j] += abs_pct_err(Sn + inc, V_true)

            if INCLUDE_CHEBYSHEV:
                CHEB_K = G_total
                inc = predict_new_chebyshev_increment(phi, n, CHEB_K, c0=c_0, c1=c_1)
                mean["cheby"][j] += abs_pct_err(Sn + inc, V_true)

        print(f"Completed run {s+1}/{num_runs}")

    # 6) average over runs
    denom = float(num_runs)
    for k in mean:
        mean[k] /= denom

    # 7) optionally save curves + settings to file
    if SAVE_RESULTS:
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
        # --- add N_boosters to name only for MTG ---
        ds_for_name = DATA_SOURCE
        if DATA_SOURCE == "mtg":
            ds_for_name = f"mtg_N{N_boosters}"
    
        fname = f"{ds_for_name}_TO{int(TRUE_ORDER)}_K{GROUP_SUBSAMPLE_K}_S{num_runs}_{RUN_TAG}.json"
        out_path = os.path.join(RESULTS_DIR, fname)

        payload = {
            "data_source": DATA_SOURCE,
            "dataset_label": dataset_label,
            "true_order": bool(TRUE_ORDER),
            "num_runs": int(num_runs),
            "num_shuffles_requested": int(NUM_SHUFFLES),
            "group_subsample_k": int(GROUP_SUBSAMPLE_K),
            "G_total": int(G_total),
            "V_true": int(V_true),
            "T_total": int(T_total),
            "include": {
                "QSIP": INCLUDE_QSIP,
                "SGT": INCLUDE_SGT,
                "TRIV": INCLUDE_TRIV,
                "RATIO": INCLUDE_RATIO,
                "FAVARO": INCLUDE_FAVARO,
                "PADEGT_PLAIN": INCLUDE_PADEGT_PLAIN,
                "PADEGT_BOOT": INCLUDE_PADEGT_BOOT,
                "OPTIMISTIC": INCLUDE_OPTIMISTIC,
                "CHEBYSHEV": INCLUDE_CHEBYSHEV,
            },
            "pade_params": {"P": PADE_P, "Q": PADE_Q},
            "cheb_params": {"c0": c_0, "c1": c_1},
            "n_vals": n_vals.tolist(),
            "perc_vals": perc_vals.tolist(),
            "mean_abs_pct_err": {k: mean[k].tolist() for k in mean},
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[save] wrote results to {out_path}")

    # 8) results + plot
    print(f"\n=== Mean absolute % errors — {dataset_label} ===")
    for idx, n in enumerate(n_vals):
        if n > 0:
            r = (G_total - n) / n
            r_str = f"{r:7.3f}"
        else:
            r_str = "   n/a"
        vals = [
            f"{mean[k][idx]:6.2f}%" if np.isfinite(mean[k][idx]) else "   nan"
            for k in mean
        ]
        print(
            f"n={n:6d} ({100*n/G_total:5.2f}%) | r={r_str} | "
            f"QSIP={vals[0]} | SGT={vals[1]} | TRIV={vals[2]} | "
            f"RATIO={vals[3]} | FAV={vals[4]} | PADÉGT={vals[5]} | "
            f"OPT={vals[6]} | CHEB={vals[8]}"
        )

    plt.figure(figsize=(10, 6))
    xlabel = "Percentage of groups processed (%)"
    if INCLUDE_QSIP:
        plt.plot(perc_vals, mean["qsip"], "o-", label="QSIP", lw=2)
    if INCLUDE_SGT:
        plt.plot(perc_vals, mean["SGT"], "s-", label="SGT", lw=2)
    if INCLUDE_TRIV:
        plt.plot(perc_vals, mean["triv"], "^--", label="Trivial=S_n", lw=2, alpha=0.8)
    if INCLUDE_RATIO:
        plt.plot(perc_vals, mean["ratio"], "x-", label="Ratio", lw=2)
    if INCLUDE_FAVARO:
        plt.plot(perc_vals, mean["fav"], "d-", label="Favaro", lw=2)
    if INCLUDE_PADEGT_PLAIN:
        plt.plot(perc_vals, mean["padegt_plain"], "*-", label=f"Padé–GT [{PADE_P},{PADE_Q}]", lw=2)
    if INCLUDE_PADEGT_BOOT:
        plt.plot(perc_vals, mean["padegt_boot"], "o--", label=f"Padé–GT boot [{PADE_P},{PADE_Q}]", lw=2)
    if INCLUDE_OPTIMISTIC:
        plt.plot(perc_vals, mean["optim"], "v-", label="Optimistic φ₁·r", lw=2)
    if INCLUDE_CHEBYSHEV:
        plt.plot(perc_vals, mean["cheby"], "o-", label="Chebyshev", lw=2)

    plt.xlabel(xlabel)
    plt.ylabel("Mean absolute percentage error (%)")
    plt.title(f"{dataset_label} — TOTAL distinct tokens |%error|")
    plt.legend()
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()
