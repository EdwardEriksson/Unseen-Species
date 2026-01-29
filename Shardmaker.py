#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sharded genome preprocessor (intervals), modeled after your previous working setup.

Behavior:
  - Uses the same inputs/structure as before (uncompressed FASTQ path).
  - Processes reads in batches (shards) to avoid big temp files.
  - Builds (once) and reuses a minimap2 .mmi index for the sliced FASTA.
  - For each read, stores a single interval (start, end) on the slice contig
    where start is 0-based inclusive and end is 0-based exclusive.
  - Drops unaligned reads by default; use --keep-empty to store (-1, -1).

Outputs:
  out_dir/
    manifest.json
    slice.fasta
    slice.fasta.mmi
    shards/
      intervals_shard_00001.pkl.gz
      intervals_shard_00002.pkl.gz
      ...
"""

import os
import sys
import re
import json
import gzip
import time
import shutil
import pickle
import argparse
import subprocess
from typing import List, Tuple, Iterator
from contextlib import contextmanager

# -------------------------
# Small utilities
# -------------------------

def win_to_wsl(path: str) -> str:
    path = os.path.abspath(path)
    drive = path[0].lower()
    tail = path[2:].replace("\\", "/").lstrip("/")
    return f"/mnt/{drive}/{tail}"

def run_cmd_capture(cmd_list, desc="command"):
    proc = subprocess.run(cmd_list, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"=== STDOUT ({desc}) ===\n{proc.stdout}")
        print(f"=== STDERR ({desc}) ===\n{proc.stderr}")
        raise subprocess.CalledProcessError(proc.returncode, cmd_list,
                                            output=proc.stdout, stderr=proc.stderr)
    return proc.stdout, proc.stderr

@contextmanager
def opener_any(path: str, mode: str):
    """Text opener that supports .gz or plain .fastq."""
    if path.lower().endswith(".gz"):
        f = gzip.open(path, mode + "t", encoding="utf-8", errors="ignore")
    else:
        f = open(path, mode, encoding="utf-8", errors="ignore", newline="\n")
    try:
        yield f
    finally:
        f.close()

# -------------------------
# FASTA slice + index
# -------------------------

def make_small_fasta(src_fasta, dst_fasta, max_bases=100_000, force=False):
    if (not force) and os.path.exists(dst_fasta):
        print(f"[slice] reusing {dst_fasta}")
        return dst_fasta
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
        raise RuntimeError("[slice] could not slice FASTA")
    os.makedirs(os.path.dirname(dst_fasta), exist_ok=True)
    with open(dst_fasta, "w", newline="\n") as out:
        out.write(f">{name}\n")
        seq = "".join(seq_chunks)
        for i in range(0, len(seq), 60):
            out.write(seq[i:i+60] + "\n")
    print(f"[slice] wrote {total} bases of {name} to {dst_fasta}")
    return dst_fasta

def get_first_contig_and_len(fasta_path) -> Tuple[str, int]:
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

def ensure_minimap2_index_wsl(ref_fasta_win, wsl_exe=None) -> str:
    if wsl_exe is None:
        wsl_exe = r"C:\Windows\Sysnative\wsl.exe" if "PROCESSOR_ARCHITEW6432" in os.environ \
                  else r"C:\Windows\System32\wsl.exe"
    mmi_win = ref_fasta_win + ".mmi"
    if not os.path.exists(mmi_win):
        print("[idx] building minimap2 index (.mmi)…")
        run_cmd_capture([wsl_exe, "minimap2", "-d", win_to_wsl(mmi_win), win_to_wsl(ref_fasta_win)],
                        desc="minimap2 -d")
    else:
        print(f"[idx] reusing {mmi_win}")
    return mmi_win

# -------------------------
# Alignment and intervals
# -------------------------

def run_minimap2_wsl_indexed(mmi_win, fastq_win, sam_out_win,
                             wsl_exe=None, minimap2_opts=("-x", "sr", "-a")):
    if wsl_exe is None:
        wsl_exe = r"C:\Windows\Sysnative\wsl.exe" if "PROCESSOR_ARCHITEW6432" in os.environ \
                  else r"C:\Windows\System32\wsl.exe"
    cmd = [wsl_exe, "minimap2", *minimap2_opts, win_to_wsl(mmi_win), win_to_wsl(fastq_win)]
    sam_txt, _ = run_cmd_capture(cmd, desc="minimap2 (indexed)")
    os.makedirs(os.path.dirname(sam_out_win), exist_ok=True)
    with open(sam_out_win, "w", newline="\n") as f:
        f.write(sam_txt)
    return sam_out_win

def cigar_ref_len(cigar: str) -> int:
    ref_consume = set("MDN=X")
    num = ""
    total = 0
    for ch in cigar:
        if ch.isdigit():
            num += ch
        else:
            if ch in ref_consume and num:
                total += int(num)
            num = ""
    return total

def intervals_from_sam(sam_path: str,
                       contig: str,
                       contig_len: int,
                       *,
                       drop_gappy: bool = True,
                       keep_empty: bool = False) -> List[Tuple[int, int]]:
    """
    One interval per QNAME: [min_start, max_end) across its primary alignments on `contig`.
    Unaligned → dropped (or (-1, -1) if keep_empty=True).
    """
    intervals: List[Tuple[int, int]] = []

    current_qname = None
    have_any_for_read = False
    span_start = None
    span_end = None

    def flush_current():
        nonlocal have_any_for_read, span_start, span_end
        if current_qname is None:
            return
        if have_any_for_read:
            intervals.append((span_start, span_end))
        elif keep_empty:
            intervals.append((-1, -1))
        have_any_for_read = False
        span_start = None
        span_end = None

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
                flush_current()
                current_qname = qname

            if flag & 0x100 or flag & 0x800:
                continue
            if rname == "*" or cigar == "*":
                continue
            if drop_gappy and ("N" in cigar or "D" in cigar):
                continue
            if rname != contig:
                continue

            ref_len = cigar_ref_len(cigar)
            start = pos - 1
            end = min(start + ref_len, contig_len)
            if end <= start:
                continue

            have_any_for_read = True
            if span_start is None or start < span_start:
                span_start = start
            if span_end is None or end > span_end:
                span_end = end

    flush_current()
    return intervals

# -------------------------
# Sharded pipeline
# -------------------------

def stream_fastq_records(src_fastq_path: str) -> Iterator[Tuple[str,str,str,str]]:
    with opener_any(src_fastq_path, "r") as fin:
        while True:
            h = fin.readline()
            if not h:
                break
            s = fin.readline()
            p = fin.readline()
            q = fin.readline()
            if not q:
                break
            yield h, s, p, q

def write_shard(out_dir: str, shard_idx: int,
                intervals: List[Tuple[int, int]], meta_patch: dict) -> str:
    shards_dir = os.path.join(out_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)
    shard_path = os.path.join(shards_dir, f"intervals_shard_{shard_idx:05d}.pkl.gz")
    meta = {"shard_index": shard_idx, **meta_patch}
    with gzip.open(shard_path, "wb") as f:
        pickle.dump({"intervals": intervals, "meta": meta}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return shard_path

def save_manifest(out_dir: str, manifest: dict):
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

def load_manifest(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def process_all_reads_sharded_intervals(*,
                                        ref_full: str,
                                        out_dir: str,
                                        fastq_path: str,
                                        slice_path: str,
                                        sam_tmp_dir: str,
                                        reads_per_shard: int = 100_000,
                                        slice_bases: int = 100_000,
                                        force_reslice: bool = False,
                                        keep_empty: bool = False,
                                        drop_gappy: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(sam_tmp_dir, exist_ok=True)

    # slice + index
    ref_slice = make_small_fasta(ref_full, slice_path, max_bases=slice_bases, force=force_reslice)
    contig, contig_len = get_first_contig_and_len(ref_slice)
    mmi_path = ensure_minimap2_index_wsl(ref_slice)

    shard_idx = 1
    total_reads = 0
    total_intervals = 0
    shard_entries = []

    t_all = time.time()
    batch_records = []
    batch_reads = 0
    batch_fastq = os.path.join(sam_tmp_dir, "batch.fastq")
    batch_sam   = os.path.join(sam_tmp_dir, "batch.sam")

    for h, s, p, q in stream_fastq_records(fastq_path):
        batch_records.append((h, s, p, q))
        batch_reads += 1
        total_reads += 1

        if batch_reads >= reads_per_shard:
            with open(batch_fastq, "w", encoding="utf-8", newline="\n") as fout:
                for H,S,P,Q in batch_records:
                    fout.write(H); fout.write(S); fout.write(P); fout.write(Q)

            run_minimap2_wsl_indexed(mmi_path, batch_fastq, batch_sam)

            intervals = intervals_from_sam(batch_sam, contig, contig_len,
                                           drop_gappy=drop_gappy, keep_empty=keep_empty)

            meta_patch = dict(
                contig=contig,
                contig_len=contig_len,
                slice_bases=slice_bases,
                dropped_empty=(not keep_empty),
                reads_in_shard=batch_reads,
                n_intervals=len(intervals),
            )
            shard_path = write_shard(out_dir, shard_idx, intervals, meta_patch)
            shard_entries.append({"index": shard_idx,
                                  "path": os.path.relpath(shard_path, out_dir),
                                  "intervals": len(intervals),
                                  "reads": batch_reads})
            total_intervals += len(intervals)

            print(f"[shard {shard_idx:05d}] reads={batch_reads} intervals={len(intervals)} -> {shard_path}")

            batch_records.clear()
            batch_reads = 0
            shard_idx += 1
            for pth in (batch_fastq, batch_sam):
                try: os.remove(pth)
                except FileNotFoundError: pass

    # tail
    if batch_reads > 0:
        with open(batch_fastq, "w", encoding="utf-8", newline="\n") as fout:
            for H,S,P,Q in batch_records:
                fout.write(H); fout.write(S); fout.write(P); fout.write(Q)
        run_minimap2_wsl_indexed(mmi_path, batch_fastq, batch_sam)
        intervals = intervals_from_sam(batch_sam, contig, contig_len,
                                       drop_gappy=drop_gappy, keep_empty=keep_empty)
        meta_patch = dict(
            contig=contig,
            contig_len=contig_len,
            slice_bases=slice_bases,
            dropped_empty=(not keep_empty),
            reads_in_shard=batch_reads,
            n_intervals=len(intervals),
        )
        shard_path = write_shard(out_dir, shard_idx, intervals, meta_patch)
        shard_entries.append({"index": shard_idx,
                              "path": os.path.relpath(shard_path, out_dir),
                              "intervals": len(intervals),
                              "reads": batch_reads})
        total_intervals += len(intervals)
        print(f"[shard {shard_idx:05d}] reads={batch_reads} intervals={len(intervals)} -> {shard_path}")
        for pth in (batch_fastq, batch_sam):
            try: os.remove(pth)
            except FileNotFoundError: pass

    manifest = dict(
        version=1,
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        fastq=os.path.abspath(fastq_path),
        ref_full=os.path.abspath(ref_full),
        ref_slice=os.path.abspath(ref_slice),
        contig=contig,
        contig_len=contig_len,
        slice_bases=slice_bases,
        dropped_empty=(not keep_empty),
        drop_gappy=drop_gappy,
        reads_per_shard=reads_per_shard,
        total_reads=total_reads,
        total_intervals=total_intervals,
        shards=shard_entries,
    )
    save_manifest(out_dir, manifest)
    print(f"[done] shards={len(shard_entries)} total_reads={total_reads} "
          f"total_intervals={total_intervals} elapsed={time.time()-t_all:.1f}s")
    print(f"[done] manifest: {os.path.join(out_dir, 'manifest.json')}")

# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-full", required=True, help="Full reference FASTA (Windows path).")
    ap.add_argument("--fastq", required=True, help="FASTQ or FASTQ.gz (Windows path).")
    ap.add_argument("--out-dir", required=True, help="Output directory (shards + manifest).")
    ap.add_argument("--slice", required=True, help="Sliced FASTA output path (Windows path).")
    ap.add_argument("--sam-tmp", required=True, help="Temp directory for batch FASTQ/SAM.")
    ap.add_argument("--reads-per-shard", type=int, default=100_000)
    ap.add_argument("--slice-bases", type=int, default=100_000)
    ap.add_argument("--force-reslice", action="store_true")
    ap.add_argument("--keep-empty", action="store_true",
                    help="Keep unaligned reads as (-1,-1). Default: drop them.")
    ap.add_argument("--no-drop-gappy", action="store_true",
                    help="Do not drop N/D CIGARs. Default: drop them.")
    args = ap.parse_args()

    process_all_reads_sharded_intervals(
        ref_full=args.ref_full,
        out_dir=args.out_dir,
        fastq_path=args.fastq,
        slice_path=args.slice,
        sam_tmp_dir=args.sam_tmp,
        reads_per_shard=args.reads_per_shard,
        slice_bases=args.slice_bases,
        force_reslice=args.force_reslice,
        keep_empty=args.keep_empty,
        drop_gappy=(not args.no_drop_gappy),
    )

if __name__ == "__main__":
    # IDE defaults modeled after your previous working setup (uncompressed FASTQ path).
    if len(sys.argv) == 1:
        sys.argv += [
            "--ref-full",  r"C:\Users\Edwar\ref_human_GRCh38.fasta",
            "--fastq",     r"C:\Users\Edwar\SRR504410_1.fastq",   # same as before
            "--out-dir",   r"C:\Users\Edwar\toy_align\sharded_intervals",
            "--slice",     r"C:\Users\Edwar\toy_align\slice.fasta",
            "--sam-tmp",   r"C:\Users\Edwar\toy_align\tmp",
            "--reads-per-shard", "100000",
            "--slice-bases",     "100000",
            "--force-reslice",
            # "--keep-empty",     # keep (-1,-1) for unaligned
            # "--no-drop-gappy",  # include N/D spans
        ]
    main()
