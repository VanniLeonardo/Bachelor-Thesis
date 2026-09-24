#!/usr/bin/env python3
# Copyright (c) 2026 Leonardo Vanni. CC BY-NC 4.0 (see LICENSE.txt).
"""Pack CO3D into one uncompressed zip per sequence.

CO3D ships four small files per frame.  On a filesystem with large allocation units
(1 MB clusters on exFAT, for instance) each 10 KB JPEG costs a whole cluster and the
dataset inflates several times over.  One archive per sequence turns millions of tiny
files into a few thousand large ones and removes that overhead, while keeping random
access cheap: the archives are stored uncompressed, so reading a frame is a seek.

Two sources are supported:

    # already extracted directories  <root>/<category>/<sequence>/images/...
    python scripts/repack_co3d.py --root /data/CO3D --from-dir --delete-source

    # a downloaded CO3D chunk        tv_000.zip
    python scripts/repack_co3d.py --root /data/CO3D --from-zip tv_000.zip

Only the files the training loader opens are kept: images/, depths/ and depth_masks/.
Object masks and point clouds are dropped.  Frames missing any of the three are left
out of the manifest so the loader never sees a half-present frame.
"""
import argparse
import json
import os
import shutil
import sys
import zipfile
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))

from data.archive_store import MANIFEST_NAME, append_manifest, frame_members, read_manifest  # noqa: E402

KEEP_DIRS = ("/images/", "/depths/", "/depth_masks/")
SEQUENCE_LEVEL = 2  # <category>/<sequence>/...
# per-category metadata directories, not sequences
NON_SEQUENCE_DIRS = ("set_lists", "eval_batches")


def wanted(member: str) -> bool:
    """True for the per-frame files the loader reads."""
    return any(d in member for d in KEEP_DIRS)


def sequence_of(member: str):
    """"tv/123_456/images/f.jpg" -> ("tv", "123_456"), or None for per-category files."""
    parts = member.split("/")
    if len(parts) <= SEQUENCE_LEVEL or not wanted(member):
        return None
    return parts[0], parts[1]


def manifest_frames(members) -> list:
    """Frame filepaths for which the image, the depth and the depth mask are all present."""
    present = set(members)
    frames = []
    for m in members:
        if "/images/" not in m:
            continue
        if all(p in present for p in frame_members(m).values()):
            frames.append(m)
    return frames


def pack_sequence(root: str, category: str, seq: str, members: dict, verify: bool = True, merge: bool = True) -> int:
    """Write <root>/<category>/<seq>.zip from {member name: bytes or path}. Returns frames packed.

    CO3D chunks are cut at a fixed size, not at sequence boundaries, so one sequence can
    arrive in two chunks.  When an archive for the sequence already exists its members are
    carried over, otherwise the second chunk would silently drop the first one's frames.
    """
    out_dir = os.path.join(root, category)
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, f"{seq}.zip")
    tmp = final + ".part"
    carried = 0
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
        if merge and os.path.isfile(final):
            with zipfile.ZipFile(final) as old:
                for name in old.namelist():
                    if name not in members:  # a re-packed member wins
                        zf.writestr(name, old.read(name))
                        carried += 1
        for name in sorted(members):
            src = members[name]
            if isinstance(src, bytes):
                zf.writestr(name, src)
            else:
                zf.write(src, arcname=name)
    with zipfile.ZipFile(tmp) as zf:
        written = set(zf.namelist())
        if verify:
            bad = zf.testzip()
            if bad is not None:
                os.remove(tmp)
                raise IOError(f"corrupt member {bad} in {tmp}")
            if not set(members) <= written:
                os.remove(tmp)
                raise IOError(f"member mismatch in {tmp}")
    os.replace(tmp, final)
    frames = manifest_frames(written)
    append_manifest(root, seq, frames)
    return len(frames)


def repack_from_zip(root: str, zip_path: str, verify: bool = True) -> dict:
    """Stream one downloaded CO3D chunk into per-sequence archives, without unpacking to disk."""
    stats = defaultdict(int)
    os.makedirs(root, exist_ok=True)
    with zipfile.ZipFile(zip_path) as src:
        by_seq = defaultdict(list)
        others = []
        for info in src.infolist():
            if info.is_dir():
                continue
            key = sequence_of(info.filename)
            if key is None:
                if not any(d in info.filename for d in ("/masks/", "pointcloud.ply")):
                    others.append(info.filename)
            else:
                by_seq[key].append(info.filename)
        for (category, seq), names in sorted(by_seq.items()):
            members = {n: src.read(n) for n in names}
            stats["frames"] += pack_sequence(root, category, seq, members, verify)
            stats["sequences"] += 1
        for name in others:  # per-category annotations and set lists stay as plain files
            dst = os.path.join(root, name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with src.open(name) as fsrc, open(dst, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)
            stats["side_files"] += 1
    return dict(stats)


def repack_from_dir(root: str, categories=None, delete_source: bool = False, verify: bool = True, limit=None) -> dict:
    """Pack already-extracted <root>/<category>/<sequence>/ trees, one sequence at a time."""
    stats = defaultdict(int)
    done = set(read_manifest(root))
    cats = categories or sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith("_")
    )
    for category in cats:
        cat_dir = os.path.join(root, category)
        seqs = sorted(
            d for d in os.listdir(cat_dir)
            if os.path.isdir(os.path.join(cat_dir, d)) and d not in NON_SEQUENCE_DIRS
        )
        for seq in seqs:
            if limit is not None and stats["sequences"] >= limit:
                return dict(stats)
            seq_dir = os.path.join(cat_dir, seq)
            if seq in done and os.path.isfile(os.path.join(cat_dir, f"{seq}.zip")):
                stats["skipped"] += 1
                if delete_source:
                    shutil.rmtree(seq_dir)
                continue
            members = {}
            for sub in ("images", "depths", "depth_masks"):
                sub_dir = os.path.join(seq_dir, sub)
                if not os.path.isdir(sub_dir):
                    continue
                for fn in os.listdir(sub_dir):
                    members[f"{category}/{seq}/{sub}/{fn}"] = os.path.join(sub_dir, fn)
            if not members:
                stats["empty"] += 1
                continue
            stats["frames"] += pack_sequence(root, category, seq, members, verify)
            stats["sequences"] += 1
            if delete_source:
                shutil.rmtree(seq_dir)
            if stats["sequences"] % 50 == 0:
                print(f"  {stats['sequences']} sequences, {stats['frames']} frames", flush=True)
    return dict(stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="CO3D root (holds <category>/<sequence>.zip and the manifest)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-dir", action="store_true", help="pack already-extracted sequence directories")
    src.add_argument("--from-zip", help="pack one downloaded CO3D chunk")
    ap.add_argument("--categories", help="comma-separated subset for --from-dir")
    ap.add_argument("--limit", type=int, help="stop after N sequences (for testing)")
    ap.add_argument("--delete-source", action="store_true", help="remove each directory once its archive is verified")
    ap.add_argument("--no-verify", action="store_true", help="skip the read-back check of every archive")
    args = ap.parse_args()

    if args.from_zip:
        stats = repack_from_zip(args.root, args.from_zip, verify=not args.no_verify)
    else:
        cats = [c.strip() for c in args.categories.split(",")] if args.categories else None
        stats = repack_from_dir(args.root, cats, args.delete_source, not args.no_verify, args.limit)
    print(json.dumps(stats, indent=2))
    print(f"manifest: {os.path.join(args.root, MANIFEST_NAME)}")


if __name__ == "__main__":
    main()
