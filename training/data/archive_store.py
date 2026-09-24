# Copyright (c) 2026 Leonardo Vanni.
# Part of a derivative work of VGGT (Meta Platforms, Inc.); distributed under the
# same CC BY-NC 4.0 license found in the LICENSE.txt file in the root directory.

"""Reading CO3D frames from one uncompressed zip per sequence.

CO3D stores four small files per frame.  On filesystems with large allocation units
(a 1 MB cluster on exFAT, say) a 10 KB JPEG costs a whole cluster, which inflates the
dataset several times over.  Packing each sequence into a single archive removes that
cost: the dataset becomes a few thousand large files instead of millions of tiny ones.

Layout produced by ``scripts/repack_co3d.py``::

    <CO3D_DIR>/_archive_frames.jsonl       one JSON line per sequence, the frame manifest
    <CO3D_DIR>/<category>/<sequence>.zip   uncompressed, members keep their full
                                           "<category>/<sequence>/images/frame###.jpg" name
    <CO3D_DIR>/<category>/frame_annotations.jgz, set_lists/, ...   left as plain files

Archives are stored without compression, so reading a frame is a seek and a read.
"""

import json
import os
import zipfile
from typing import Dict, List, Optional

MANIFEST_NAME = "_archive_frames.jsonl"


def archive_relpath(frame_filepath: str) -> str:
    """"tv/123_456/images/frame001.jpg" -> "tv/123_456.zip"."""
    parts = frame_filepath.split("/")
    if len(parts) < 3:
        raise ValueError(f"unexpected CO3D filepath: {frame_filepath!r}")
    return f"{parts[0]}/{parts[1]}.zip"


def frame_members(frame_filepath: str) -> Dict[str, str]:
    """The three members the loader needs for one frame, keyed by kind.

    Mirrors the path arithmetic in ``Co3dDataset``: the depth map and the MVS depth mask
    sit next to the image in sibling directories.
    """
    return {
        "image": frame_filepath,
        "depth": frame_filepath.replace("/images/", "/depths/") + ".geometric.png",
        "depth_mask": frame_filepath.replace("/images/", "/depth_masks/").replace(".jpg", ".png"),
    }


def read_manifest(root: str) -> Dict[str, List[str]]:
    """Load ``_archive_frames.jsonl`` -> {sequence name: [frame filepath, ...]}.

    The file is append-only, so a sequence repacked twice appears twice and the last
    entry wins.  Returns an empty dict when the dataset is not in archive form.
    """
    path = os.path.join(root, MANIFEST_NAME)
    if not os.path.isfile(path):
        return {}
    out: Dict[str, List[str]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["seq"]] = rec["frames"]
    return out


def append_manifest(root: str, seq_name: str, frames: List[str]) -> None:
    """Record the frames packed for one sequence (append-only, crash safe)."""
    with open(os.path.join(root, MANIFEST_NAME), "a") as f:
        f.write(json.dumps({"seq": seq_name, "frames": sorted(frames)}) + "\n")


class ArchiveStore:
    """Per-process cache of open sequence archives.

    DataLoader workers are forked, and a ``ZipFile`` handle created before the fork
    shares a file offset between processes.  The cache is therefore keyed by pid and
    rebuilt transparently in each worker.
    """

    def __init__(self, root: str, max_open: int = 8):
        self.root = root
        self.max_open = max_open
        self._pid: Optional[int] = None
        self._open: "Dict[str, zipfile.ZipFile]" = {}

    def _cache(self) -> Dict[str, zipfile.ZipFile]:
        pid = os.getpid()
        if pid != self._pid:  # forked into a new worker: drop inherited handles
            self._open = {}
            self._pid = pid
        return self._open

    def archive(self, frame_filepath: str) -> zipfile.ZipFile:
        rel = archive_relpath(frame_filepath)
        cache = self._cache()
        zf = cache.get(rel)
        if zf is None:
            path = os.path.join(self.root, rel)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"sequence archive missing: {path}")
            zf = zipfile.ZipFile(path)
            if len(cache) >= self.max_open:
                _, old = cache.popitem()
                try:
                    old.close()
                except Exception:
                    pass
            cache[rel] = zf
        return zf

    def read(self, member: str) -> bytes:
        """Raw bytes of one member, addressed by its full CO3D-relative path."""
        return self.archive(member).read(member)

    def has(self, member: str) -> bool:
        try:
            self.archive(member).getinfo(member)
            return True
        except (KeyError, FileNotFoundError):
            return False

    def close(self) -> None:
        for zf in self._cache().values():
            try:
                zf.close()
            except Exception:
                pass
        self._open = {}

    # A dataset holding this object must stay picklable for DataLoader workers.
    def __getstate__(self):
        return {"root": self.root, "max_open": self.max_open}

    def __setstate__(self, state):
        self.root = state["root"]
        self.max_open = state["max_open"]
        self._pid = None
        self._open = {}
