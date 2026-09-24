# Copyright (c) 2026 Leonardo Vanni. CC BY-NC 4.0 (see LICENSE.txt).
"""Tests for the zip-backed CO3D storage.

The dataset can read frames either from loose files or from one uncompressed archive per
sequence.  These tests build a small synthetic CO3D tree, pack it, and check that the two
paths agree exactly: same bytes, same decoded arrays, same dataset output, including under
DataLoader workers.  A final test repeats the comparison on the real dataset when it is
mounted, and is skipped otherwise.
"""
import gzip
import io
import json
import os
import sys
import zipfile

import cv2
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from data.archive_store import ArchiveStore, archive_relpath, frame_members, read_manifest  # noqa: E402
from data.dataset_util import (  # noqa: E402
    decode_16big_png_depth,
    decode_depth,
    decode_image_cv2,
    decode_mask_gray,
    load_16big_png_depth,
    read_depth,
    read_image_cv2,
)
from repack_co3d import repack_from_dir, repack_from_zip  # noqa: E402

CATEGORIES = {"tv": ["28_001_0001", "28_002_0002"], "cup": ["30_003_0003"]}
FRAMES_PER_SEQ = 6
H, W = 24, 32


# --------------------------------------------------------------------------- fixtures
def _depth_png_bytes(rng) -> bytes:
    """A CO3D depth map: float16 values whose raw bits are stored in a 16-bit PNG."""
    depth = (rng.random((H, W)).astype(np.float16) * 3 + 0.5).astype(np.float16)
    as_u16 = np.frombuffer(depth.tobytes(), dtype=np.uint16).reshape(H, W)
    buf = io.BytesIO()
    Image.fromarray(as_u16.astype(np.int32), mode="I").save(buf, format="PNG", bits=16)
    return buf.getvalue()


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


@pytest.fixture
def co3d_tree(tmp_path):
    """A miniature extracted CO3D dataset plus matching annotation files."""
    rng = np.random.default_rng(0)
    root = tmp_path / "CO3D"
    ann_dir = tmp_path / "CO3D_ann"
    ann_dir.mkdir(parents=True)
    annotations = {cat: {} for cat in CATEGORIES}
    for cat, seqs in CATEGORIES.items():
        for seq in seqs:
            frames = []
            for i in range(FRAMES_PER_SEQ):
                stem = f"frame{i:06d}.jpg"
                rel = f"{cat}/{seq}/images/{stem}"
                img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
                _write(str(root / rel), cv2.imencode(".jpg", img)[1].tobytes())
                _write(str(root / f"{cat}/{seq}/depths/{stem}.geometric.png"), _depth_png_bytes(rng))
                mask = (rng.random((H, W)) > 0.3).astype(np.uint8) * 255
                _write(str(root / f"{cat}/{seq}/depth_masks/frame{i:06d}.png"), cv2.imencode(".png", mask)[1].tobytes())
                # files the loader never reads; the repacker must drop them
                _write(str(root / f"{cat}/{seq}/masks/frame{i:06d}.png"), cv2.imencode(".png", mask)[1].tobytes())
                extri = np.eye(4)[:3].tolist()
                extri[0][3] = float(i)
                intri = [[50.0, 0, W / 2], [0, 50.0, H / 2], [0, 0, 1]]
                frames.append({"filepath": rel, "extri": extri, "intri": intri})
            _write(str(root / f"{cat}/{seq}/pointcloud.ply"), b"ply\n")
            annotations[cat][seq] = frames
        _write(str(root / f"{cat}/frame_annotations.jgz"), b"side-car")
        for split in ("train", "test"):
            with gzip.open(ann_dir / f"{cat}_{split}.jgz", "wt") as f:
                json.dump(annotations[cat], f)
    return root, ann_dir, annotations


def _conf(**over):
    base = dict(
        img_size=28, patch_size=14, augs={"scales": [1.0, 1.0]}, rescale=True, rescale_aug=False,
        landscape_check=False, debug=True, training=False, get_nearby=False, load_depth=True,
        inside_random=False, allow_duplicate_img=True,
    )
    base.update(over)
    return OmegaConf.create(base)


def _dataset(root, ann_dir, storage, **over):
    import data.datasets.co3d as co3d_mod
    from data.datasets.co3d import Co3dDataset

    co3d_mod.SEEN_CATEGORIES = list(CATEGORIES)  # the fixture only has these
    return Co3dDataset(
        common_conf=_conf(**over), split="train", CO3D_DIR=str(root),
        CO3D_ANNOTATION_DIR=str(ann_dir), storage=storage, min_num_images=2, len_train=10,
    )


# --------------------------------------------------------------------------- decoders
def test_decoders_match_the_file_readers(co3d_tree):
    root, _, ann = co3d_tree
    rel = ann["tv"]["28_001_0001"][0]["filepath"]
    members = frame_members(rel)
    img_path = str(root / members["image"])
    assert np.array_equal(decode_image_cv2(open(img_path, "rb").read()), read_image_cv2(img_path))
    d_path = str(root / members["depth"])
    a, b = decode_depth(open(d_path, "rb").read()), read_depth(d_path, 1.0)
    assert a.dtype == b.dtype == np.float32 and np.array_equal(a, b)
    assert np.array_equal(decode_16big_png_depth(open(d_path, "rb").read()), load_16big_png_depth(d_path))
    m_path = str(root / members["depth_mask"])
    assert np.array_equal(decode_mask_gray(open(m_path, "rb").read()), cv2.imread(m_path, cv2.IMREAD_GRAYSCALE))


def test_frame_members_paths():
    m = frame_members("tv/28_001/images/frame000003.jpg")
    assert m["image"] == "tv/28_001/images/frame000003.jpg"
    assert m["depth"] == "tv/28_001/depths/frame000003.jpg.geometric.png"
    assert m["depth_mask"] == "tv/28_001/depth_masks/frame000003.png"
    assert archive_relpath(m["image"]) == "tv/28_001.zip"


# --------------------------------------------------------------------------- repacking
def test_repack_from_dir_keeps_bytes_and_drops_unused(co3d_tree):
    root, _, ann = co3d_tree
    originals = {}
    for cat, seqs in CATEGORIES.items():
        for seq in seqs:
            for f in ann[cat][seq]:
                for member in frame_members(f["filepath"]).values():
                    originals[member] = open(root / member, "rb").read()

    stats = repack_from_dir(str(root))
    assert stats["sequences"] == 3 and stats["frames"] == 3 * FRAMES_PER_SEQ

    manifest = read_manifest(str(root))
    assert set(manifest) == {s for seqs in CATEGORIES.values() for s in seqs}
    assert all(len(v) == FRAMES_PER_SEQ for v in manifest.values())

    store = ArchiveStore(str(root))
    for member, data in originals.items():
        assert store.read(member) == data, member

    with zipfile.ZipFile(root / "tv/28_001_0001.zip") as zf:
        names = zf.namelist()
        assert all(i.compress_type == zipfile.ZIP_STORED for i in zf.infolist())  # seekable reads
        assert not any("/masks/" in n or n.endswith("pointcloud.ply") for n in names)
        assert len(names) == 3 * FRAMES_PER_SEQ
    store.close()


def test_repack_is_resumable_and_can_delete_sources(co3d_tree):
    root, _, _ = co3d_tree
    first = repack_from_dir(str(root), limit=1)
    assert first["sequences"] == 1
    second = repack_from_dir(str(root), delete_source=True)
    assert second["skipped"] == 1 and second["sequences"] == 2
    assert not (root / "tv/28_001_0001").exists()  # source directory reclaimed
    assert (root / "tv/28_001_0001.zip").is_file()
    assert len(read_manifest(str(root))) == 3


def test_incomplete_frames_are_left_out_of_the_manifest(co3d_tree):
    root, _, ann = co3d_tree
    victim = ann["cup"]["30_003_0003"][2]["filepath"]
    os.remove(root / frame_members(victim)["depth_mask"])
    repack_from_dir(str(root), categories=["cup"])
    frames = read_manifest(str(root))["30_003_0003"]
    assert victim not in frames and len(frames) == FRAMES_PER_SEQ - 1


def test_repack_from_zip_matches_repack_from_dir(co3d_tree, tmp_path):
    root, _, _ = co3d_tree
    chunk = tmp_path / "tv_000.zip"
    with zipfile.ZipFile(chunk, "w") as zf:
        for dirpath, _, filenames in os.walk(root / "tv"):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                zf.write(full, arcname=os.path.relpath(full, root))
    out = tmp_path / "from_zip"
    stats = repack_from_zip(str(out), str(chunk))
    assert stats["sequences"] == 2 and stats["side_files"] >= 1
    repack_from_dir(str(root), categories=["tv"])
    for seq in CATEGORIES["tv"]:
        with zipfile.ZipFile(out / f"tv/{seq}.zip") as a, zipfile.ZipFile(root / f"tv/{seq}.zip") as b:
            assert sorted(a.namelist()) == sorted(b.namelist())
            for n in a.namelist():
                assert a.read(n) == b.read(n)
    assert (out / "tv/frame_annotations.jgz").is_file()  # side-car kept as a plain file


# --------------------------------------------------------------------------- store
def test_archive_store_missing_and_pickle(co3d_tree):
    import pickle

    root, _, ann = co3d_tree
    repack_from_dir(str(root))
    store = ArchiveStore(str(root))
    rel = ann["tv"]["28_001_0001"][0]["filepath"]
    assert store.has(rel) and not store.has(rel.replace("frame000000", "frame009999"))
    with pytest.raises(FileNotFoundError):
        store.read("nope/missing/images/frame000000.jpg")
    revived = pickle.loads(pickle.dumps(store))  # datasets are pickled to DataLoader workers
    assert revived.read(rel) == store.read(rel)
    store.close()


def test_archive_store_drops_handles_inherited_across_fork(co3d_tree):
    root, _, ann = co3d_tree
    repack_from_dir(str(root))
    store = ArchiveStore(str(root))
    rel = ann["tv"]["28_001_0001"][0]["filepath"]
    expected = store.read(rel)  # opens a handle in the parent
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: the inherited handle must not be reused
        os.close(r)
        try:
            ok = store.read(rel) == expected and store._pid == os.getpid()
            os.write(w, b"1" if ok else b"0")
        finally:
            os._exit(0)
    os.close(w)
    assert os.read(r, 1) == b"1"
    os.waitpid(pid, 0)
    store.close()


# --------------------------------------------------------------------------- dataset
def _same_batch(a, b):
    assert a["seq_name"] == b["seq_name"] and a["frame_num"] == b["frame_num"]
    for key in ("images", "depths", "extrinsics", "intrinsics", "cam_points", "world_points", "point_masks"):
        for x, y in zip(a[key], b[key]):
            assert np.array_equal(np.asarray(x), np.asarray(y)), key


def test_dataset_archive_mode_matches_file_mode(co3d_tree):
    root, ann_dir, _ = co3d_tree
    files_ds = _dataset(root, ann_dir, "files")
    assert files_ds.storage == "files"
    repack_from_dir(str(root))
    arch_ds = _dataset(root, ann_dir, "archives")
    assert arch_ds.storage == "archives"
    assert sorted(files_ds.data_store) == sorted(arch_ds.data_store)
    for seq in sorted(files_ds.data_store):
        assert [f["filepath"] for f in files_ds.data_store[seq]] == [f["filepath"] for f in arch_ds.data_store[seq]]
        ids = list(range(FRAMES_PER_SEQ))
        _same_batch(files_ds.get_data(seq_name=seq, ids=ids), arch_ds.get_data(seq_name=seq, ids=ids))


def test_dataset_auto_detects_storage_and_archives_mode_requires_manifest(co3d_tree):
    root, ann_dir, _ = co3d_tree
    assert _dataset(root, ann_dir, "auto").storage == "files"
    with pytest.raises(FileNotFoundError):
        _dataset(root, ann_dir, "archives")
    repack_from_dir(str(root))
    assert _dataset(root, ann_dir, "auto").storage == "archives"
    with pytest.raises(ValueError):
        _dataset(root, ann_dir, "nonsense")


@pytest.mark.xfail(
    strict=True,
    reason="upstream limitation: crop_image_depth_and_intrinsic_by_pp indexes the depth map "
    "unconditionally, so load_depth=False has never worked for CO3D in either storage mode",
)
def test_dataset_works_without_depth(co3d_tree):
    root, ann_dir, _ = co3d_tree
    repack_from_dir(str(root))
    ds = _dataset(root, ann_dir, "archives", load_depth=False)
    batch = ds.get_data(seq_name="28_001_0001", ids=[0, 1])
    assert batch["frame_num"] == 2 and all(d is None for d in batch["depths"])


def test_dataset_reads_correctly_from_dataloader_workers(co3d_tree):
    root, ann_dir, _ = co3d_tree
    repack_from_dir(str(root))
    ds = _dataset(root, ann_dir, "archives")
    single = ds.get_data(seq_name=ds.sequence_list[0], ids=[0, 1])

    class OneSeq(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, i):
            b = ds.get_data(seq_name=ds.sequence_list[0], ids=[0, 1])
            return torch.from_numpy(np.asarray(b["images"])), torch.from_numpy(np.asarray(b["depths"]))

    loader = torch.utils.data.DataLoader(OneSeq(), batch_size=1, num_workers=2)
    seen = 0
    for imgs, depths in loader:
        assert np.array_equal(imgs[0].numpy(), np.asarray(single["images"]))
        assert np.array_equal(depths[0].numpy(), np.asarray(single["depths"]))
        seen += 1
    assert seen == 4


# --------------------------------------------------------------------------- real data
REAL_ROOT = os.environ.get("CO3D_DIR", "/data/CO3D")
REAL_ANN = os.environ.get("CO3D_ANNOTATION_DIR", "/data/CO3D_ann")


NON_SEQUENCE_DIRS = ("set_lists", "eval_batches")


def _real_categories():
    if not os.path.isdir(REAL_ROOT):
        return []
    return [d for d in sorted(os.listdir(REAL_ROOT)) if os.path.isdir(os.path.join(REAL_ROOT, d)) and not d.startswith("_")]


def _real_extracted_sequences(cat, limit=2):
    """Sequence directories that still hold loose frames (none once the repack has run)."""
    d = os.path.join(REAL_ROOT, cat)
    out = []
    for name in sorted(os.listdir(d)):
        if name in NON_SEQUENCE_DIRS or not os.path.isdir(os.path.join(d, name)):
            continue
        if os.path.isdir(os.path.join(d, name, "images")) and os.listdir(os.path.join(d, name, "images")):
            out.append(name)
        if len(out) >= limit:
            break
    return out


@pytest.mark.skipif(not os.path.isdir(REAL_ROOT), reason="the real CO3D dataset is not mounted")
def test_real_sequence_round_trips_byte_for_byte(tmp_path):
    """Pack two real sequences into a scratch copy and compare every byte with the originals."""
    import shutil

    cats = _real_categories()
    if not cats:
        pytest.skip("no categories on disk")
    cat, seqs = None, []
    for c in cats:
        seqs = _real_extracted_sequences(c)
        if seqs:
            cat = c
            break
    if not seqs:
        pytest.skip("no loose extracted sequences left; the dataset is already packed")

    scratch = tmp_path / "CO3D"
    for seq in seqs:  # copy so the real dataset is never touched
        shutil.copytree(os.path.join(REAL_ROOT, cat, seq), scratch / cat / seq)
    originals = {}
    for seq in seqs:
        for sub in ("images", "depths", "depth_masks"):
            d = scratch / cat / seq / sub
            for fn in os.listdir(d):
                originals[f"{cat}/{seq}/{sub}/{fn}"] = (d / fn).read_bytes()

    stats = repack_from_dir(str(scratch))
    assert stats["sequences"] == len(seqs) and stats["frames"] > 0
    store = ArchiveStore(str(scratch))
    for member, data in originals.items():
        assert store.read(member) == data, member

    # and the decoded arrays match what the on-disk readers produce
    member = next(m for m in originals if "/images/" in m)
    assert np.array_equal(decode_image_cv2(store.read(member)), read_image_cv2(str(scratch / member)))
    dm = frame_members(member)["depth"]
    assert np.array_equal(decode_depth(store.read(dm)), read_depth(str(scratch / dm), 1.0))
    store.close()
    print(f"\n  verified {len(originals)} real files from {cat}/{seqs}")


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(REAL_ROOT, "_archive_frames.jsonl")),
    reason="the real CO3D dataset is not packed",
)
def test_real_archives_are_consistent_and_decodable():
    """Sample the packed dataset: manifest agrees with the archives and frames decode."""
    import random

    manifest = read_manifest(REAL_ROOT)
    assert manifest, "manifest is empty"
    store = ArchiveStore(REAL_ROOT)
    rng = random.Random(0)
    sample = rng.sample(sorted(manifest), min(15, len(manifest)))
    frames_checked = 0
    for seq in sample:
        frames = manifest[seq]
        assert frames, f"{seq} has no frames in the manifest"
        archive = os.path.join(REAL_ROOT, archive_relpath(frames[0]))
        assert os.path.isfile(archive), f"missing archive for {seq}"
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
        for frame in rng.sample(frames, min(3, len(frames))):
            members = frame_members(frame)
            assert set(members.values()) <= names, f"{seq}: incomplete frame {frame}"
            img = decode_image_cv2(store.read(members["image"]))
            depth = decode_depth(store.read(members["depth"]))
            mask = decode_mask_gray(store.read(members["depth_mask"]))
            assert img is not None and img.ndim == 3 and img.shape[2] == 3
            assert depth.dtype == np.float32 and depth.shape == img.shape[:2]
            assert mask.shape == img.shape[:2]
            assert np.isfinite(depth).all()
            frames_checked += 1
    store.close()
    print(f"\n  checked {frames_checked} frames across {len(sample)} real sequences")


def test_sequence_split_across_two_chunks_keeps_every_frame(co3d_tree, tmp_path):
    """CO3D chunks are cut by size, so one sequence can arrive in two. Neither half may be lost."""
    root, _, ann = co3d_tree
    seq, cat = "28_001_0001", "tv"
    frames = [f["filepath"] for f in ann[cat][seq]]
    half = len(frames) // 2

    def chunk(path, wanted_frames, subs):
        with zipfile.ZipFile(path, "w") as zf:
            for fp in wanted_frames:
                for kind, member in frame_members(fp).items():
                    if kind in subs:
                        zf.write(root / member, arcname=member)

    # first chunk: images only for the first half; second: everything else
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    chunk(a, frames[:half], {"image"})
    with zipfile.ZipFile(b, "w") as zf:
        for fp in frames:
            for kind, member in frame_members(fp).items():
                if fp in frames[:half] and kind == "image":
                    continue
                zf.write(root / member, arcname=member)

    out = tmp_path / "packed"
    repack_from_zip(str(out), str(a))
    repack_from_zip(str(out), str(b))

    with zipfile.ZipFile(out / f"{cat}/{seq}.zip") as zf:
        names = set(zf.namelist())
    expected = {m for fp in frames for m in frame_members(fp).values()}
    assert expected <= names, f"lost {len(expected - names)} members across the chunk boundary"
    assert set(read_manifest(str(out))[seq]) == set(frames)
    store = ArchiveStore(str(out))
    for fp in frames:  # and the bytes still match the originals
        for member in frame_members(fp).values():
            assert store.read(member) == (root / member).read_bytes()
    store.close()
