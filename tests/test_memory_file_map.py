"""Tests that MemoryFileMap preserves file permission bits and mtime end-to-end."""

import base64
import os

from kernelfoundry.eval_pipeline.utils.custom_task_helper import update_block
from kernelfoundry.eval_pipeline.utils.memory_file_map import MemoryFileMap

PAST_MTIME = 1577836800  # 2020-01-01 UTC


def test_encode_decode_to_disk_roundtrip(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    script = src / "build.sh"
    script.write_text("#!/bin/bash\necho hi\n")
    os.chmod(script, 0o755)
    os.utime(script, (PAST_MTIME, PAST_MTIME))
    plain = src / "notes.txt"
    plain.write_text("hello\n")
    os.chmod(plain, 0o644)

    src_map = MemoryFileMap()
    src_map.from_disk(src)

    # Mimic the Celery/DB boundary: encode -> decode into a fresh map.
    dst_map = MemoryFileMap()
    dst_map.decode(src_map.encode())

    out = tmp_path / "out"
    dst_map.to_disk(out)

    extracted_script = out / "build.sh"
    assert abs(extracted_script.stat().st_mtime - PAST_MTIME) < 2

    extracted_plain = out / "notes.txt"

    # Windows os.chmod only toggles the read-only bit, so mode bits don't round-trip exactly.
    if os.name != "nt":
        assert extracted_script.stat().st_mode & 0o777 == 0o755
        assert extracted_plain.stat().st_mode & 0o777 == 0o644


def test_tarball_roundtrip_preserves_mode_and_mtime(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    script = src / "run.sh"
    script.write_text("x")
    os.chmod(script, 0o755)
    os.utime(script, (PAST_MTIME, PAST_MTIME))

    src_map = MemoryFileMap()
    src_map.from_disk(src)

    dst_map = MemoryFileMap()
    dst_map.from_tarball(tarball_bytes=src_map.to_tarball())

    assert abs(dst_map.meta_map["run.sh"]["mtime"] - PAST_MTIME) < 2

    # Windows os.chmod only toggles the read-only bit, so the captured mode isn't 0o755 there.
    if os.name != "nt":
        assert dst_map.meta_map["run.sh"]["mode"] == 0o755


def test_decode_legacy_string_format(tmp_path):
    legacy = {"a.txt": base64.b64encode(b"hi").decode("utf-8")}
    m = MemoryFileMap()
    m.decode(legacy)

    assert m["a.txt"] == b"hi"
    assert m.meta_map["a.txt"]["mode"] == 0o644

    out = tmp_path / "out"
    m.to_disk(out)
    assert (out / "a.txt").read_text() == "hi"


def test_remove_root_dir_preserves_meta():
    m = MemoryFileMap()
    m["root/build.sh"] = b"x"
    m.meta_map["root/build.sh"]["mode"] = 0o755
    m["root/sub/a.txt"] = b"y"

    m.remove_root_dir()

    assert "build.sh" in m.file_map
    assert "sub/a.txt" in m.file_map
    assert m.meta_map["build.sh"]["mode"] == 0o755


def test_update_block_preserves_exec_bit():
    content = "a\n# [EVOLVE_START]\nold\n# [EVOLVE_END]\nb\n"
    m = MemoryFileMap()
    m["kernel.sh"] = content.encode()
    m.meta_map["kernel.sh"]["mode"] = 0o755

    update_block(m, key="EVOLVE", path="kernel.sh", content="new")

    assert m.meta_map["kernel.sh"]["mode"] == 0o755
    assert b"new" in (m["kernel.sh"] or b"")


def test_task_hash_id_stable_across_encode_format():
    from kernelfoundry.eval_pipeline.database.tables import Task as DBTask

    b64 = base64.b64encode(b"#!/bin/bash\necho hi\n").decode("utf-8")
    legacy_encoded = {"build.sh": b64}
    new_encoded = {"build.sh": {"content": b64, "mode": 0o755, "mtime": 123.0}}

    legacy_task = DBTask()
    legacy_task.generate_hash_id(legacy_encoded)
    new_task = DBTask()
    new_task.generate_hash_id(new_encoded)

    # mode/mtime must not affect the id, and the new format must match the legacy id.
    assert legacy_task.id == new_task.id
