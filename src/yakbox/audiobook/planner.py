from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from yakbox.audiobook.manifest import AudiobookManifest
from yakbox.audiobook.sources import (
    Chapter,
    NormalizedDocument,
    Pause,
    SpeechSegment,
    chunk_text,
)
from yakbox.contracts import runtime_metadata


class BuildStage(StrEnum):
    SYNTHESIZE = "synthesize"
    MASTER = "master"
    ENCODE_MP3 = "encode_mp3"
    INSPECT = "inspect"
    RELEASE = "release"


@dataclass(frozen=True, slots=True)
class PlanNode:
    id: str
    stage: BuildStage
    chapter_id: str
    fingerprint: str
    dependencies: tuple[str, ...]
    output: Path
    chunks: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BuildPlan:
    schema_version: int
    target: str
    profile: str
    document_sha256: str
    fingerprint: str
    nodes: tuple[PlanNode, ...]

    def to_dict(self, *, root: Path | None = None) -> dict[str, object]:
        def path_value(path: Path) -> str:
            if root is not None and path.is_relative_to(root):
                return str(path.relative_to(root))
            return str(path)

        return {
            **runtime_metadata("audiobook-plan"),
            "target": self.target,
            "profile": self.profile,
            "document_sha256": self.document_sha256,
            "fingerprint": self.fingerprint,
            "nodes": [
                {
                    "id": node.id,
                    "stage": node.stage.value,
                    "chapter_id": node.chapter_id,
                    "fingerprint": node.fingerprint,
                    "dependencies": list(node.dependencies),
                    "output": path_value(node.output),
                    "chunks": [
                        {
                            "sha256": hashlib.sha256(chunk.encode()).hexdigest(),
                            "characters": len(chunk),
                        }
                        for chunk in node.chunks
                    ],
                }
                for node in self.nodes
            ],
        }


def plan_audiobook(
    manifest: AudiobookManifest,
    document: NormalizedDocument,
    *,
    target_name: str = "default",
    profile_override: str | None = None,
    chapter_selector: str | None = None,
) -> BuildPlan:
    target = manifest.target(target_name)
    profile = manifest.profile(profile_override or target.profile)
    chapters = _select_chapters(document.chapters, chapter_selector)
    profile_payload = json.dumps(asdict(profile), sort_keys=True, default=str)
    nodes: list[PlanNode] = []
    for chapter in chapters:
        chunk_items: list[str] = []
        for item in chapter.segments:
            if isinstance(item, SpeechSegment):
                chunk_items.extend(chunk_text(item.text, target.chunk_chars))
            elif isinstance(item, Pause):
                chunk_items.append(f"__YAKBOX_PAUSE_MS={item.milliseconds}__")
        chunks = tuple(chunk_items)
        speech_text = "\n".join(chunks)
        synthesis_fingerprint = _fingerprint(
            "synthesis-v1",
            chapter.id,
            speech_text,
            profile_payload,
            str(target.chunk_chars),
        )
        raw_output = target.output_root / "raw" / f"{chapter.id}.wav"
        synth_id = f"{chapter.id}:synthesize"
        nodes.append(
            PlanNode(
                id=synth_id,
                stage=BuildStage.SYNTHESIZE,
                chapter_id=chapter.id,
                fingerprint=synthesis_fingerprint,
                dependencies=(),
                output=raw_output,
                chunks=chunks,
            )
        )
        master_fingerprint = _fingerprint(
            "master-v1",
            synthesis_fingerprint,
            str(target.wav_sample_rate),
            str(target.mastering),
        )
        master_id = f"{chapter.id}:master"
        master_output = target.output_root / "mastered" / f"{chapter.id}.wav"
        nodes.append(
            PlanNode(
                id=master_id,
                stage=BuildStage.MASTER,
                chapter_id=chapter.id,
                fingerprint=master_fingerprint,
                dependencies=(synth_id,),
                output=master_output,
            )
        )
        mp3_fingerprint = _fingerprint("mp3-v1", master_fingerprint, target.mp3_bitrate)
        mp3_id = f"{chapter.id}:encode_mp3"
        mp3_output = target.output_root / "release" / "mp3" / f"{chapter.id}.mp3"
        nodes.append(
            PlanNode(
                id=mp3_id,
                stage=BuildStage.ENCODE_MP3,
                chapter_id=chapter.id,
                fingerprint=mp3_fingerprint,
                dependencies=(master_id,),
                output=mp3_output,
            )
        )
        nodes.append(
            PlanNode(
                id=f"{chapter.id}:inspect",
                stage=BuildStage.INSPECT,
                chapter_id=chapter.id,
                fingerprint=_fingerprint("inspect-v1", mp3_fingerprint),
                dependencies=(master_id, mp3_id),
                output=target.output_root / "reports" / f"{chapter.id}.inspection.json",
            )
        )
    plan_fingerprint = _fingerprint(
        "audiobook-plan-v1",
        manifest.book.title,
        target_name,
        profile.name,
        document.sha256,
        *(node.fingerprint for node in nodes),
    )
    return BuildPlan(
        schema_version=1,
        target=target_name,
        profile=profile.name,
        document_sha256=document.sha256,
        fingerprint=plan_fingerprint,
        nodes=tuple(nodes),
    )


def shard_plan(plan: BuildPlan, count: int) -> tuple[tuple[PlanNode, ...], ...]:
    if count < 1:
        raise ValueError("Shard count must be at least 1")
    chapter_ids = list(dict.fromkeys(node.chapter_id for node in plan.nodes))
    assignments: list[list[str]] = [[] for _ in range(min(count, len(chapter_ids)))]
    for index, chapter_id in enumerate(chapter_ids):
        assignments[index % len(assignments)].append(chapter_id)
    return tuple(
        tuple(node for node in plan.nodes if node.chapter_id in chapter_set)
        for chapter_set in (set(items) for items in assignments)
    )


def _select_chapters(
    chapters: tuple[Chapter, ...], selector: str | None
) -> tuple[Chapter, ...]:
    if selector is None:
        return chapters
    selected = tuple(
        chapter
        for chapter in chapters
        if selector == chapter.id
        or selector.casefold() in chapter.title.casefold()
        or selector == str(chapter.order)
    )
    if not selected:
        raise ValueError(f"No chapter matches {selector!r}")
    return selected


def _fingerprint(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()
