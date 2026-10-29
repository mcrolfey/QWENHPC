"""Configurable label space: rename classes, merge classes, drop classes.

Every raw label found in a manifest (image labels and bounding-box labels) is
normalised and mapped onto a *canonical class key*. Each canonical class has a
display name (the exact text the model is trained to answer with) and an
``asbestos`` flag that defines the binary detection target.

Configuration comes from a YAML file (see ``configs/``) and can be modified
from the command line:

    --merge a_cf,a_cp=a_c:"Chrysotile"
    --rename a_am="Amosite (brown asbestos)"
    --drop_class na_of

Merging is many-to-one aliasing: every raw label that mapped to any merged key
now maps to the new key, so merges compose with aliases defined in YAML.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import yaml


def normalise(label: str) -> str:
    """Normalise a raw label: lower-case, collapse whitespace/hyphens to '_'."""
    label = str(label).strip().lower()
    label = re.sub(r"[\s\-/]+", "_", label)
    return re.sub(r"[^a-z0-9_]", "", label)


@dataclass
class ClassDef:
    key: str
    name: str
    asbestos: bool


@dataclass
class LabelSpace:
    classes: list[ClassDef]
    aliases: dict[str, str] = field(default_factory=dict)
    drop: set[str] = field(default_factory=set)
    unknown_policy: str = "error"  # "error" | "drop"
    binary_positive: str = "Yes"
    binary_negative: str = "No"

    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------ build
    @classmethod
    def from_dict(cls, cfg: dict) -> "LabelSpace":
        classes = []
        for key, spec in cfg["classes"].items():
            spec = spec or {}
            classes.append(
                ClassDef(
                    key=normalise(key),
                    name=str(spec.get("name", key.replace("_", " ").capitalize())),
                    asbestos=bool(spec.get("asbestos", not normalise(key).startswith(("non_asbestos", "na_")))),
                )
            )
        binary = cfg.get("binary", {}) or {}
        return cls(
            classes=classes,
            aliases={normalise(k): normalise(v) for k, v in (cfg.get("aliases") or {}).items()},
            drop={normalise(d) for d in (cfg.get("drop") or [])},
            unknown_policy=cfg.get("unknown_policy", "error"),
            binary_positive=str(binary.get("positive", "Yes")),
            binary_negative=str(binary.get("negative", "No")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LabelSpace":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))

    @classmethod
    def from_json(cls, path: str | Path) -> "LabelSpace":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        data["classes"] = [ClassDef(**c) for c in data["classes"]]
        data["drop"] = set(data.get("drop", []))
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        data = asdict(self)
        data["drop"] = sorted(self.drop)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    @classmethod
    def union(cls, spaces: list["LabelSpace"]) -> "LabelSpace":
        """Combine label spaces (e.g. micro + macro for a unified model).

        Classes with the same key, or with the same answer text, become one
        class; the later key is kept as an alias of the earlier one.
        """
        classes: list[ClassDef] = []
        by_key: dict[str, ClassDef] = {}
        by_name: dict[str, ClassDef] = {}
        aliases: dict[str, str] = {}
        for s in spaces:
            for c in s.classes:
                same = by_key.get(c.key) or by_name.get(normalise(c.name))
                if same is None:
                    new = ClassDef(c.key, c.name, c.asbestos)
                    classes.append(new)
                    by_key[c.key], by_name[normalise(c.name)] = new, new
                    continue
                if same.asbestos != c.asbestos:
                    raise ValueError(f"Cannot combine '{c.key}' and '{same.key}': asbestos flags differ")
                if c.key == same.key and normalise(c.name) != normalise(same.name):
                    raise ValueError(f"Class '{c.key}' has different names: '{same.name}' vs '{c.name}'")
                if c.key != same.key:
                    aliases[c.key] = same.key
        canon = {c.key for c in classes}
        for s in spaces:
            for raw, tgt in s.aliases.items():
                tgt = aliases.get(tgt, tgt)
                if raw in aliases and aliases[raw] != tgt:
                    raise ValueError(f"Alias '{raw}' points to different classes in the combined configs")
                if raw not in canon:
                    aliases[raw] = tgt
        if len({(s.binary_positive, s.binary_negative) for s in spaces}) != 1:
            raise ValueError("Combined configs must use the same binary answers")
        return cls(
            classes=classes,
            aliases=aliases,
            drop=set().union(*(s.drop for s in spaces)) - canon,
            unknown_policy="error" if any(s.unknown_policy == "error" for s in spaces) else "drop",
            binary_positive=spaces[0].binary_positive,
            binary_negative=spaces[0].binary_negative,
        )

    @classmethod
    def load(
        cls,
        path: str | Path | list,
        merges: Iterable[str] = (),
        renames: Iterable[str] = (),
        drops: Iterable[str] = (),
    ) -> "LabelSpace":
        """Load YAML/JSON config(s) and apply CLI merge/rename/drop operations.

        Several paths are combined with :meth:`union`.
        """
        paths = [Path(p) for p in path] if isinstance(path, (list, tuple)) else [Path(path)]
        spaces = [cls.from_json(p) if p.suffix == ".json" else cls.from_yaml(p) for p in paths]
        space = spaces[0] if len(spaces) == 1 else cls.union(spaces)
        for spec in merges or ():
            space.merge_from_spec(spec)
        for spec in renames or ():
            space.rename_from_spec(spec)
        for key in drops or ():
            space.drop_class(key)
        return space

    # ------------------------------------------------------------- mutations
    def merge(self, keys: list[str], new_key: str, new_name: str | None = None) -> None:
        keys = [normalise(k) for k in keys]
        new_key = normalise(new_key)
        missing = [k for k in keys if k not in self.keys]
        if missing:
            raise ValueError(f"Cannot merge unknown classes {missing}; known: {self.keys}")
        merged = [c for c in self.classes if c.key in keys]
        flags = {c.asbestos for c in merged}
        if len(flags) != 1:
            raise ValueError(
                f"Refusing to merge asbestos and non-asbestos classes {keys}: "
                "the binary detection target would be undefined."
            )
        name = new_name or " / ".join(c.name for c in merged)
        # The merged class takes the position of the first merged class.
        first = min(i for i, c in enumerate(self.classes) if c.key in keys)
        remaining = [c for c in self.classes if c.key not in keys or c is self.classes[first]]
        idx = remaining.index(self.classes[first])
        remaining[idx] = ClassDef(key=new_key, name=name, asbestos=flags.pop())
        self.classes = remaining
        # Redirect aliases and the old canonical keys themselves.
        self.aliases = {raw: (new_key if tgt in keys else tgt) for raw, tgt in self.aliases.items()}
        for k in keys:
            if k != new_key:
                self.aliases[k] = new_key
        self._validate()

    def rename(self, key: str, new_name: str) -> None:
        key = normalise(key)
        for c in self.classes:
            if c.key == key:
                c.name = new_name
                self._validate()
                return
        raise ValueError(f"Cannot rename unknown class '{key}'; known: {self.keys}")

    def drop_class(self, key: str) -> None:
        key = normalise(key)
        if key not in self.keys:
            raise ValueError(f"Cannot drop unknown class '{key}'; known: {self.keys}")
        self.classes = [c for c in self.classes if c.key != key]
        self.drop.add(key)
        self.drop.update(raw for raw, tgt in self.aliases.items() if tgt == key)
        self.aliases = {raw: tgt for raw, tgt in self.aliases.items() if tgt != key}
        self._validate()

    def merge_from_spec(self, spec: str) -> None:
        """Parse 'a,b,c=new_key[:Display Name]'."""
        if "=" not in spec:
            raise ValueError(f"Bad --merge spec '{spec}', expected 'a,b=new_key[:Name]'")
        left, right = spec.split("=", 1)
        new_key, _, name = right.partition(":")
        self.merge([k for k in left.split(",") if k.strip()], new_key, name.strip() or None)

    def rename_from_spec(self, spec: str) -> None:
        """Parse 'key=Display Name'."""
        if "=" not in spec:
            raise ValueError(f"Bad --rename spec '{spec}', expected 'key=Display Name'")
        key, name = spec.split("=", 1)
        self.rename(key, name.strip())

    # --------------------------------------------------------------- queries
    @property
    def keys(self) -> list[str]:
        return [c.key for c in self.classes]

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.classes]

    def by_key(self, key: str) -> ClassDef:
        for c in self.classes:
            if c.key == key:
                return c
        raise KeyError(key)

    def map(self, raw: str) -> str | None:
        """Map a raw label to a canonical key, or None if it should be dropped."""
        n = normalise(raw)
        if n in self.drop:
            return None
        if n in self.keys:
            return n
        if n in self.aliases:
            return self.aliases[n]
        for c in self.classes:  # display names are implicit aliases
            if normalise(c.name) == n:
                return c.key
        if self.unknown_policy == "drop":
            return None
        raise ValueError(
            f"Unknown label '{raw}' (normalised '{n}'). Add it to 'aliases' or 'drop' "
            f"in the class config, or set unknown_policy: drop. Known classes: {self.keys}"
        )

    def is_asbestos(self, key: str) -> bool:
        return self.by_key(key).asbestos

    def answers(self, task: str) -> list[str]:
        """Candidate answer strings for a task, in class-index order."""
        if task == "binary":
            return [self.binary_negative, self.binary_positive]
        if task == "type":
            return self.names
        raise ValueError(f"Unknown task '{task}'")

    def target_index(self, key: str, task: str) -> int:
        if task == "binary":
            return int(self.is_asbestos(key))
        return self.keys.index(key)

    def target_answer(self, key: str, task: str) -> str:
        return self.answers(task)[self.target_index(key, task)]

    def describe(self) -> str:
        lines = ["Label space:"]
        for i, c in enumerate(self.classes):
            raws = sorted(r for r, t in self.aliases.items() if t == c.key)
            lines.append(
                f"  [{i}] {c.key:<20} -> '{c.name}' (asbestos={c.asbestos})"
                + (f"  aliases: {', '.join(raws)}" if raws else "")
            )
        lines.append(f"  binary answers: negative='{self.binary_negative}', positive='{self.binary_positive}'")
        if self.drop:
            lines.append(f"  dropped labels: {', '.join(sorted(self.drop))}")
        return "\n".join(lines)

    # ------------------------------------------------------------ validation
    def _validate(self) -> None:
        keys = self.keys
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate class keys: {keys}")
        names = [normalise(n) for n in self.names]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate class display names: {self.names}")
        if self.unknown_policy not in ("error", "drop"):
            raise ValueError("unknown_policy must be 'error' or 'drop'")
        if normalise(self.binary_positive) == normalise(self.binary_negative):
            raise ValueError("Binary positive and negative answers must differ")
        for raw, tgt in self.aliases.items():
            if tgt not in keys:
                raise ValueError(f"Alias '{raw}' points to unknown class '{tgt}'")


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_CONFIGS = {
    "microscopic": [CONFIG_DIR / "classes_micro.yaml"],
    "smartphone": [CONFIG_DIR / "classes_macro.yaml"],
    "unified": [CONFIG_DIR / "classes_micro.yaml", CONFIG_DIR / "classes_macro.yaml"],
}


def add_label_args(parser) -> None:
    """Register the shared label-space CLI arguments on an argparse parser."""
    g = parser.add_argument_group("label space")
    g.add_argument("--class_config", nargs="+", default=None,
                   help="YAML (or label_space.json) defining classes, aliases and drops. Several files "
                        "are combined. Default: classes_micro.yaml for microscopic, classes_macro.yaml "
                        "for smartphone, both for unified.")
    g.add_argument("--merge", action="append", default=[], metavar="A,B=NEW[:Name]",
                   help="Merge classes A and B into NEW (repeatable).")
    g.add_argument("--rename", action="append", default=[], metavar="KEY=Name",
                   help="Change the display name (model answer text) of a class (repeatable).")
    g.add_argument("--drop_class", action="append", default=[], metavar="KEY",
                   help="Remove a class and all images labelled with it (repeatable).")


def label_space_from_args(args, modality: str = "unified") -> LabelSpace:
    configs = args.class_config or DEFAULT_CONFIGS[modality]
    return LabelSpace.load(configs, args.merge, args.rename, args.drop_class)
